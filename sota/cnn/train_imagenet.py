from torch.utils.tensorboard import SummaryWriter
import argparse
import glob
import logging as text_logger
import sys
sys.path.insert(0, '../../')
import time

import numpy as np
import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
import torchmetrics as tm
from pytorch_lightning.callbacks import ModelCheckpoint

from sota.cnn import utils
from sota.cnn.model_imagenet import NetworkImageNet  as Network
import sota.cnn.genotypes as genotypes


parser = argparse.ArgumentParser("imagenet")
parser.add_argument('--workers', type=int, default=16, help='number of workers to load dataset')
parser.add_argument('--data', type=str, default='/mounts/work/ayyoob/sajjad/SmoothDARTS/data/imagenet/ILSVRC', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.5, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=100, help='report frequency')
parser.add_argument('--epochs', type=int, default=250, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=48, help='num of init channels')
parser.add_argument('--layers', type=int, default=14, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--drop_path_prob', type=float, default=0, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random_ws seed')
parser.add_argument('--arch', type=str, default='corrdarts_corr_3', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
parser.add_argument('--label_smooth', type=float, default=0.1, help='label smoothing')
parser.add_argument('--lr_scheduler', type=str, default='linear', help='lr scheduler, linear or cosine')
parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
parser.add_argument('--num_gpu', type=int, default=8)
args, unparsed = parser.parse_known_args()

CLASSES = 1000

class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, epsilon):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (-targets * log_probs).mean(0).sum()
        return loss


def adjust_lr(optimizer, epoch):
    # Smaller slope for the last 5 epochs because lr * 1/250 is relatively large
    if args.epochs -  epoch > 5:
        lr = args.learning_rate * (args.epochs - 5 - epoch) / (args.epochs - 5)
    else:
        lr = args.learning_rate * (args.epochs - epoch) / ((args.epochs - 5) * 5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr        


def prepare_log():
    utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
    log_format = '%(asctime)s %(message)s'
    text_logger.basicConfig(stream=None, level=text_logger.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = text_logger.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(text_logger.Formatter(log_format))
    text_logger.getLogger().addHandler(fh)
    text_logger.getLogger().propagate = False
    



class ImageNetNetwork(pl.LightningModule):
    def __init__(self,C, num_classes, layers, auxiliary, genotype):
        super().__init__()
        self.model = Network(C, num_classes, layers, auxiliary, genotype)
        self.criterion = nn.CrossEntropyLoss()
        self.criterion_smooth = CrossEntropyLabelSmooth(CLASSES, args.label_smooth)
        self.current_lr =  None
        self.learning_rate = args.learning_rate
        self.genotype = genotype

        self.train_acc_top1 = tm.Accuracy(top_k=1)
        self.train_acc_top5 = tm.Accuracy(top_k=5)
        self.valid_acc_top1 = tm.Accuracy(top_k=1)
        self.valid_acc_top5 = tm.Accuracy(top_k=5)

        self.best_acc_top1 = 0
        self.best_acc_top5 = 0


        
    def forward(self, input):
        return self.model(input)

    def on_fit_start(self):
        if self.global_rank==0:
            prepare_log()
            text_logger.info("args = %s", args)
            text_logger.info(self.genotype)
            text_logger.info("param size = %fMB", utils.count_parameters_in_MB(self.model))

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
        self.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )
        if args.lr_scheduler == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
            return [optimizer], [scheduler]
        else:
            return optimizer

    def training_step(self, train_batch, batch_idx):
        input, target = train_batch
        logits, logits_aux = self(input)
        loss = self.criterion_smooth(logits, target)
        if args.auxiliary:
            loss_aux = self.criterion_smooth(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        
        self.train_acc_top1(logits,target)
        self.train_acc_top5(logits,target)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        self.log("train_acc1", self.train_acc_top1, on_step=True, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc5", self.train_acc_top5, on_step=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        input, target = val_batch
        logits, _ = self(input)
        loss = self.criterion(logits, target)

        self.valid_acc_top1(logits,target)
        self.valid_acc_top5(logits,target)
        self.log("valid_loss", loss, on_step=True, on_epoch=True)
        self.log("valid_acc1", self.valid_acc_top1, on_step=True, prog_bar=True, on_epoch=True)
        self.log("valid_acc5", self.valid_acc_top5, on_step=True, on_epoch=True)
        return loss

    def on_train_epoch_start(self):
        if args.lr_scheduler == 'linear':
            self.current_lr = adjust_lr(self.optimizers(), self.current_epoch)
        else:
            self.current_lr = self.lr_schedulers().get_lr()[0]
        
        if self.global_rank==0: 
            text_logger.info('Epoch: %d lr %e', self.current_epoch, self.current_lr)

        if self.current_epoch < 5 and args.batch_size > 256:
            for param_group in self.optimizers().param_groups:
                param_group['lr'] = self.learning_rate * (self.current_epoch + 1) / 5.0
            if self.global_rank==0:
                text_logger.info('Warming-up Epoch: %d, LR: %e', self.current_epoch, self.learning_rate * (self.current_epoch + 1) / 5.0)

        self.model.drop_path_prob = args.drop_path_prob * self.current_epoch / args.epochs

    def on_train_epoch_end(self):
        train_acc_top1 = self.train_acc_top1.compute().data
        if self.global_rank==0:
            text_logger.info('Train_acc: %f', train_acc_top1*100)
        self.train_acc_top1.reset()

    def on_validation_epoch_end(self):
        valid_acc_top1= self.valid_acc_top1.compute().data
        valid_acc_top5= self.valid_acc_top5.compute().data
        
        if valid_acc_top5 > self.best_acc_top5:
            self.best_acc_top5 = valid_acc_top5
        if valid_acc_top1 > self.best_acc_top1:
            self.best_acc_top1 = valid_acc_top1

        if self.global_rank==0:
            text_logger.info('Valid_acc_top1: %f', valid_acc_top1*100)
            text_logger.info('Valid_acc_top5: %f', valid_acc_top5*100)
            text_logger.info('Best_acc_top1: %f', self.best_acc_top1*100)
            text_logger.info('Best_acc_top5: %f', self.best_acc_top5*100)
            text_logger.info('End of Epoch: %d', self.current_epoch)

        
        # self.valid_acc_top1.reset()
        # self.valid_acc_top5.reset()
        

class ImageNetDataModule(pl.LightningDataModule):

  def setup(self, stage):
    traindir = os.path.join(args.data, 'Data/CLS-LOC/train')
    validdir = os.path.join(args.data, 'Data/CLS-LOC/val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    self.train_data = dset.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.2),
            transforms.ToTensor(),
            normalize,
        ]))
    self.valid_data = dset.ImageFolder(
        validdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

  def train_dataloader(self):
    return torch.utils.data.DataLoader(self.train_data, batch_size=int(args.batch_size/args.num_gpu), shuffle=True, pin_memory=True, num_workers=args.workers)

  def val_dataloader(self):
    return torch.utils.data.DataLoader(self.valid_data, batch_size=int(args.batch_size/args.num_gpu), shuffle=False, pin_memory=True, num_workers=args.workers)

def main():
    args.save = '../../experiments/sota/imagenet/eval-{}-{}-{}-{}'.format(
    args.save, time.strftime("%Y%m%d-%H%M%S"), args.arch, args.seed)
    if args.auxiliary:
        args.save += '-auxiliary-' + str(args.auxiliary_weight)
    args.save += '-' + str(np.random.randint(10000))
    
    
    if args.seed is not None:
        pl.seed_everything(args.seed)

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.save,'checkpoints'), 
        save_top_k=1, 
        monitor="valid_acc1_epoch",
        # auto_insert_metric_name=True,
        save_last=True,
        mode='max',
        filename="{epoch:03d}-{valid_acc1_epoch:.3f}"
    )

    trainer = pl.Trainer(
        devices=args.num_gpu,
        accelerator="gpu",
        strategy=DDPPlugin(find_unused_parameters=False),
        benchmark=True,
        max_epochs=args.epochs,
        gradient_clip_val = args.grad_clip,
        log_every_n_steps= args.report_freq,
        callbacks=[checkpoint_callback],
        sync_batchnorm=True
    )
    genotype = eval("genotypes.%s" % args.arch)
    model = ImageNetNetwork(args.init_channels, CLASSES, args.layers, args.auxiliary, genotype)
    data_module = ImageNetDataModule()
    trainer.fit(model, data_module)

    print(checkpoint_callback.best_model_path)
    print(checkpoint_callback.best_model_score)

if __name__ == '__main__':
    main()