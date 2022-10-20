import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import traceback
sys.path.insert(0, '../../')
from sota.rnn.genotypes import PRIMITIVES, STEPS, CONCAT, Genotype
from torch.autograd import Variable
from collections import namedtuple
from sota.rnn.model import DARTSCell, RNNModel


class DARTSCellSearch(DARTSCell):

  def __init__(self, ninp, nhid, dropouth, dropoutx):
    super(DARTSCellSearch, self).__init__(ninp, nhid, dropouth, dropoutx, genotype=None)
    self.bn = nn.BatchNorm1d(nhid, affine=False)
    self.weights_ = []

  def cell(self, x, h_prev, x_mask, h_mask, pert=None):
    s0 = self._compute_init_state(x, h_prev, x_mask, h_mask)
    s0 = self.bn(s0)
    
    probs = F.softmax(self.weights, dim=-1)
    self.weights_ = []
    if self.training:
      probs.retain_grad()
      self.weights_.append(probs)
    if pert is not None:
      probs = probs - pert

    offset = 0
    states = s0.unsqueeze(0)
    for i in range(STEPS):
      if self.training:
        masked_states = states * h_mask.unsqueeze(0)
      else:
        masked_states = states
      ch = masked_states.view(-1, self.nhid).mm(self._Ws[i]).view(i+1, -1, 2*self.nhid)
      c, h = torch.split(ch, self.nhid, dim=-1)
      c = c.sigmoid()

      s = torch.zeros_like(s0)
      for k, name in enumerate(PRIMITIVES):
        if name == 'none':
          continue
        fn = self._get_activation(name)
        unweighted = states + c * (fn(h) - states)
        s += torch.sum(probs[offset:offset+i+1, k].unsqueeze(-1).unsqueeze(-1) * unweighted, dim=0)
      s = self.bn(s)
      states = torch.cat([states, s.unsqueeze(0)], 0)
      offset += i+1
    output = torch.mean(states[-CONCAT:], dim=0)
    return output


class RNNModelSearch(RNNModel):

    def __init__(self, *args,_lambda=0.1,_eps_0=0.05):
        super(RNNModelSearch, self).__init__(*args, cell_cls=DARTSCellSearch, genotype=None)
        self._args = args
        self._initialize_arch_parameters()

        self.eps_0 = _eps_0
        self.lambda_ = _lambda
        self.eps = 0.

    def new(self):
        model_new = RNNModelSearch(*self._args)
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new

    def _initialize_arch_parameters(self):
      k = sum(i for i in range(1, STEPS+1))
      weights_data = torch.randn(k, len(PRIMITIVES)).mul_(1e-3)
      self.weights = Variable(weights_data.cuda(), requires_grad=True)
      self._arch_parameters = [self.weights]
      for rnn in self.rnns:
        rnn.weights = self.weights

    def arch_parameters(self):
      return self._arch_parameters
    
  

    def _loss(self, hidden, input, target):
      log_prob, hidden_next = self(input, hidden, return_h=False, redrop=True)
      loss = nn.functional.nll_loss(log_prob.view(-1, log_prob.size(2)), target)
      return loss, hidden_next

    def genotype(self):

      def _parse(probs):
        gene = []
        start = 0
        for i in range(STEPS):
          end = start + i + 1
          W = probs[start:end].copy()
          j = sorted(range(i + 1), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[0]
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
          start = end
        return gene

      gene = _parse(F.softmax(self.weights, dim=-1).data.cpu().numpy())
      genotype = Genotype(recurrent=gene, concat=range(STEPS+1)[-CONCAT:])
      return genotype


    def get_arch_grads(self):
        weights = []
        for l, rnn in enumerate(self.rnns):
            weights = weights + rnn.weights_

        grads = [w.grad.data.clone().detach().reshape(-1) for w in weights]
        return grads
    
    def get_perturbations(self):
        # get layer gradients
        grads_normal = self.get_arch_grads()
        alpha_shape = self.arch_parameters()[0].shape

        def get_perturbation_for_cell(grad_list):
            # function for estimating the perturbation vectors
            weight = 1 / ((len(grad_list) * (len(grad_list) - 1)) / 2)
            identity = torch.eye(grad_list[0].shape[0]).cuda()
            norms = [g.norm(p=2.0) for g in grad_list]
            normalized_grads = [g / norm for g, norm in zip(grad_list, norms)]
            sum_normalized_grads = sum(normalized_grads)
            P_l = [(1 / norm) * (identity - torch.ger(normalized_g, normalized_g))
                   for normalized_g, norm in zip(normalized_grads, norms)]
            u_l = [weight * P @ sum_normalized_grads for P in P_l]
            return u_l

        pert_normal = get_perturbation_for_cell(grads_normal)
        if torch.cat(pert_normal, dim=0).norm(p=2.0).item() > 0:
            self.eps = self.eps_0 / torch.cat(pert_normal, dim=0).norm(p=2.0).item()
        else:
            self.eps = 0

        idx = 0
        pert = []
        # get perturbation for each layer
        for rnn in self.rnns:
            if isinstance(rnn, DARTSCellSearch):
                pert.append(pert_normal[idx].reshape(alpha_shape) * self.eps)
                idx += 1

        return pert

    def get_reg_grads(self, forward_grads, backward_grads):
        reg_grad = [(f - b).div_(2 * self.eps) for f, b in zip(forward_grads, backward_grads)]
        for idx, (name, param) in enumerate(self.named_parameters()):
            param.grad.data.add_(self.lambda_ * reg_grad[idx])

    def get_corr(self):
        grads_normal = self.get_arch_grads()

        def corr(x):

            res = []
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    res.append(
                        (torch.dot(x[i], x[j]) / (x[i].norm(p=2.0) * x[j].norm(p=2.0))).item())
            return sum(res) / len(res)

        def sign_corr(x):
            res = []
            for i in range(len(x)):
                for j in range(i + 1, len(x)):
                    res.append(torch.dot(torch.sign(x[i]), torch.sign(x[j])) / (
                            torch.sign(x[i]).norm(p=2.0).item() * torch.sign(x[j]).norm(p=2.0).item()))
            return sum(res) / len(res)

        return corr(grads_normal), sign_corr(grads_normal)


