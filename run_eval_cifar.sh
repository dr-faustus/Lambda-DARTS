
gpu=0
drop_path=0.2
arch=svhn_s3_corrdarts_corr_
dataset='svhn'

seed=0
cd ./sota/cnn && nohup python train.py --init_channels 16 --layers 8 --dataset $dataset --drop_path_prob $drop_path --auxiliary --cutout --arch $arch --gpu $gpu --seed $seed > ../../eval-$dataset-dp-$drop_path-seed-$seed-arch-$arch.log 2>&1 &

