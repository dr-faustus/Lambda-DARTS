#!/bin/bash
# bash ./scripts-search/algos/PCDARTS.sh cifar10 -1
echo script name: $0
echo $# arguments
if [ "$#" -ne 2 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 2 parameters for dataset and seed"
  exit 1
fi
#if [ "$TORCH_HOME" = "" ]; then
#  echo "Must set TORCH_HOME envoriment variable for data dir saving"
#  exit 1
#else
#  echo "TORCH_HOME : $TORCH_HOME"
#fi

dataset=$1
seed=$2
channel=16
num_cells=5
max_nodes=4
space=nas-bench-102

#if [ "$dataset" == "cifar10" ] || [ "$dataset" == "cifar100" ]; then
#  data_path="$TORCH_HOME/cifar.python"
#else
#  data_path="$TORCH_HOME/cifar.python/ImageNet16"
#fi

data_path="/mounts/work/ayyoob/sajjad/data"

save_dir=./output/search-cell-${space}/PCDARTS-${dataset}

OMP_NUM_THREADS=4 python ./exps/algos/PCDARTS.py \
	--save_dir ${save_dir} --max_nodes ${max_nodes} --channel ${channel} --num_cells ${num_cells} \
	--dataset ${dataset} --data_path ${data_path} \
	--search_space_name ${space} --corr_regularization --lambda 0.125 --epsilon_0 0.0001 --dataset cifar10 \
	--arch_nas_dataset ${data_path}/NAS-Bench-201-v1_0-e61699.pth \
	--arch_learning_rate 0.0006 --arch_weight_decay 0.001 \
	--workers 4 --print_freq 200 --rand_seed ${seed}
