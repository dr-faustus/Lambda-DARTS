
gpu=0
dataset='svhn'
epochs=50
search_space='s4'
lambda_=0.25
epsilon_0=0.001
reg_type='corr'
arch_learning_rate=0.0003


seed=0
cd ./sota/cnn && nohup python train_search.py --batch_size 96 --dataset $dataset --corr_regularization $reg_type --lambda_ $lambda_ --epsilon_0 $epsilon_0 --epochs $epochs --arch_learning_rate $arch_learning_rate --gpu $gpu --seed $seed --search_space $search_space > ../../search-$search_space-$dataset-seed-$seed-lambda-$lambda_-epsilon-$epsilon_0-$reg_type-epochs-$epochs.log 2>&1 &
