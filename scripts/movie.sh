set -e
run_idx=$1
gpu=$2


cmd="python train_no_test.py --dataset_mode=movie --gpu_ids=$gpu
--input_dim=130 --model=movie 
--warmup --warmup_lr=1e-7 --warmup_epoch=5
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=0
--output_dim=6 --cls_layers=128,128 --nhead=4 --num_layers=6 --dim_feedforward=512
--niter=30 --niter_decay=30 --verbose --beta1=0.9 --init_type normal
--batch_size=256 --lr=5e-4 --run_idx=$run_idx --enc_channel=256 --no_test
--name=movie_cnn1d_onlyKD --suffix=layer{num_layers}_nhead{nhead}_ffn{dim_feedforward}_run{run_idx}"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done