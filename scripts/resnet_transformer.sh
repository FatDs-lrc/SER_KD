set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=audio --gpu_ids=$gpu
--ft_type=comparE_raw --input_dim=130 --model=resnet_transformer 
--warmup --warmup_lr=1e-8 --warmup_epoch=5
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=4
--output_dim=4 --cls_layers=128,128 --nhead=4 --num_layers=2 --dim_feedforward=256
--niter=30 --niter_decay=20 --verbose --beta1=0.9 --init_type normal
--batch_size=128 --lr=2e-4 --run_idx=$run_idx --enc_channel=128
--name=resnet18_Transformer --suffix=num-layer{num_layers}_nhead{nhead}_dim-feedforward{dim_feedforward}_lr-{lr}_run{run_idx} 
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done