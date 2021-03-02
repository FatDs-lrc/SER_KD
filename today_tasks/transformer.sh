set -e
num_layers=$1
embd_method=$2
run_idx=$3
gpu=$4


for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --model=transformer --gpu_ids=$gpu
--warmup --warmup_lr=1e-6 --warmup_epoch=5
--dataset_mode=audio --ft_type=comparE_downsampled --input_dim=130 
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=0
--output_dim=4 --cls_layers=128,128 --affine_dim=256
--nhead=4 --num_layers=$num_layers --dim_feedforward=256 --embd_method=$embd_method
--niter=20 --niter_decay=20 --verbose --beta1=0.9 --init_type normal
--batch_size=256 --lr=5e-4 --run_idx=$run_idx
--name=Transformer_comparE --suffix=embd-{embd_method}_num-layer{num_layers}_nhead{nhead}_dim-feedforward{dim_feedforward}_lr-{lr}_run{run_idx} 
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done