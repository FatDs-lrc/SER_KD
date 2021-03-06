set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_raw_sigslice --model=sincnet_lstm --gpu_ids=$gpu
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=4
--output_dim=4 --cls_layers=128,128 --hidden_size=128 --embd_method=maxpool
--niter=20 --niter_decay=20 --verbose --beta1=0.9 --init_type kaiming --norm_sig=None
--batch_size=32 --lr=1e-4 --run_idx=$run_idx --bidirection
--name=sincnet_lstm --suffix=norm-{norm_sig}_bi{bidirection}_run{run_idx}
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done