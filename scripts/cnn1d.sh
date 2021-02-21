set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_raw_sig --model=cnn_1d --gpu_ids=$gpu
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--output_dim=4 --cls_layers=128,128 --hidden_size=128 --embd_method=maxpool
--niter=20 --niter_decay=30 --verbose --beta1=0.9 --init_type kaiming
--batch_size=128 --lr=5e-4 --run_idx=$run_idx --enc_hidden_size=32 --bidirection
--name=cnn1d_simple --suffix=enc{enc_hidden_size}_bi{bidirection}_run{run_idx}
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done