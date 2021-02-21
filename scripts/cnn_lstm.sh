set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_10fold 
--A_type=fbank --hidden_size=256 --rnn_layers=1
--model=vgglstm_audio --gpu_ids=$gpu
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--output_dim=4 --cls_layers=128,128
--niter=20 --niter_decay=30 --beta1=0.9 --init_type kaiming
--batch_size=128 --lr=1e-2 --run_idx=$run_idx
--name=vgg_lstm --suffix=run{run_idx}
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done