set -e
run_idx=$1
gpu=$2

# --warmup --warmup_lr=1e-7 --warmup_epoch=5
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --model=comparE_cnn_lstm --gpu_ids=$gpu
--dataset_mode=audio --ft_type=comparE_raw --input_dim=130 
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=4
--output_dim=4 --cls_layers=128,128 --hidden_size=128 --embd_method=maxpool
--niter=20 --niter_decay=20 --verbose --beta1=0.9 --init_type normal
--batch_size=256 --lr=5e-4 --run_idx=$run_idx --enc_channel=64 --bidirection 
--name=comparE_resnet18_no_warmup --suffix=enc{enc_channel}_bi{bidirection}_lr-{lr}_run{run_idx} 
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done