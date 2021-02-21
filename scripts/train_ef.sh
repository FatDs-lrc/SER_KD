set -e
modality=$1
run_idx=$2
gpu=$3


for i in `seq 1 1 10`;
do
cmd="python train_baseline.py --dataset_mode=iemocap_10fold --model=early_fusion
--gpu_ids=$gpu --modality=$modality
--V_type=denseface_torch
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--input_dim_a=130 --embd_size_a=128 --embd_method_a=maxpool
--input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
--input_dim_l=1024 --embd_size_l=128
--output_dim=4 --cls_layers=128,128 --dropout_rate=0.3
--niter=20 --niter_decay=20 --verbose --beta1=0.9 --init_type kaiming
--batch_size=256 --lr=5e-4 --run_idx=$run_idx
--name=early_fusion --suffix={modality}_run{run_idx}
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done