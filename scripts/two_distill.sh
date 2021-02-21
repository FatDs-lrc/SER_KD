set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 10`;
do
cmd="python train_baseline.py --dataset_mode=iemocap_10fold --model=two_distill
--gpu_ids=$gpu --log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--input_dim_a=130 --embd_size_a=128 --embd_method_a=maxpool
--input_dim_v=342 --embd_size_v=128  --embd_method_v=maxpool
--input_dim_l=1024 --embd_size_l=128
--T_modality=L --T_checkpoint=checkpoints/early_fusion_L_run1
--S_modality=A --S_checkpoint=checkpoints/early_fusion_A_run2
--temperature=2 --ce_weight=1.0 --kd_weight=1.0 --mse_weight=0.5
--output_dim=4 --cls_layers=128,128 --dropout_rate=0.3
--niter=20 --niter_decay=20 --verbose --beta1=0.9 --init_type kaiming
--batch_size=256 --lr=5e-4 --run_idx=$run_idx
--name=two_distill --suffix=T{T_modality}_S{S_modality}_run{run_idx}
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done