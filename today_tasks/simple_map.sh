set -e
model_type=$1
run_idx=$2
mse_weight=$3
cycle_weight=$4
gpu_ids=$5
for i in `seq 1 1 10`;
do

cmd="python train_baseline.py --dataset_mode=iemocap_deep --model=simple_mapping --gpu_ids=$gpu_ids
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=2
--acoustic_ft_type=A --lexical_ft_type=L --visual_ft_type=V
--mapping_layers=128,100,72
--niter=20 --niter_decay=30 --verbose
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=$run_idx
--mse_weight=$mse_weight --cycle_weight=$cycle_weight
--model_type=$model_type
--name=simple_map_cycle --suffix={model_type}_mse{mse_weight}_cycle{cycle_weight}_run{run_idx}
--cvNo=$i"

# --name=ef_A --suffix=Adnn{mid_layers_a}_Vlstm{hidden_size}_{embd_size_v}_Lcnn{embd_size_l}_fusion{mid_layers_fusion}run{run_idx}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done