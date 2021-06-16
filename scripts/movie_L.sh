set -e
dataset=$1
name=$2
run_idx=$3
gpu=$4
mse1=$5
mse2=$6 
mse3=$7
mse4=$8
ratio=$9

cmd="python train_baseline.py --dataset_mode=movie --gpu_ids=$gpu
--input_dim=130 --model=movie_L
--db_dir=/data4/lrc/movie_dataset/dbs/$dataset
--warmup --warmup_lr=1e-7 --warmup_epoch=5 --ratio=$ratio
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=16
--output_dim=5 --cls_layers=128,128 --nhead=4 --num_layers=6 --dim_feedforward=512
--kd_weight=1.0 --mse_weight1=$mse1 --mse_weight2=$mse2 --mse_weight3=$mse3 --mse_weight4=$mse4
--niter=25 --niter_decay=25 --beta1=0.9 --init_type normal
--batch_size=256 --lr=1e-4 --run_idx=$run_idx --enc_channel=256 --no_test
--name=$name --suffix={ratio}_enc{enc_channel}_layer{num_layers}_nhead{nhead}_ffn{dim_feedforward}_run{run_idx}"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

# bash scripts/movie_L.sh v4 XV4_10_4layer 2 2 0.1 0.1 0.2 0.3
# bash scripts/movie_L.sh v4 XV4_10_4layer 1 3 0.1 0.1 0.2 0.3

# bash scripts/movie_L.sh v4 XV4_4layer 1 5 0.1 0.1 0.2 0.3 1-2
# bash scripts/movie_L.sh v4 XV4_4layer 1 6 0.1 0.1 0.2 0.3 1-3
# bash scripts/movie_L.sh v4 XV4_4layer 1 7 0.1 0.1 0.2 0.3 1-5
# bash scripts/movie_L.sh v4 XV4_4layer_noKD 1 7 0.1 0.1 0.2 0.3 full
# bash scripts/movie_L.sh v2 XV2_4layer 1 7 0.1 0.1 0.2 0.3 full