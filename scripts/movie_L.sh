set -e
dataset=$1
name=$2
run_idx=$3
gpu=$4
mse1=$5
mse2=$6 
mse3=$7
mse4=$8


cmd="python train_baseline.py --dataset_mode=movie --gpu_ids=$gpu
--input_dim=130 --model=pooler
--db_dir=/data4/lrc/movie_dataset/dbs/$dataset
--warmup --warmup_lr=1e-7 --warmup_epoch=5
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=16
--output_dim=5 --cls_layers=128,128 --nhead=4 --num_layers=6 --dim_feedforward=512
--kd_weight=0.0 --mse_weight1=$mse1 --mse_weight2=$mse2 --mse_weight3=$mse3 --mse_weight4=$mse4
--niter=25 --niter_decay=25 --beta1=0.9 --init_type normal
--batch_size=256 --lr=1e-4 --run_idx=$run_idx --enc_channel=128 --no_test
--name=$name --suffix=enc{enc_channel}_layer{num_layers}_nhead{nhead}_ffn{dim_feedforward}_run{run_idx}"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

# --resume --resume_dir=checkpoints/KD_v2_4MSE_2layer_enc128_layer6_nhead4_ffn512_run1/0 --resume_epoch=10
# bash scripts/movie_L.sh v2 KD_v2_4MSE 1 0 0.1 0.1 0.2 0.3
# bash scripts/movie_L.sh v2 KD_v2_4MSE_2layer 1 1 0.0 0.0 0.1 0.2
# bash scripts/movie_L.sh v4 CMKD_4MSE 1 2 0.1 0.1 0.2 0.3
# bash scripts/movie_L.sh v4 CMKD_4MSE_3layer 1 3 0.0 0.1 0.2 0.3
# bash scripts/movie_L.sh v4 CMKD_4MSE_2layer 1 4 0.0 0.0 0.2 0.3
# bash scripts/movie_L.sh v4 CMKD_4MSE_1layer 1 5 0.0 0.0 0.0 0.3
# bash scripts/movie_L.sh v4 CMKD_4MSE_onlyKD 1 6 0.0 0.0 0.0 0.0
# bash scripts/movie_L.sh v2 KD_v2_onlyKD 1 7 0.0 0.0 0.0 0.0


# bash scripts/movie_L.sh v2 KD_v2_4MSE 2 0 0.1 0.1 0.2 0.3
# bash scripts/movie_L.sh v2 KD_v2_4MSE_2layer 2 1 0.0 0.0 0.1 0.2

# resume
# bash scripts/movie_L.sh v4 CMKD_4MSE_4layer_RSM_sample 1 2 0.1 0.1 0.2 0.3
# bash scripts/movie_L.sh v4 CMKD_4MSE_3layer_RSM_sample 1 3 0.0 0.1 0.2 0.3
# bash scripts/movie_L.sh v4 CMKD_4MSE_2layer_RSM_sample 1 4 0.0 0.0 0.2 0.3
# bash scripts/movie_L.sh v4 CMKD_4MSE_1layer_RSM_sample 1 5 0.0 0.0 0.0 0.3
# bash scripts/movie_L.sh v4 CMKD_4MSE_0layer_RSM_sample 1 5 0.0 0.0 0.0 0.0

# 保留checkpoints
# bash scripts/movie_L.sh v4 SAMPLE2_CMKD_4MSE_4layer_RSM 1 2 0.1 0.1 0.2 0.3
# bash scripts/movie_L.sh v4 SAMPLE2_CMKD_4MSE_3layer_RSM_sample 1 3 0.0 0.1 0.2 0.3
# bash scripts/movie_L.sh v4 SAMPLE2_CMKD_4MSE_2layer_RSM_sample 1 4 0.0 0.0 0.2 0.3
# bash scripts/movie_L.sh v4 SAMPLE2_CMKD_4MSE_1layer_RSM_sample 1 5 0.0 0.0 0.0 0.3
# bash scripts/movie_L.sh v4 SAMPLE2_CMKD_4MSE_0layer_RSM_sample 1 5 0.0 0.0 0.0 0.0

# XXXXX
# --resume --resume_dir=checkpoints/YYY_4layer_RSM_enc128_layer6_nhead4_ffn512_run2/0 --resume_epoch=16
# bash scripts/movie_L.sh v2 YYY_4layer_RSM 2 0 0.1 0.1 0.2 0.3
# bash scripts/movie_L.sh v4 XXX_4layer_RSM 2 0 0.1 0.1 0.2 0.3

# add utterance-level pooler
# bash scripts/movie_L.sh v2 PoolerX_no_kd_4layer 2 0 0.1 0.1 0.2 0.3