set -e
dataset=$1
name=$2
run_idx=$3
gpu=$4
kd_weight=$5
utt_weight=$6
word_weight=$7


cmd="python train_baseline.py --dataset_mode=movie_lmdb --gpu_ids=$gpu
--input_dim=130 --model=pooler
--db_dir=/data4/lrc/movie_dataset/dbs/$dataset
--warmup --warmup_lr=1e-7 --warmup_epoch=5
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=16
--output_dim=5 --cls_layers=128,128 --nhead=4 --num_layers=6 --dim_feedforward=512
--kd_weight=$kd_weight --word_weight=$word_weight --utt_weight=$utt_weight
--niter=25 --niter_decay=25 --beta1=0.9 --init_type normal
--batch_size=256 --lr=1e-4 --run_idx=$run_idx --enc_channel=256 --no_test
--name=$name --suffix=enc{enc_channel}_layer{num_layers}_nhead{nhead}_ffn{dim_feedforward}_run{run_idx}"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
