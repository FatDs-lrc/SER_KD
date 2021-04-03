set -e

cmd="python train.py --dataset_mode=movie_lmdb --gpu_ids=0
--input_dim=130 --model=pretrain
--db_dir=/data4/lrc/movie_dataset/dbs/v2
--warmup --warmup_lr=1e-7 --warmup_epoch=5
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=16
--output_dim=5 --cls_layers=128,128 --nhead=4 --num_layers=6 --dim_feedforward=512
--kd_weight=1.0 --mse_weight1=0.1 --mse_weight2=0.1 --mse_weight3=0.2 --mse_weight4=0.3
--niter=25 --niter_decay=25 --beta1=0.9 --init_type normal
--batch_size=256 --lr=1e-4 --run_idx=1 --enc_channel=256 --no_test
--name=pretrain --suffix={ratio}_enc{enc_channel}_layer{num_layers}_nhead{nhead}_ffn{dim_feedforward}_run{run_idx}"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh