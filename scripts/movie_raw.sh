set -e
run_idx=$1
gpu=$2


cmd="python train_no_test.py --dataset_mode=movie --gpu_ids=$gpu
--input_dim=130 --model=movie_raw
--resume --resume_dir=checkpoints/KD_raw_resume/0 --resume_epoch=80
--warmup --warmup_lr=1e-7 --warmup_epoch=5
--A_db_dir=/data4/lrc/movie_dataset/dbs/v2/comparE.db 
--L_db_dir=/data4/lrc/movie_dataset/dbs/v2/bert_light.db
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=16
--output_dim=5 --cls_layers=128,128 --nhead=4 --num_layers=6 --dim_feedforward=512
--niter=20 --niter_decay=20 --verbose --beta1=0.9 --init_type normal
--batch_size=256 --lr=1e-4 --run_idx=$run_idx --enc_channel=256 --no_test
--name=KD_raw --suffix=enc{enc_channel}_layer{num_layers}_nhead{nhead}_ffn{dim_feedforward}_run{run_idx}"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done