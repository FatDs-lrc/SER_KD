set -e
run_idx=$1
gpu=$2


cmd="python train_no_test.py --dataset_mode=movie_v2 --gpu_ids=$gpu
--input_dim=130 --model=movie --data_type=name
--warmup --warmup_lr=1e-7 --warmup_epoch=5
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=16
--output_dim=5 --cls_layers=128,128 --nhead=4 --num_layers=6 --dim_feedforward=512
--niter=40 --niter_decay=60 --verbose --beta1=0.9 --init_type normal
--batch_size=256 --lr=1e-4 --run_idx=$run_idx --enc_channel=256 --no_test
--name=TEST_5class --suffix=enc{enc_channel}_layer{num_layers}_nhead{nhead}_ffn{dim_feedforward}_run{run_idx}"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done