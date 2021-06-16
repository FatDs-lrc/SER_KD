set -e
run_idx=$1
gpu=$2

cmd="python train_baseline.py  --gpu_ids=$gpu
--dataset_mode=audio_finetune  --data_type=meld --A_db_dir=/data12/lrc/lrc/movie_dataset/dbs/meld/comparE.db
--input_dim=130 --val_method=weighted
--model=finetune --pretrained_dir=checkpoints/best_model --pretrained_epoch=30
--output_dim=7 --cls_layers=128 --nhead=4 --num_layers=6 --dim_feedforward=512
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=16
--niter=10 --niter_decay=5 --beta1=0.9 --init_type normal --init_gain=0.02
--batch_size=128 --lr=4e-5 --run_idx=$run_idx --enc_channel=256
--name=Finetune_meld_new --suffix=num-layer{num_layers}_nhead{nhead}_dim-feedforward{dim_feedforward}_lr-{lr}_run{run_idx} 
--cvNo=0"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
