set -e
run_idx=$1
gpu=$2


cmd="python train_baseline.py  --gpu_ids=$gpu
--dataset_mode=audio_finetune_part  --data_type=meld --A_db_dir=/data4/lrc/movie_dataset/dbs/meld/comparE.db
--input_dim=130 --ratio=1-5
--model=finetune --pretrained_dir=checkpoints/XV4_10_4layer_enc256_layer6_nhead4_ffn512_run2 --pretrained_epoch=33
--output_dim=7 --cls_layers=128 --nhead=4 --num_layers=6 --dim_feedforward=512
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=16
--niter=10 --niter_decay=5 --beta1=0.9 --init_type normal --init_gain=0.02
--batch_size=128 --lr=2e-5 --run_idx=$run_idx --enc_channel=256
--name=Finetune_meld_1-5 --suffix=num-layer{num_layers}_nhead{nhead}_dim-feedforward{dim_feedforward}_lr-{lr}_run{run_idx} 
--cvNo=0"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
