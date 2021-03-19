set -e
run_idx=$1
gpu=$2

# KD RAW: checkpoints/KD_raw_enc256_layer6_nhead4_ffn512_run1
# KD 4MSE: 
# KD 2MSE:
# only MSE: 

for i in `seq 1 1 10`;
do

cmd="python train_baseline.py  --gpu_ids=$gpu
--dataset_mode=audio_finetune --A_db_dir=/data4/lrc/movie_dataset/dbs/v2_2/comparE.db  --input_dim=130 
--model=finetune 
--pretrained_dir=checkpoints/KD_raw_enc256_layer6_nhead4_ffn512_run1 --pretrained_epoch=35
--output_dim=4 --cls_layers=128 --nhead=4 --num_layers=6 --dim_feedforward=512
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=0
--niter=10 --niter_decay=0 --beta1=0.9 --init_type normal --init_gain=1
--batch_size=128 --lr=2e-5 --run_idx=$run_idx --enc_channel=256
--name=Finetune_KD_raw --suffix=num-layer{num_layers}_nhead{nhead}_dim-feedforward{dim_feedforward}_lr-{lr}_run{run_idx} 
--cvNo=$i"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done