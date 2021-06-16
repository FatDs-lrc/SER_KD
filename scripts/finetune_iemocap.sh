set -e
run_idx=$1
gpu=$2

#  --lr=4e-5   --lr=8e-5  --lr=1e-4 
# --A_db_dir=/data4/lrc/movie_dataset/dbs/iemocap/comparE.db
for i in `seq 1 1 10`;
do

cmd="python train_no_test.py  --gpu_ids=$gpu
--dataset_mode=audio_finetune   --A_db_dir=/data12/lrc/lrc/movie_dataset/dbs/iemocap/comparE.db
--input_dim=130 --val_method=macro
--model=finetune --pretrained_dir=checkpoints/best_model --pretrained_epoch=30
--output_dim=4 --cls_layers=128 --nhead=4 --num_layers=6 --dim_feedforward=512
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=16
--niter=5 --niter_decay=5 --beta1=0.9 --init_type normal --init_gain=0.02
--batch_size=128 --lr=4e-5 --run_idx=$run_idx --enc_channel=256
--name=Finetune_IEMOCAP_new --suffix=num-layer{num_layers}_nhead{nhead}_dim-feedforward{dim_feedforward}_lr-{lr}_run{run_idx} 
--cvNo=$i"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done

