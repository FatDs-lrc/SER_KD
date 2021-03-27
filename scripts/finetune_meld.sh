set -e
run_idx=$1
gpu=$2

# KD RAW: 
# KD 4MSE: SAMPLE2_CMKD_4MSE_4layer_RSM_enc128_layer6_nhead4_ffn512_run1 7
# KD 3MSE: SAMPLE2_CMKD_4MSE_3layer_RSM_sample_enc128_layer6_nhead4_ffn512_run1 5
# KD 2MSE: SAMPLE2_CMKD_4MSE_2layer_RSM_sample_enc128_layer6_nhead4_ffn512_run1 5
# KD 1MSE: SAMPLE2_CMKD_4MSE_1layer_RSM_sample_enc128_layer6_nhead4_ffn512_run1 3
# only MSE: SAMPLE2_CMKD_4MSE_1layer_RSM_sample_enc128_layer6_nhead4_ffn512_run1 4

cmd="python train_baseline.py  --gpu_ids=$gpu
--dataset_mode=audio_finetune  --data_type=meld --A_db_dir=/data4/lrc/movie_dataset/dbs/meld/comparE.db
--input_dim=130 
--model=finetune --pretrained_dir=checkpoints/YYY_4layer_RSM_enc128_layer6_nhead4_ffn512_run2 --pretrained_epoch=16
--output_dim=7 --cls_layers=128 --nhead=4 --num_layers=6 --dim_feedforward=512
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 --num_threads=16
--niter=10 --niter_decay=10 --beta1=0.9 --init_type normal --init_gain=0.02
--batch_size=128 --lr=4e-5 --run_idx=$run_idx --enc_channel=128
--name=Finetune_on_MELD_YYY --suffix=num-layer{num_layers}_nhead{nhead}_dim-feedforward{dim_feedforward}_lr-{lr}_run{run_idx} 
--cvNo=0"

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
