set -e
run_idx=$1
gpu=$2

# train_baseline
cmd="python train.py --dataset_mode=text 
--model=bert_cls --bert_type=roberta-base --gpu_ids=$gpu --data_type=emotionX_frd
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--output_dim=6 --niter=5 --niter_decay=1
--beta1=0.9 --init_type=normal --embd_method=max
--batch_size=32 --lr=1e-5 --run_idx=$run_idx
--name=finetune_with_test --suffix={bert_type}_{embd_method}_{data_type}_onlyVal{no_test}_lr{lr}_run{run_idx}
--cvNo=1"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
