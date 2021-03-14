set -e
embd_method=$1
run_idx=$2
gpu=$3

# train_baseline
cmd="python train_no_test.py --dataset_mode=text 
--model=bert_cls --bert_type=roberta-base --gpu_ids=$gpu --data_type=filtered
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--output_dim=6 --niter=5 --niter_decay=1
--beta1=0.9 --init_type=normal --no_test --embd_method=$embd_method
--batch_size=32 --lr=1e-5 --run_idx=$run_idx
--name=TEXT_filtered --suffix={bert_type}_{embd_method}_{data_type}_onlyVal{no_test}_lr{lr}_run{run_idx}
--cvNo=1"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
