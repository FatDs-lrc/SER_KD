set -e
run_idx=$1
gpu=$2

# train_baseline
cmd="python train_no_test.py --dataset_mode=text 
--model=bert --bert_type=roberta-base --gpu_ids=$gpu --data_type=combined
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--output_dim=6 --niter=5 --niter_decay=1
--beta1=0.9 --init_type=normal --no_test
--batch_size=32 --lr=1e-5 --run_idx=$run_idx
--name=finetune --suffix={data_type}_onlyVal{no_test}_{bert_type}_lr{lr}_run{run_idx}
--cvNo=1"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
