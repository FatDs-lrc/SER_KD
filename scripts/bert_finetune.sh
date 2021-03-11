set -e
run_idx=$1
gpu=$2


for i in `seq 1 1 10`;
do
cmd="python train_baseline.py --dataset_mode=bert_tokenize 
--model=bert_cls --bert_type=bert-base-uncased --gpu_ids=$gpu
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=10 
--output_dim=4  --niter=5 --niter_decay=1 --embd_method=max
--beta1=0.9 --init_type normal --num_thread=4
--batch_size=128 --lr=1e-5 --run_idx=$run_idx
--name=finetune --suffix={bert_type}_{embd_method}_lr{lr}_run{run_idx}
--cvNo=$i"


echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh

done