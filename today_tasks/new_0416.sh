set -e
t=$1
f=$2
mse=$3
cycle=$4
run_idx=$5
gpu=$6

for i in `seq 1 1 10`;
do

cmd="python train_new_0416.py --dataset_mode=iemocap_miss --model=new_translation --gpu_ids=$gpu
--log_dir=./logs --checkpoints_dir=./checkpoints --print_freq=4
--acoustic_ft_type=IS10 --lexical_ft_type=text --visual_ft_type=denseface
--teacher_path='checkpoints/ef_AVL_Adnn512,256,128_Vlstm128_maxpool_Lcnn128_fusion256,128run1'
--input_dim_a=1582 --mid_layers_a=512,256,128
--input_dim_v=342 --hidden_size_v=128 --embd_size_v=128 --embd_method=maxpool
--input_dim_l=1024 --embd_size_l=128
--AE_layers=256,128,64
--fusion_size=384 --mid_layers_fusion=256,128 --output_dim=4 
--ce_weight_t=$t --ce_weight_f=$f --mse_weight=$mse --cycle_weight=$cycle
--AE_layers=256,128,64 --n_blocks=5
--niter=60 --niter_decay=90 --verbose --real_data_rate=0.2
--batch_size=256 --lr=1e-3 --dropout_rate=0.5 --run_idx=$run_idx
--miss_num=mix --miss2_rate=0.5
--name=new_0416_simpleAE --suffix=ce_t{ce_weight_t}_f{ce_weight_f}_mse{mse_weight}_cycle{cycle_weight}_run{run_idx}
--cvNo=$i"

# miss2_fix{miss2_rate}_real{real_data_rate}_AE{AE_layers}_blocks{n_blocks}_ce{ce_weight}_mse_{mse_weight}

echo "\n-------------------------------------------------------------------------------------"
echo "Execute command: $cmd"
echo "-------------------------------------------------------------------------------------\n"
echo $cmd | sh
done