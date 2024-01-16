#! /bin/bash
#This script will run various rnn configurations

set -e

# Run for Kernel sizes from DeepBench
m=(50 25 25 50) #time_steps
n=(256 512 512 512) #input_size
k=(256 512 512 512) #hidden_size
depth=(16 32 64 128)
batch_sizes=(64 32 16 32)

# Based on Kernel sizes from DeepBench 
# Model Dir name convention : gemm_model_dir_depth_BatchSize_m_n_k
for d in ${!depth[@]}; do
    for i in ${!m[@]}; do

        echo "RUNNING Kernel_$i for Depth batch_size m n k : " ${depth[$d]} ${batch_sizes[$i]} ${m[$i]} ${n[$i]} ${k[$i]}

        csrun_wse python run.py --mode train --model rnn --model_dir rnn_model_dir_kernel"$i"_${depth[$d]}_${batch_sizes[$i]}_${m[$i]}_${n[$i]}_${k[$i]} --params configs/params.yaml --cs_ip 192.168.220.50 --max_steps 1 --feature_shape1 ${m[$i]} --feature_shape2 ${n[$i]} --label_shape ${m[$i]} --hidden_size ${k[$i]} --depth ${depth[$d]} --batch_size ${batch_sizes[$i]}
    done
done



