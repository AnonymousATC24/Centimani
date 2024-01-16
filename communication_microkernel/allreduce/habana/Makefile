nccl_single:
	nvcc nccl_single_all_reduce.cu -o nccl_single_all_reduce -I ../kernels/ -I /soft/nccl/nccl-v2.11.4-1/include/ -I /soft/cuda/cudnn-11.3-linux-x64-v8.2.0.53/include/ -L /soft/nccl/nccl-v2.11.4-1/lib/ -L /soft/cuda/cudnn-11.3-linux-x64-v8.2.0.53/lib64 -lnccl -lcudart -lcurand  -arch=sm_80 -std=c++11

clean:
	rm -rf nccl_single_all_reduce

rebuild: clean all
