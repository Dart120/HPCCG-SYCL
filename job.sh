#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=debug
#SBATCH --job-name=HPCCG

source /etc/profile

module load module load intel-oneapi/2022.1.2/compiler
make
export SYCL_QUEUE_THREAD_POOL_SIZE=1 
./test_HPCCG 3 3 3