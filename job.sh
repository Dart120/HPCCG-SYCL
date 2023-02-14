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
./test_HPCCG 200 300 100