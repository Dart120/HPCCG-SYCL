#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu
#SBATCH -p ug-gpu-small
#SBATCH --qos=debug
#SBATCH --job-name=bigggggpuplsletmehaveit
#SBATCH --gres=gpu:turing:1
#SBATCH --time=01:30:00

source /etc/profile
module unload cuda
module load llvm-clang/15.0.0 
module load cuda/11.5
#  make -f MakefileSYCL clean
#  make -f MakefileSYCL
./run_tests_tur.sh