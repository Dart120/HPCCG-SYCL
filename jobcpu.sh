#!/bin/bash
#SBATCH -N 1
#SBATCH -c 100
#SBATCH -p cpu
#SBATCH --qos=debug
#SBATCH --job-name=biggggcpuplsletmehaveit
#SBATCH --time=01:30:00

source /etc/profile
module unload cuda
module load llvm-clang/15.0.0 
module load cuda/11.5
 make -f MakefileOMP
./run_tests.sh