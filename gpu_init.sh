module load llvm-clang/15.0.0 
module unload cuda
module load cuda/11.5
# clang++ -fsycl -fsycl-targets=nvptx64-cuda sycl_hello_world.cpp -o hw
