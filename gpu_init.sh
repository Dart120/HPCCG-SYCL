module load llvm-clang/15.0.0 
module load cuda/11.7
# clang++ -fsycl -fsycl-targets=nvptx64-cuda sycl_hello_world.cpp -o hw