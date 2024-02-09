101
module load llvm-clang
module load cuda/11.5
# clang++ -fsycl -fsycl-targets=nvptx64-cuda sycl_hello_world.cpp -o hw