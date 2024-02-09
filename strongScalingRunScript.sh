# These commands will run the HPCCG executable on 1-64 processors at powers
# of 2 and change the problem size so that the same global problem is
# being solved regardless of processor count.  This is sometimes referred
# to as "strong scaling".

export SYCL_QUEUE_THREAD_POOL_SIZE=1 
time ./test_HPCCG 300 300 300
export SYCL_QUEUE_THREAD_POOL_SIZE=2 
time ./test_HPCCG 300 300 300
export SYCL_QUEUE_THREAD_POOL_SIZE=4 
time ./test_HPCCG 300 300 300
export SYCL_QUEUE_THREAD_POOL_SIZE=8 
time ./test_HPCCG 300 300 300
export SYCL_QUEUE_THREAD_POOL_SIZE=16 
time ./test_HPCCG 300 300 300
export SYCL_QUEUE_THREAD_POOL_SIZE=32 
time ./test_HPCCG 300 300 300
export SYCL_QUEUE_THREAD_POOL_SIZE=64 
time ./test_HPCCG 300 300 300
export SYCL_QUEUE_THREAD_POOL_SIZE=128
time ./test_HPCCG 300 300 300
