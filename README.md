# HPCCG-SYCL
This is a SYCL port of the HPCCG benchmark.

## Compilation
To compile with OpenMP support run `make -f MakefileOMP`

To compile with SYCL support run `make -f MakefileSYCL`

## Execution
To run with SYCL on the CPU run `./test_HPCCG nx ny nz --cpu`

To run with SYCL on the GPU run `./test_HPCCG nx ny nz --gpu`

To run the original OpenMP version run `./test_HPCCG nx ny nz`

Where `nx ny nz` is the grid size of the chimenny discretisation