
//@HEADER
// ************************************************************************
// 
//               HPCCG: Simple Conjugate Gradient Benchmark Code
//                 Copyright (2006) Sandia Corporation
// 
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
// 
// BSD 3-Clause License
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// * Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
// 
// * Neither the name of the copyright holder nor the names of its
//   contributors may be used to endorse or promote products derived from
//   this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// 
// Questions? Contact Michael A. Heroux (maherou@sandia.gov) 
// 
// ************************************************************************
//@HEADER

/////////////////////////////////////////////////////////////////////////

// Routine to compute matrix vector product y = Ax where:
// First call exchange_externals to get off-processor values of x

// A - known matrix 
// x - known vector
// y - On exit contains Ax.

/////////////////////////////////////////////////////////////////////////

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cassert>
#include <string>
#include <cmath>
#include "HPC_sparsemv.hpp"

// int HPC_sparsemv( HPC_Sparse_Matrix *A, 
// 		 const double * const x, double * const y)
// {

//   const int nrow = (const int) A->local_nrow;

// #ifdef USING_OMP
// #pragma omp parallel for
// #endif
// // For each row
//   for (int i=0; i< nrow; i++)
//     {
//       double sum = 0.0;
//       const double* const cur_vals = (const double * const) A->ptr_to_vals_in_row[i];

//       const int* const cur_inds = (const int    * const) A->ptr_to_inds_in_row[i];

//       const int cur_nnz = (const int) A->nnz_in_row[i];
//       // For each number in the row
//       for (int j=0; j< cur_nnz; j++)
//           // Sum = the number * 
//           sum += cur_vals[j]*x[cur_inds[j]];
//       y[i] = sum;
//     }
//   return(0);
// }


// EDITED CODE

#include <CL/sycl.hpp>


int HPC_sparsemv( HPC_Sparse_Matrix *A, 
    const double * const x, double * const y
    )
{
  const int nrow = (const int) A->local_nrow;
  // Trying to find the max value of X that is accessed
  int max = 0;
  for (int i = 0; i < nrow; i++){
    const int* const list = (const int    * const) A->ptr_to_inds_in_row[i];
    int list_size = sizeof(*list) / sizeof(int);
    for (int j = 0; j < list_size; j++){
      if (list[j] > max){
        max = list[j];
      }
    }
  }
  // std::cout << max << std::endl;
  // exit(0);
  sycl::default_selector selector;
  sycl::queue q(selector);
  
  const double** const ptr_to_vals_in_row = malloc_shared<const double*>(nrow,q);
  const int** const ptr_to_inds_in_row = malloc_shared<const int* >(nrow,q);
  const int* nnz_in_row = malloc_shared<int>(nrow,q);
  double* y_device =  malloc_shared<double>(nrow,q);
  double* x_device =  malloc_shared<double>(nrow,q);

q.submit([&](sycl::handler &h) {
  h.memcpy(ptr_to_vals_in_row, &A->ptr_to_vals_in_row, nrow * sizeof(double*));

});

q.wait();
// allocate just that space to x
q.submit([&](sycl::handler &h) {

  h.memcpy(x_device, x, max * sizeof(double));

});

q.wait();
q.submit([&](sycl::handler &h) {

  h.memcpy(y_device, y, nrow * sizeof(double));

});

q.wait();
q.submit([&](sycl::handler &h) {


  h.memcpy(ptr_to_inds_in_row, &A->ptr_to_inds_in_row, nrow * sizeof(int*));

});

q.wait();
q.submit([&](sycl::handler &h) {

  h.memcpy((void *) nnz_in_row, &A->nnz_in_row, nrow * sizeof(int));
});

q.wait();
// Still segfaults here
q.submit([&](sycl::handler &h) {
   h.parallel_for(sycl::range<1>(nrow), [=](sycl::id<1> i) {
    double sum = 0.0;
     const double* const cur_vals = ptr_to_vals_in_row[i];
     const int* const cur_inds = ptr_to_inds_in_row[i];
     const int cur_nnz = nnz_in_row[i];
     for (int j=0; j< cur_nnz; j++) {
      // std::cout << cur_inds[j] << std::endl;
        sum += cur_vals[j]*x[cur_inds[j]];
      }
      y_device[i] = sum;
     });
  });
 q.wait();
  return(0);
}


