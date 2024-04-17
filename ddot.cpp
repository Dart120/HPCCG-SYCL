
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

// Routine to compute the dot product of two vectors where:

// n - number of vector elements (on this processor)

// x, y - input vectors

// residual - pointer to scalar value, on exit will contain result.

/////////////////////////////////////////////////////////////////////////

#include "ddot.hpp"

int init_size = 32;
int mult = 3;


int ddot (const int n, const double * const x, const double * const y, 
	  double * const result, double & time_allreduce)
{  
  double local_result = 0.0;
  if (y==x)
#ifdef USING_OMP
#pragma omp parallel for reduction (+:local_result)
#endif
    for (int i=0; i<n; i++) local_result += x[i]*x[i];
  else
#ifdef USING_OMP
#pragma omp parallel for reduction (+:local_result)
#endif
    for (int i=0; i<n; i++) local_result += x[i]*y[i];

#ifdef USING_MPI
  // Use MPI's reduce function to collect all partial sums
  double t0 = mytimer();
  double global_result = 0.0;
  MPI_Allreduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 
                MPI_COMM_WORLD);
  *result = global_result;
  time_allreduce += mytimer() - t0;
#else
  *result = local_result;
#endif

  return(0);
}


// *** EDITED CODE ***
#ifdef USING_SYCL
#include <CL/sycl.hpp>

sycl::event ddot_sycl(sycl::queue* q, const int n, const double * const x, const double * const y, double * const result)
{  
    
    
    const size_t numGroups = init_size * std::pow(2, mult);
    *result = 0;
    size_t globalSize = n;
    const size_t initial_local_size = n / numGroups;    // Desired work-group size
    const size_t localSize = std::pow(2, std::round(std::log2(initial_local_size)));
    if (globalSize % localSize != 0){
      globalSize = ((globalSize / localSize) + 1) * localSize;
    }
        

    sycl::event e_ddot;
    // std::cout << "Global size in ddot: "<< globalSize<<std::endl;
    // std::cout << "Local size in ddot: "<< localSize<<std::endl;
    auto sumr = sycl::reduction(result,sycl::plus<>());
    if (y == x) {
      
      e_ddot = q->submit([&](auto &h) {
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(globalSize), sycl::range<1>(localSize)), sumr, [=](sycl::nd_item<1> it, auto &sum) {
        size_t i = it.get_global_id(0);
        if (i < n){
          sum += x[i] * x[i];
        }
        
      });
    }); 
    } else {
      e_ddot = q->submit([&](auto &h) {
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(globalSize), sycl::range<1>(localSize)), sumr, [=](sycl::nd_item<1> it, auto &sum) {
        size_t i = it.get_global_id(0);
			if (i < n){
          sum += x[i] * y[i];
        }
      });
    });
  }
  // exit(0);
  return e_ddot;
}

#endif
