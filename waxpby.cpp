
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

// Routine to compute the update of a vector with the sum of two 
// scaled vectors where:

// w = alpha*x + beta*y

// n - number of vector elements (on this processor)

// x, y - input vectors

// alpha, beta - scalars applied to x and y respectively.

// w - output vector.

/////////////////////////////////////////////////////////////////////////

#include "waxpby.hpp"
#include <iostream>


extern int init_size;
extern int mult;





int waxpby (const int n, const double alpha, const double * const x, 
	    const double beta, const double * const y, 
		     double * const w)
{  
  if (alpha==1.0) {
#ifdef USING_OMP
#pragma omp parallel for
#endif
    for (int i=0; i<n; i++) w[i] = x[i] + beta * y[i];
  }
  else if(beta==1.0) {
#ifdef USING_OMP
#pragma omp parallel for
#endif
    for (int i=0; i<n; i++) w[i] = alpha * x[i] + y[i];
  }
  else {
#ifdef USING_OMP
#pragma omp parallel for
#endif
    for (int i=0; i<n; i++) w[i] = alpha * x[i] + beta * y[i];
  }

  return(0);
}



#ifdef USING_SYCL
#include <CL/sycl.hpp>

int waxpby_sycl_tasked(sycl::queue* q ,const int n, const double alpha, const double * const x, 
	const double beta, const double * const y, 
	double * const w, sycl::event e) {  

    
    const size_t numGroups = init_size * mult;
	

    size_t globalSize = n;
    const size_t initial_local_size = n / numGroups;    // Desired work-group size
    const size_t localSize = std::pow(2, std::round(std::log2(initial_local_size)));
    if (globalSize % localSize != 0){
      globalSize = ((globalSize / localSize) + 1) * localSize;
    }

	if (alpha==1.0) {
		q->parallel_for(sycl::nd_range<1>(sycl::range<1>(globalSize), sycl::range<1>(localSize)),{e}, [=](sycl::nd_item<1> it) {
			size_t i = it.get_global_id(0);
        	if (i < n){
			w[i] = x[i] + beta * y[i]; 
			}
	});  
	}
	else if (beta==1.0) {

		q->parallel_for(sycl::nd_range<1>(sycl::range<1>(globalSize), sycl::range<1>(localSize)),{e}, [=](sycl::nd_item<1> it) {
			size_t i = it.get_global_id(0);
        	if (i < n){
			w[i] = alpha * x[i] + y[i]; 
			}
	});
	} else {
		q->parallel_for(sycl::nd_range<1>(sycl::range<1>(globalSize), sycl::range<1>(localSize)),{e}, [=](sycl::nd_item<1> it) { 
			size_t i = it.get_global_id(0);
        	if (i < n){
			w[i] = alpha * x[i] + beta * y[i]; 
			}
		}); 
	}
	return(0);
}
int waxpby_sycl(sycl::queue* q ,const int n, const double alpha, const double * const x, const double beta, const double * const y, double * const w)
{ 
	
    
    const size_t numGroups = init_size * std::pow(2, mult);

    size_t globalSize = n;
    const size_t initial_local_size = n / numGroups;    // Desired work-group size
    const size_t localSize = std::pow(2, std::round(std::log2(initial_local_size)));
    if (globalSize % localSize != 0){
      globalSize = ((globalSize / localSize) + 1) * localSize;
    }
	if (alpha==1.0) {
		q->parallel_for(sycl::nd_range<1>(sycl::range<1>(globalSize), sycl::range<1>(localSize)), [=](sycl::nd_item<1> it) {
			size_t i = it.get_global_id(0);
        	if (i < n){
				w[i] = x[i] + beta * y[i]; 
			}
		}); 
	} else if (beta==1.0) {
		q->parallel_for(sycl::nd_range<1>(sycl::range<1>(globalSize), sycl::range<1>(localSize)), [=](sycl::nd_item<1> it) {
			size_t i = it.get_global_id(0);
        if (i < n){
			w[i] = alpha * x[i] + y[i]; 
			}
		});
	} else {
		q->parallel_for(sycl::nd_range<1>(sycl::range<1>(globalSize), sycl::range<1>(localSize)),[=](sycl::nd_item<1> it) { 
			size_t i = it.get_global_id(0);
        if (i < n){
			w[i] = alpha * x[i] + beta * y[i]; 
			}
		});   
	}
return(0);
}

#endif 
