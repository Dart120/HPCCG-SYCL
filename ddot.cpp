
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
#include <CL/sycl.hpp>

#ifdef USING_SYCL

int ddot (const int n, const double * const x, const double * const y, 
	  double * const result, double & time_allreduce)
{  
  sycl::default_selector selector;
	sycl::queue q(selector);
  double sump = 0;
  {
    
    sycl::buffer<double> sum_buf(&sump,1);
    sycl::buffer<double> x_buf(x,cl::sycl::range<1>(n));
    if(y==x){
      // std::cout<<"eq"<<std::endl;
      q.submit([&](auto &h) {
      sycl::accessor x_acc(x_buf, h, sycl::read_only);
      auto sumr =sycl::reduction(sum_buf,h, sycl::ext::oneapi::plus<>());
      h.parallel_for(sycl::range<1>{static_cast<unsigned long>(n)}, sumr, [=](sycl::id<1> i, auto &sum) {
      
        sum += x_acc[i] * x_acc[i];
      });
    });
    }else{
      // std::cout<<"diff"<<std::endl;
      sycl::buffer<double> y_buf(y,cl::sycl::range<1>(n));
      


q.submit([&](auto &h) {
      sycl::accessor x_acc(x_buf, h, sycl::read_only);
      sycl::accessor y_acc(y_buf, h, sycl::read_only);
      auto sumr =sycl::reduction(sum_buf,h, sycl::ext::oneapi::plus<>());
      h.parallel_for(sycl::range<1>{static_cast<unsigned long>(n)}, sumr, [=](sycl::id<1> i, auto &sum) {
      
        sum += x_acc[i] * y_acc[i];
      });
    });


    }
    
  }
  *result = sump;
  // std::cout<<"result "<< *result<<std::endl;

  return 0;
}

#else

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

#endif
// *** EDITED CODE ***

int ddot_sycl(sycl::queue* q, const int n, const double * const x, const double * const y, 
	  double * const result, double & time_allreduce)
{  
  {
    sycl::buffer<double> sum_buf(result,1);
    if(y==x){
      // std::cout<<"eq"<<std::endl;
      q->submit([&](auto &h) {
   
      
      auto sumr =sycl::reduction(sum_buf,h, sycl::ext::oneapi::plus<>());
      h.parallel_for(sycl::range<1>{static_cast<unsigned long>(n)}, sumr, [=](sycl::id<1> i, auto &sum) {
        sum += x[i] * x[i];
      });
    });





    }else{
      // std::cout<<"diff"<<std::endl;

      q->submit([&](auto &h) {
   
      
      auto sumr =sycl::reduction(sum_buf,h, sycl::ext::oneapi::plus<>());
      h.parallel_for(sycl::range<1>{static_cast<unsigned long>(n)}, sumr, [=](sycl::id<1> i, auto &sum) {
        sum += x[i] * y[i];
      });
    });





    }
  // std::cout<<"result "<< *result<<std::endl;
  }
  return 0;
}
