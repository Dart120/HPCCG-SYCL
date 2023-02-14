
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
// #include <CL/sycl.hpp>


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


// ***EDITED CODE ***



// int waxpby (const int n, const double alpha, const double * const x, 
// 	    const double beta, const double * const y, 
// 		     double * const w)
// {  
//   // std::cout<< "START " << x[0] << std::endl;
//   // std::cout << y[0] << std::endl;
//   // std::cout << alpha << std::endl;
//   // std::cout << beta << std::endl;

//   sycl::default_selector selector;
//   // sycl::gpu_selector selector;
//   auto R = sycl::range<1>(n);
//   sycl::queue q(selector);
//   {
//   cl::sycl::buffer<double, 1> x_sycl(x, cl::sycl::range<1>(n));
//   cl::sycl::buffer<double, 1> y_sycl(y, cl::sycl::range<1>(n));
//   cl::sycl::buffer<double, 1> w_sycl(w, cl::sycl::range<1>(n));





//   if (alpha==1.0) {
//     q.submit([&](sycl::handler& h) {
//         auto x_acc = x_sycl.get_access<cl::sycl::access::mode::read>(h);
//          auto y_acc = y_sycl.get_access<cl::sycl::access::mode::read>(h);
//          auto w_acc = w_sycl.get_access<cl::sycl::access::mode::discard_write>(h);
//        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
//           // int sum = 0;
//          w_acc[i] = x_acc[i] + beta * y_acc[i]; }); 
//          }).wait();
//   }
//   else if(beta==1.0) {
//     q.submit([&](sycl::handler& h) {
// auto x_acc = x_sycl.get_access<cl::sycl::access::mode::read>(h);
//          auto y_acc = y_sycl.get_access<cl::sycl::access::mode::read>(h);
//          auto w_acc = w_sycl.get_access<cl::sycl::access::mode::discard_write>(h);
//        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
//         // int sum = 0;
//         w_acc[i] = alpha * x_acc[i] + y_acc[i]; 
//         }); }).wait();
//   }
//   else {
//     q.submit([&](sycl::handler& h) {
//         auto x_acc = x_sycl.get_access<cl::sycl::access::mode::read>(h);
//          auto y_acc = y_sycl.get_access<cl::sycl::access::mode::read>(h);
//          auto w_acc = w_sycl.get_access<cl::sycl::access::mode::discard_write>(h);
//        h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { 
//         // int sum = 0;
//          w_acc[i] = alpha * x_acc[i] + beta * y_acc[i]; }); 
//          }).wait();
//   }
//   // std::cout << w[0] << std::endl;
//   }
//   return(0);
// }