
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

// Routine to compute an approximate solution to Ax = b where:

// A - known matrix stored as an HPC_Sparse_Matrix struct

// b - known right hand side vector

// x - On entry is initial guess, on exit new approximate solution

// max_iter - Maximum number of iterations to perform, even if
//            tolerance is not met.

// tolerance - Stop and assert convergence if norm of residual is <=
//             to tolerance.

// niters - On output, the number of iterations actually performed.

/////////////////////////////////////////////////////////////////////////

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;
#include <cmath>
#include "mytimer.hpp"
#include "HPCCG.hpp"
#include <CL/sycl.hpp>
using namespace sycl;

#define TICK()  t0 = mytimer() // Use TICK and TOCK to time a code section
#define TOCK(t) t += mytimer() - t0








int HPCCG(HPC_Sparse_Matrix * A,
	  const double * const b, double * const x,
	  const int max_iter, const double tolerance, int &niters, double & normr,
	  double * times)

{
  sycl::default_selector selector;
	sycl::queue q(selector);
  
  std::cout << "Mem Allocation Started"<< std::endl;

  double t_begin = mytimer();  // Start timing right away

  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;

  int nrow = A->local_nrow;
  int ncol = A->local_ncol;
  
  double * r = static_cast<double*>(sycl::malloc_device(sizeof(double) * nrow, q));
  double * p = static_cast<double*>(sycl::malloc_device(sizeof(double) * ncol, q));
  double * Ap = static_cast<double*>(sycl::malloc_device(sizeof(double) * nrow, q));
  double * x_device = static_cast<double*>(sycl::malloc_device(sizeof(double) * nrow, q));
  q.memcpy(x_device, x, sizeof(double) * nrow).wait();
  double * b_device = static_cast<double*>(sycl::malloc_device(sizeof(double) * nrow, q));
  q.memcpy(b_device, b, sizeof(double) * nrow).wait();
  double** pointer_to_cur_vals_lst = static_cast<double**>(sycl::malloc_device(sizeof(double*) * nrow, q));
	int** pointer_to_cur_inds_lst = static_cast<int**>(sycl::malloc_device(sizeof(int*) * nrow, q));
  std::cout << "Entering loop"<< std::endl;
	// For each row
	for (int i = 0; i < nrow; i++)
	{
		pointer_to_cur_vals_lst[i] = static_cast<double*>(sycl::malloc_device(sizeof(double) * A->nnz_in_row[i], q));
		pointer_to_cur_inds_lst[i] = static_cast<int*>(sycl::malloc_device(sizeof(int) * A->nnz_in_row[i], q));
		q.memcpy(pointer_to_cur_vals_lst[i], A->ptr_to_vals_in_row[i], sizeof(double) * A->nnz_in_row[i]).wait();
		q.memcpy(pointer_to_cur_inds_lst[i], A->ptr_to_inds_in_row[i], sizeof(int) * A->nnz_in_row[i]).wait();
	}
  std::cout << "Leavingloop"<< std::endl;
	double* pointer_to_y = malloc_device<double>(nrow, q);
	int* pointer_to_cur_nnz = malloc_device<int>(nrow, q);
	q.memcpy(pointer_to_cur_nnz, A->nnz_in_row, sizeof(int) * nrow).wait();
  double * rtrans = static_cast<double*>(sycl::malloc_device(sizeof(double), q));
  double * oldrtrans = static_cast<double*>(sycl::malloc_device(sizeof(double), q));
  double * normr_shared = static_cast<double*>(sycl::malloc_shared(sizeof(double), q));
  q.memcpy(normr_shared, &normr, sizeof(double)).wait();
  *rtrans = 0.0;
  *oldrtrans = 0.0;
  double* beta = static_cast<double*>(sycl::malloc_device(sizeof(double), q));
  double* alpha = static_cast<double*>(sycl::malloc_device(sizeof(double), q));
#ifdef USING_MPI
  int rank; // Number of MPI processes, My process ID
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  int rank = 0; // Serial case (not using MPI)
#endif

  int print_freq = max_iter/10; 
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;
  std::cout << "Mem Allocation Finished"<< std::endl;
  // p is of length ncols, copy x to p for sparse MV operation
  TICK(); waxpby_sycl(&q, nrow, 1.0, x_device, 0.0, x_device, p); TOCK(t2);
#ifdef USING_MPI
  TICK(); exchange_externals(A,p); TOCK(t5); 
#endif

  TICK(); HPC_sparsemv_sycl(&q,pointer_to_cur_vals_lst,pointer_to_cur_inds_lst,pointer_to_cur_nnz,nrow, p, Ap);  TOCK(t3); // 2*nnz ops
  
  TICK(); waxpby_sycl(&q,nrow, 1.0, b_device, -1.0, Ap, r); TOCK(t2);
  TICK(); ddot_sycl(&q,nrow, r, r, rtrans, t4); TOCK(t1);

  q.submit([&](handler& h) {
    sycl::stream out(655, 655, h);
    h.single_task([=]() {
    *normr_shared = sqrt(*rtrans);
    if (rank==0) out << "Initial Residual = "<< *normr_shared << cl::sycl::endl;
    });
    
    
  }).wait();




for(int k=1; k<max_iter && *normr_shared > tolerance; k++ )
    {
      if (k == 1)
	{
    

	  TICK(); waxpby_sycl(&q,nrow, 1.0, r, 0.0, r, p); TOCK(t2);
  
  
	}
      else
	{
	  

    q.submit([&](handler& h) {

    // sycl::stream out(655, 655, h);
    h.single_task([=]() {
     
     *oldrtrans = *rtrans;
  //  for (size_t i = 0; i < nrow; i++)
  // {
  //   out << "r = "<< r[i] << cl::sycl::endl;
   
  // }
 
    });
  }).wait();
  
  
  
	  TICK(); ddot_sycl(&q,nrow, r, r, rtrans, t4);// 2*nrow ops 
    // q.submit([&](handler& h) {
    // sycl::stream out(655, 655, h);
    // h.single_task([=]() {
    // out << "rt = "<< *rtrans << cl::sycl::endl;
    // });
  // }).wait();
  
    
    q.submit([&](handler& h) {

    // sycl::stream out(655, 655, h);
    h.single_task([=]() {
     
     *beta = *rtrans / *oldrtrans;

 
    });
  }).wait();
  
	  
	  TICK(); waxpby_sycl(&q,nrow, 1.0, r, *beta, p, p);  TOCK(t2);// 2*nrow ops 
	}

  q.submit([&](handler& h) {
   
    h.single_task([=]() {
    *normr_shared = sqrt(*rtrans);

    });
    
    
  }).wait();
        if (rank==0 && (k%print_freq == 0 || k+1 == max_iter))
      std::cout << "Iteration = "<< k << "   Residual = "<< *normr_shared << std::endl;
  
  TICK();HPC_sparsemv_sycl(&q,pointer_to_cur_vals_lst,pointer_to_cur_inds_lst,pointer_to_cur_nnz,nrow, p, Ap);  TOCK(t3); // 2*nnz ops

      

      TICK(); ddot_sycl(&q,nrow, p, Ap, alpha, t4); TOCK(t1); // 2*nrow ops 
      q.submit([&](handler& h) {
    h.single_task([=]() {
           *alpha = *rtrans/ *alpha;
    });
  }).wait();

      TICK(); waxpby_sycl(&q,nrow, 1.0, x_device, *alpha, p, x_device); // 2*nrow ops 

      waxpby_sycl(&q,nrow, 1.0, r, -(*alpha), Ap, r);  TOCK(t2);// 2*nrow ops 

      niters = k;
  
    }

  // Store times
  times[1] = t1; // ddot time
  times[2] = t2; // waxpby time
  times[3] = t3; // sparsemv time
  times[4] = t4; // AllReduce time
#ifdef USING_MPI
  times[5] = t5; // exchange boundary time
#endif
    std::cout << "Begun Freeing"<< std::endl;
   sycl::free(r,q);
   sycl::free(p,q);
   sycl::free(Ap,q);
   sycl::free(x_device,q);
   sycl::free(b_device,q);
   std::cout << "Enter Loop"<< std::endl;
   for (int i = 0; i < nrow; i++)
	{
    sycl::free(pointer_to_cur_inds_lst[i],q);
    sycl::free(pointer_to_cur_vals_lst[i],q);
	}
  std::cout << "Finished Loop"<< std::endl;
  sycl::free(pointer_to_cur_inds_lst,q);
  sycl::free(pointer_to_cur_vals_lst,q);
  sycl::free(pointer_to_y,q);
  sycl::free(pointer_to_cur_nnz,q);
  sycl::free(rtrans,q);
  sycl::free(oldrtrans,q);
  sycl::free(beta,q);
  sycl::free(alpha,q);
  q.memcpy(&normr, normr_shared, sizeof(double)).wait();
  sycl::free(normr_shared,q);
  std::cout << "All Memeory Free"<< std::endl;
  times[0] = mytimer() - t_begin;  // Total time. All done...
  return(0);
}
