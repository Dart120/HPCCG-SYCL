
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


#define TICK()  t0 = mytimer() // Use TICK and TOCK to time a code section
#define TOCK(t) t += mytimer() - t0






#ifdef USING_SYCL
#include <CL/sycl.hpp>
using namespace sycl;
template <typename T>
void sample(sycl::queue* q,T * lst, char* c){
  std::cout <<"New Output: "<< c << std::endl;
  q->submit([&](handler& h) {
    stream out(256, 1024, h);
        h.single_task([=]() {
          for (size_t i = 0; i < 40; i++)
          {
            out << lst[i] <<",";
          }
          out << sycl::endl;
        });
    }).wait();
}
template <typename T>
void one(sycl::queue* q,T val, char* c){
  std::cout <<"New Output: "<< c << std::endl;
  q->submit([&](handler& h) {
    stream out(256, 1024, h);
        h.single_task([=]() {
            out << *val <<sycl::endl;
        });
    }).wait();
}

int HPCCG_sycl(sycl::queue *q,HPC_Sparse_Matrix * A,
	  double * const b_device, double * const x_device,
	  const int max_iter, const double tolerance, int &niters, double & normr,
	  double * times)

{

  int c = 0;
  
  std::cout << "Mem Allocation Started"<< std::endl;


  double t_begin = mytimer();  // Start timing right away

  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;

  int nrow = 0;
  int ncol = 0;


  q->memcpy(&nrow, &(A->local_nrow), sizeof(int)).wait(); 
  q->memcpy(&ncol, &(A->local_ncol), sizeof(int)).wait(); 
  



  
  double * r = static_cast<double*>(sycl::malloc_device(sizeof(double) * nrow, *q));
  double * p = static_cast<double*>(sycl::malloc_device(sizeof(double) * ncol, *q));
  double * Ap = static_cast<double*>(sycl::malloc_device(sizeof(double) * nrow, *q));




  double * rtrans = static_cast<double*>(sycl::malloc_device(sizeof(double), *q));
  double * oldrtrans = static_cast<double*>(sycl::malloc_device(sizeof(double), *q));
  double * normr_shared = static_cast<double*>(sycl::malloc_shared(sizeof(double), *q));
  
 
  
  // q->memcpy(normr_shared, &normr, sizeof(double)).wait();
 
  q->submit([&](handler& h) {
    h.single_task([=]() {
        *rtrans = 0.0;
        *oldrtrans = 0.0 ;
    });
}).wait(); 

// double** pointer_to_cur_vals_lst = A->ptr_to_vals_in_row;
// 	int** pointer_to_cur_inds_lst = A->ptr_to_inds_in_row;
  


  double* beta = static_cast<double*>(sycl::malloc_shared(sizeof(double), *q));
  double* alpha = static_cast<double*>(sycl::malloc_shared(sizeof(double), *q));
  std::cout << "Mem Allocation Finished"<< std::endl;
  
 
  
  
  


  int rank = 0; // Serial case (not using MPI)


  int print_freq = max_iter/10; 
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;
  
  // p is of length ncols, copy x to p for sparse MV operation

  TICK(); waxpby_sycl(q, nrow, 1.0, x_device, 0.0, x_device, p); q->wait(); TOCK(t2);
  sample(q,p, "init waxpby");
  
  
  
  


  

   

  TICK(); HPC_sparsemv_sycl(q,A,p, Ap, nrow);q->wait();  TOCK(t3); // 2*nnz ops
  sample(q,Ap, "init spmv");
  
  


  TICK(); waxpby_sycl(q,nrow, 1.0, b_device, -1.0, Ap, r); q->wait(); TOCK(t2);
  sample(q,r, "init wax");
  


  
  TICK(); ddot_sycl(q,nrow, r, r, rtrans); q->wait(); TOCK(t1);
   one(q,rtrans, "init dd");
  
 
  
  



  

    
    q->submit([&](handler& h) {
    sycl::stream out(655, 655, h);
    h.single_task([=]() {
      *normr_shared = sqrt(*rtrans);
    *normr_shared = sqrt(*rtrans);
    if (rank==0) out << "Initial Residual = "<< *normr_shared << cl::sycl::endl;
    });
  }).wait();

  





for(int k=1; k<max_iter && *normr_shared > tolerance; k++ )

    {
      if (k == 1)
	{
    
	  TICK(); waxpby_sycl(q,nrow, 1.0, r, 0.0, r, p); q->wait(); TOCK(t2);
    sample(q,p, "k1 wax");
   
    
 
  
	}
      else
	{
	  
 
    q->submit([&](handler& h) {


    h.single_task([=]() {
     
     *oldrtrans = *rtrans;

 
    });
  }).wait();
  
  
    sample(q,r, "r into ddot"); 
	  TICK(); ddot_sycl(q,nrow, r, r, rtrans);// 2*nrow ops
    one(q,rtrans, "cycle ddot"); 
    
    q->wait();
    TOCK(t1);
    
    q->submit([&](handler& h) {

    // sycl::stream out(655, 655, h);
    h.single_task([=]() {
     
     *beta = *rtrans / *oldrtrans;

 
    });
  }).wait();
 
  
	  
	  TICK(); waxpby_sycl(q,nrow, 1.0, r, *beta, p, p); 
    sample(q,p, "cycle wax");  
   
    q->wait();
    TOCK(t2);// 2*nrow ops 
    
    
  
	}
 one(q,rtrans, "residual b4 sqrt");
  q->submit([&](handler& h) {
   
    h.single_task([=]() {
    *normr_shared = sqrt(*rtrans);

    });
    
    
  }).wait();
 one(q,normr_shared, "residual");
  
  
        if (rank==0 && (k%print_freq == 0 || k+1 == max_iter)){
          std::cout << "Iteration = "<< k << "   Residual = "<< *normr_shared << std::endl;
         
        }
        
      
     
  // for (size_t i = 0; i < nrow; i++)
  // {
  //   if (i < nrow){
	// 		int cur_nnz = A->nnz_in_row[i];
	// 		double* cur_vals = A->ptr_to_vals_in_row[i];
	// 		int* cur_inds = A->ptr_to_inds_in_row[i];

	// 		for (size_t j = 0; j < cur_nnz; j++)
	// 		{
    
  //           std::cout << cur_vals[j]  <<",";
       
          
				 
	// 		}
	// 	std::cout << std::endl;
	// 	 }
  // }
  
   
  //    exit(0);
  TICK();HPC_sparsemv_sycl(q,A, p, Ap,nrow);q->wait();  TOCK(t3); // 2*nnz ops
  sample(q,Ap, "cycle spmv out");
  

      

      TICK(); sycl::event e_ddot = ddot_sycl(q,nrow, p, Ap, alpha);q->wait();TOCK(t1); // 2*nrow ops 
      one(q,alpha, "cycle ddot out");

           
           q->submit([&](handler& h) {
   
              h.single_task([=]() {
              *alpha = *rtrans/ *alpha;

              });
              
              
            }).wait();
            


      TICK(); waxpby_sycl_tasked(q,nrow, 1.0, x_device, *alpha, p, x_device, e_ddot); // 2*nrow ops 
      


      waxpby_sycl_tasked(q,nrow, 1.0, r, -(*alpha), Ap, r, e_ddot);  
      
      
      q->wait();
      sample(q,x_device, "wax task 1");
      sample(q,r, "wax task 2");
      TOCK(t2);// 2*nrow ops 
     
    
      niters = k;
      
    }
    

  // Store times
  times[1] = t1; // ddot time
  times[2] = t2; // waxpby time
  times[3] = t3; // sparsemv time
  times[4] = t4; // AllReduce time

  // exit(0);
  std::cout << "Begun Freeing"<< std::endl;
   sycl::free(r,*q);
   sycl::free(p,*q);
   sycl::free(Ap,*q);
   sycl::free(x_device,*q);
   sycl::free(b_device,*q);
   std::cout << "arrays free"<< std::endl;
  // sycl::free(A->nnz_in_row,*q);
  // std::cout << "nnz freee"<< std::endl;
  // sycl::free(A->ptr_to_vals_in_row,*q);
  // sycl::free(A->ptr_to_inds_in_row,*q);
  // sycl::free(A->ptr_to_diags,*q);
  
  // // Allocate arrays that are of length local_nnz
  // sycl::free(A->list_of_vals,*q);
  // sycl::free(A->list_of_inds,*q);
  std::cout << "fields free"<< std::endl;
  // sycl::free(A,*q);
  normr = *normr_shared;
  std::cout << "All Memory Free"<< std::endl;
  times[0] = mytimer() - t_begin;  // Total time. All done...
  return(0);
}




#endif





int HPCCG(HPC_Sparse_Matrix * A,
	  double * const b, double * const x,
	  const int max_iter, const double tolerance, int &niters, double & normr,
	  double * times)

{
  double t_begin = mytimer();  // Start timing right away

  double t0 = 0.0, t1 = 0.0, t2 = 0.0, t3 = 0.0, t4 = 0.0;
#ifdef USING_MPI
  double t5 = 0.0;
#endif
  int nrow = A->local_nrow;
  int ncol = A->local_ncol;

  double * r = new double [nrow];
  double * p = new double [ncol]; // In parallel case, A is rectangular
  double * Ap = new double [nrow];

  normr = 0.0;
  double rtrans = 0.0;
  double oldrtrans = 0.0;

#ifdef USING_MPI
  int rank; // Number of MPI processes, My process ID
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else
  int rank = 0; // Serial case (not using MPI)
#endif

  int print_freq = max_iter/10; 
  if (print_freq>50) print_freq=50;
  if (print_freq<1)  print_freq=1;

  // p is of length ncols, copy x to p for sparse MV operation
  TICK(); waxpby(nrow, 1.0, x, 0.0, x, p); TOCK(t2);
#ifdef USING_MPI
  TICK(); exchange_externals(A,p); TOCK(t5); 
#endif
  TICK(); HPC_sparsemv(A, p, Ap); TOCK(t3);
  TICK(); waxpby(nrow, 1.0, b, -1.0, Ap, r); TOCK(t2);
  TICK(); ddot(nrow, r, r, &rtrans, t4); TOCK(t1);
  normr = sqrt(rtrans);

  if (rank==0) cout << "Initial Residual = "<< normr << std::endl;

  for(int k=1; k<max_iter && normr > tolerance; k++ )
    {
      if (k == 1)
	{
	  TICK(); waxpby(nrow, 1.0, r, 0.0, r, p); TOCK(t2);
	}
      else
	{
	  oldrtrans = rtrans;
	  TICK(); ddot (nrow, r, r, &rtrans, t4); TOCK(t1);// 2*nrow ops
	  double beta = rtrans/oldrtrans;
	  TICK(); waxpby (nrow, 1.0, r, beta, p, p);  TOCK(t2);// 2*nrow ops
	}
      normr = sqrt(rtrans);
      if (rank==0 && (k%print_freq == 0 || k+1 == max_iter))
      cout << "Iteration = "<< k << "   Residual = "<< normr << std::endl;
     

#ifdef USING_MPI
      TICK(); exchange_externals(A,p); TOCK(t5); 
#endif
      TICK(); HPC_sparsemv(A, p, Ap); TOCK(t3); // 2*nnz ops
      double alpha = 0.0;
      TICK(); ddot(nrow, p, Ap, &alpha, t4); TOCK(t1); // 2*nrow ops
      alpha = rtrans/alpha;
      TICK(); waxpby(nrow, 1.0, x, alpha, p, x);// 2*nrow ops
      waxpby(nrow, 1.0, r, -alpha, Ap, r);  TOCK(t2);// 2*nrow ops
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
  delete [] p;
  delete [] Ap;
  delete [] r;
  times[0] = mytimer() - t_begin;  // Total time. All done...
  
  return(0 );
}
