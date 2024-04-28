
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

// Main routine of a program that reads a sparse matrix, right side
// vector, solution vector and initial guess from a file  in HPC
// format.  This program then calls the HPCCG conjugate gradient
// solver to solve the problem, and then prints results.

// Calling sequence:

// test_HPCCG linear_system_file

// Routines called:

// read_HPC_row - Reads in linear system

// mytimer - Timing routine (compile with -DWALL to get wall clock
//           times

// HPCCG - CG Solver

// compute_residual - Compares HPCCG solution to known solution.

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
#include <chrono>
#ifdef USING_MPI
#include <mpi.h> // If this routine is compiled with -DUSING_MPI
                 // then include mpi.h
#include "make_local_matrix.hpp" // Also include this function
#endif
#ifdef USING_OMP
#include <omp.h>
#endif
#ifdef USING_SYCL
#include <CL/sycl.hpp>
#endif
#include "generate_matrix.hpp"
#include "read_HPC_row.hpp"
#include "mytimer.hpp"
#include "HPC_sparsemv.hpp"
#include "compute_residual.hpp"
#include "HPCCG.hpp"
#include "HPC_Sparse_Matrix.hpp"

#include "YAML_Element.hpp"
#include "YAML_Doc.hpp"

#undef DEBUG

int main(int argc, char *argv[])
{
  
  HPC_Sparse_Matrix *A;
  double *x, *b, *xexact;
  double norm, d;
  int ierr = 0;
  int i, j;
  int ione = 1;
  double times[7];
  double t6 = 0.0;
  int nx,ny,nz;
  
  #ifdef USING_SYCL

  if(argc!=5) {
    
      cerr << "Usage:" << endl
	   << "Mode 1: " << argv[0] << " nx ny nz --[cpu|gpu]" << endl
	   << "     where nx, ny and nz are the local sub-block dimensions and the flag is the device you want to use" << endl;
    exit(1);
  }
  bool selector_chosen = false;
  std::unique_ptr<sycl::device_selector> selector;


    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--cpu") == 0) {
           
            selector = std::make_unique<sycl::cpu_selector>();
            selector_chosen = true;
        } else if (strcmp(argv[i], "--gpu") == 0) {

            selector = std::make_unique<sycl::gpu_selector>();
            selector_chosen = true;
        }
    }

    if (!selector_chosen) {
        std::cerr << "Usage:" << std::endl
                  << "Mode 1: " << argv[0] << " nx ny nz --[cpu|gpu]" << std::endl
                  << "     where nx, ny, and nz are the local sub-block dimensions and the flag is the device you want to use" << std::endl;
        exit(1);
    }

    // Example of using the selector
    sycl::queue q(*selector);


  #else
    if(argc!=4) {
      
        cerr << "Usage:" << endl
      << "Mode 1: " << argv[0] << " nx ny nz" << endl
      << "     where nx, ny and nz are the local sub-block dimensions" << endl;
      exit(1);
    }
  #endif
  int size = 1; // Serial case (not using MPI)
  int rank = 0; 



  


  nx = atoi(argv[1]);
  ny = atoi(argv[2]);
  nz = atoi(argv[3]);
  #ifdef USING_SYCL
  generate_matrix_sycl(&q,nx, ny, nz, &A, &x, &b, &xexact);
  
  #else
  generate_matrix(nx, ny, nz, &A, &x, &b, &xexact);
  #endif
  



  double t1 = mytimer();   // Initialize it (if needed)
  int niters = 0;
  double normr = 0.0;
  int max_iter = 500;
  double tolerance = 0.0; // Set tolerance to zero to make all runs do max_iter iterations
  auto start = std::chrono::high_resolution_clock::now();
  #ifdef USING_SYCL
  ierr = HPCCG_sycl(&q, A, b, x, max_iter, tolerance, niters, normr, times);
  #else
  ierr = HPCCG(A, b, x, max_iter, tolerance, niters, normr, times);
  #endif

	if (ierr) cerr << "Error in call to CG: " << ierr << ".\n" << endl;


// initialize YAML doc

  if (rank==0)  // Only PE 0 needs to compute and report timing results
    {
      
      double fniters = niters; 
   
      double fnrow = A->total_nrow;
    
      
      double fnnz = A->total_nnz;
      
      double fnops_ddot = fniters*4*fnrow;
      double fnops_waxpby = fniters*6*fnrow;
      double fnops_sparsemv = fniters*2*fnnz;
      double fnops = fnops_ddot+fnops_waxpby+fnops_sparsemv;
      

      YAML_Doc doc("hpccg", "1.0");

      doc.add("Parallelism","");
      doc.add("Device Capability","");



#ifdef USING_OMP
          int nthreads = 1;
#pragma omp parallel
          nthreads = omp_get_num_threads();
          doc.get("Parallelism")->add("Number of OpenMP threads",nthreads);
#else
          doc.get("Parallelism")->add("OpenMP not enabled","");
#endif
#ifdef USING_SYCL
         sycl::device d = selector->select_device();
        doc.get("Device Capability")->add("Device Name", q.get_device().get_info<sycl::info::device::name>());
        doc.get("Device Capability")->add("Maximum Work-group Size", q.get_device().get_info<sycl::info::device::max_work_group_size>());
        doc.get("Device Capability")->add("Global Memory Size", std::to_string(q.get_device().get_info<sycl::info::device::global_mem_size>()/1e+9) + " gb");
        doc.get("Device Capability")->add("Local Memory Size", std::to_string(q.get_device().get_info<sycl::info::device::local_mem_size>()/1e+3) + " kb");
        doc.get("Parallelism")->add("Number of SYCL compute units",(int) d.get_info<sycl::info::device::max_compute_units>());
#else
          doc.get("Parallelism")->add("SYCL not enabled","");
#endif

      doc.add("Dimensions","");
	  doc.get("Dimensions")->add("nx",nx);
	  doc.get("Dimensions")->add("ny",ny);
	  doc.get("Dimensions")->add("nz",nz);



      doc.add("Number of iterations", niters);
      doc.add("Final residual", normr);
      doc.add("#********** Performance Summary (times in sec) ***********","");
 
      doc.add("Time Summary","");
      doc.get("Time Summary")->add("Total   ",times[0]);
      doc.get("Time Summary")->add("DDOT    ",times[1]);
      doc.get("Time Summary")->add("WAXPBY  ",times[2]);
      doc.get("Time Summary")->add("SPARSEMV",times[3]);

      doc.add("FLOPS Summary","");
      doc.get("FLOPS Summary")->add("Total   ",fnops);
      doc.get("FLOPS Summary")->add("DDOT    ",fnops_ddot);
      doc.get("FLOPS Summary")->add("WAXPBY  ",fnops_waxpby);
      doc.get("FLOPS Summary")->add("SPARSEMV",fnops_sparsemv);

      doc.add("MFLOPS Summary","");
      doc.get("MFLOPS Summary")->add("Total   ",fnops/times[0]/1.0E6);
      doc.get("MFLOPS Summary")->add("DDOT    ",fnops_ddot/times[1]/1.0E6);
      doc.get("MFLOPS Summary")->add("WAXPBY  ",fnops_waxpby/times[2]/1.0E6);
      doc.get("MFLOPS Summary")->add("SPARSEMV",fnops_sparsemv/(times[3])/1.0E6);

  
   
      std::string yaml = doc.generateYAML();
      cout << yaml;
 
    }

  #ifdef USING_SYCL
  sycl::free(xexact,q);
  sycl::free(A,q);
  #endif


  // Finish up
  return 0 ;
} 
