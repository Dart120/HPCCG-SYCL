
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
using std::cerr;
using std::cout;
using std::endl;
#include <cstdio>
#include <cstdlib>
#include <cctype>
#include <cassert>
#include <string>
#include <CL/sycl.hpp>
using namespace sycl;
#include <cmath>
#include "HPC_sparsemv.hpp"

std::vector<int> flatten_arrays(double **vals, int **idx, int *lengths, std::vector<double> *flat_vals, std::vector<int> *flat_idxs,int max_nnz,int nrow)
{
	
	std::vector<int> wrap_indicies;
	// for outer array
	int count = 0;
  
	for (int i = 0; i < nrow; ++i)
	{
		//  Iterate through lengths

		
		for (int j = 0; j < lengths[i]; ++j)
		{
			(*flat_vals).push_back(vals[i][j]);
			(*flat_idxs).push_back(idx[i][j]);
 
			wrap_indicies.push_back(i * max_nnz + j);
		}
	}
	return wrap_indicies;
}


int HPC_sparsemv(HPC_Sparse_Matrix *A,
				 const double *const x, double *const y)
{
const int nrow = (const int)A->local_nrow;
#ifdef USING_OMP
#pragma omp parallel for
#endif
	// For each row
	for (int i = 0; i < nrow; i++)
	{
    // std::cout <<"Row "<< i << std::endl;
		double sum = 0.0;
		const double *const cur_vals = (const double *const)A->ptr_to_vals_in_row[i];
		const int *const cur_inds = (const int *const)A->ptr_to_inds_in_row[i];
		const int cur_nnz = (const int)A->nnz_in_row[i];
		for (int j = 0; j < cur_nnz; j++)
		{
			// Sum = a non zero number in the row * the vector element at that same index
			sum += cur_vals[j] * x[cur_inds[j]];
		}
		y[i] = sum;
	}
	return (0);
}

int HPC_sparsemv_sycl(HPC_Sparse_Matrix *A,
				 const double *const x, double *const y)
{
	sycl::default_selector selector;
	sycl::queue q(selector);
	// std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
	const int nrow = (const int)A->local_nrow;
	double** pointer_to_cur_vals_lst = static_cast<double**>(sycl::malloc_device(sizeof(double*) * nrow, q));
	int** pointer_to_cur_inds_lst = static_cast<int**>(sycl::malloc_device(sizeof(int*) * nrow, q));
	// For each row
	for (int i = 0; i < nrow; i++)
	{
		pointer_to_cur_vals_lst[i] = static_cast<double*>(sycl::malloc_device(sizeof(double) * A->nnz_in_row[i], q));
		pointer_to_cur_inds_lst[i] = static_cast<int*>(sycl::malloc_device(sizeof(int) * A->nnz_in_row[i], q));
		q.memcpy(pointer_to_cur_vals_lst[i], A->ptr_to_vals_in_row[i], sizeof(double) * A->nnz_in_row[i]).wait();
		q.memcpy(pointer_to_cur_inds_lst[i], A->ptr_to_inds_in_row[i], sizeof(int) * A->nnz_in_row[i]).wait();
	}

	double* pointer_to_y = malloc_device<double>(nrow, q);
	int* pointer_to_cur_nnz = malloc_device<int>(nrow, q);
	q.memcpy(pointer_to_cur_nnz, A->nnz_in_row, sizeof(int) * nrow).wait();
	 q.parallel_for(range<1>(nrow), [=](id<1> i) {
		 int cur_nnz = pointer_to_cur_nnz[i];
		 double* cur_vals = pointer_to_cur_vals_lst[i];
		 int* cur_inds = pointer_to_cur_inds_lst[i];
		 double sum = 0.0;
		 for (size_t j = 0; j < cur_nnz; j++)
		 {
			sum += cur_vals[j] * x[cur_inds[j]];
		 }
		 pointer_to_y[i] = sum;
	}).wait();


	
  //# copy mem from device to host
  q.memcpy(y, pointer_to_y, sizeof(double) * nrow).wait();

  //# print output
	for (int i = 0; i < nrow; i++)
	{
		sycl::free(pointer_to_cur_vals_lst[i],q);
		sycl::free(pointer_to_cur_inds_lst[i],q);
	}
	sycl::free(pointer_to_cur_vals_lst,q);
	sycl::free(pointer_to_cur_inds_lst,q);

  return 0;
}




























































// // // EDITED CODE

// // Can we put the vals in lists of buffers
// // How many do we need?
// // iterate through nrow and add them
// // TODO: FROM MY TESTING THIS FUNCTION SEEMS TO WORK HOWEVER I THINK THE MATRIX IS STORED DIFFERENTLY FOR THE ACTUAL PROGRAM WHICH IS CAUSING THIS TO BREAK
// int HPC_sparsemv_sycl(HPC_Sparse_Matrix *A,
// 				 const double *const x, double *const y)
// {


// 	const int nrow = (const int)A->local_nrow;
// 	const int ncol = (const int)A->local_ncol;
// // 		std::cout << "xbf" << std::endl;
// // std::vector<int> vecx(x, x + ncol);
// // 	 for (auto i : vecx) {
// //         std::cout << i << " ";
// //     }
// 	// std::cout <<std::endl;
// 	// Because the matrix could be ragged under CSR we need the biggest value it can take so we know how big to make x
// 	// Use as local size
// 	int max_nnz = *std::max_element(A->nnz_in_row, A->nnz_in_row + nrow);
// 	std::vector<double> cur_vals_flat;
// 	std::vector<int> cur_inds_flat;
	
// 	std::vector<int> wrap_indicies = flatten_arrays(A->ptr_to_vals_in_row, A->ptr_to_inds_in_row, A->nnz_in_row, &cur_vals_flat, &cur_inds_flat, max_nnz,nrow);
	// sycl::default_selector selector;
	// sycl::queue q(selector);
  
// 	std::vector<double> temp_out(max_nnz * nrow, 0);


// 	{

// 		cl::sycl::buffer<double, 1> temp_out_sycl(temp_out.data(), temp_out.size());
// 		cl::sycl::buffer<double, 1> cur_vals_sycl(cur_vals_flat.data(), cur_vals_flat.size());
// 		cl::sycl::buffer<int, 1> cur_inds_sycl(cur_inds_flat.data(), cur_inds_flat.size());
// 		cl::sycl::buffer<int, 1> wrap_indicies_sycl(wrap_indicies.data(), wrap_indicies.size());
// 		cl::sycl::buffer<double, 1> x_sycl(x, cl::sycl::range<1>(nrow));
		
// 		q.submit([&](sycl::handler &h)
// 		{
			
// 			auto temp_out_acc = temp_out_sycl.get_access<cl::sycl::access::mode::write>(h);
// 			auto cur_vals_acc = cur_vals_sycl.get_access<cl::sycl::access::mode::read>(h);
// 			auto cur_inds_acc = cur_inds_sycl.get_access<cl::sycl::access::mode::read>(h);
// 			auto wrap_indicies_acc = wrap_indicies_sycl.get_access<cl::sycl::access::mode::read>(h);
// 			auto x_acc = x_sycl.get_access<cl::sycl::access::mode::read>(h);
			
// 			cl::sycl::range<1> range = cur_vals_acc.get_range();
// 			size_t length = range[0];
   
// 			h.parallel_for(sycl::range<1>(length), [=](sycl::id<1> i)
// 			{
// 				temp_out_acc[wrap_indicies_acc[i]] = cur_vals_acc[i]*x_acc[cur_inds_acc[i]];
// 			});
// 		}).wait();
// 	}


// std::cout << "output" << std::endl;
// 	 for (auto i : temp_out) {
//         std::cout << i << " ";
//     }
// // 	std::cout <<std::endl;
// // std::cout << "vals flat" << std::endl;
// // 	 for (auto i : cur_vals_flat) {
// //         std::cout << i << " ";
// //     }
// // 	std::cout <<std::endl;
// // std::cout << "inds flat" << std::endl;
// // 	 for (auto i : cur_inds_flat) {
// //         std::cout << i << " ";
// //     }
// // 	std::cout <<std::endl;
// // std::cout << "wrap indicies" << std::endl;
// // 	 for (auto i : wrap_indicies) {
// //         std::cout << i << " ";
// //     }
// // 	std::cout <<std::endl;
// // std::cout << "x" << std::endl;
// // std::vector<int> vec(x, x + ncol);
// // 	 for (auto i : vec) {
// //         std::cout << i << " ";
// //     }
// // 	std::cout <<std::endl;
	







// 		{
// 			cl::sycl::buffer<double, 1> temp_out_sycl(temp_out.data(), temp_out.size());
// 			cl::sycl::buffer<double, 1> y_sycl(y, cl::sycl::range<1>(nrow));
//     q.submit([&](sycl::handler &h)
// 		{
//        sycl::stream out(65535, 65535, h);
//   auto temp_out_acc = temp_out_sycl.get_access<cl::sycl::access::mode::read>(h);
// 	auto y_acc = y_sycl.get_access<cl::sycl::access::mode::write>(h);
// 	cl::sycl::range<1> range = temp_out_acc.get_range();
// 	size_t length = range[0];
// 	std::cout<<"size "<<length<<std::endl;
 
// 	h.parallel_for(sycl::nd_range<1>(length, max_nnz), [=](sycl::nd_item<1> item) {
// 		auto sg = item.get_sub_group();
// 		int i = item.get_global_id(0);
// 		int j = item.get_local_id(0);
		
		
// 		//# Add all elements in sub_group using sub_group algorithm
// 		double result = sycl::reduce_over_group(sg, temp_out_acc[i], sycl::plus<>());
// 		if (j == 0){
// 			y_acc[i/max_nnz] = result;
// 		}
		
// 		// printf("index: %d result %f",i/max_nnz,result);
// 		out << " i : " << i<< " j : " << j<< " result : " << result << " thing : " << sg.get_local_id()[0]  << cl::sycl::endl;
//   });
// 		});


 
//   }
// 	// std::cout <<"sum "<< y[0]<< std::endl;
// 	return 0;
// 	}

	
