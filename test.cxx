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
#include <cmath>



struct HPC_Sparse_Matrix_STRUCT {
  char   *title;
  int start_row;
  int stop_row;
  int total_nrow;
  long long total_nnz;
  int local_nrow;
  int local_ncol;  // Must be defined in make_local_matrix
  int local_nnz;
  int  * nnz_in_row;
  double ** ptr_to_vals_in_row;
  int ** ptr_to_inds_in_row;
  double ** ptr_to_diags;
  double *list_of_vals;   //needed for cleaning up memory
  int *list_of_inds;      //needed for cleaning up memory

};

typedef struct HPC_Sparse_Matrix_STRUCT HPC_Sparse_Matrix;


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


int HPC_sparsemv_old(HPC_Sparse_Matrix *A,
				 const double *const x, double *const y)
{

	const int nrow = (const int)A->local_nrow;

#ifdef USING_OMP
#pragma omp parallel for
#endif
	// For each row
	for (int i = 0; i < nrow; i++)
	{
		double sum = 0.0;
		const double *const cur_vals = (const double *const)A->ptr_to_vals_in_row[i];

		const int *const cur_inds = (const int *const)A->ptr_to_inds_in_row[i];
		// Number of non zeros
		const int cur_nnz = (const int)A->nnz_in_row[i];
		// For each number in the row
		for (int j = 0; j < cur_nnz; j++)
		{
			// Sum = a non zero number in the row * the vector element at that same index

			sum += cur_vals[j] * x[cur_inds[j]];
		}
		y[i] = sum;
	}
	return (0);
}

// // EDITED CODE

// Can we put the vals in lists of buffers
// How many do we need?
// iterate through nrow and add them
// TODO: MAKE A FAKE A X AND Y AND TEST WHY THIS DOESNT WORK
int HPC_sparsemv_new(HPC_Sparse_Matrix *A,
				 const double *const x, double *const y)
{

	const int nrow = (const int)A->local_nrow;
	// Because the matrix could be ragged under CSR we need the biggest value it can take so we know how big to make x
	// Use as local size
	int max_nnz = *std::max_element(A->nnz_in_row, A->nnz_in_row + nrow);
	std::vector<double> cur_vals_flat;
	std::vector<int> cur_inds_flat;
	
	std::vector<int> wrap_indicies = flatten_arrays(A->ptr_to_vals_in_row, A->ptr_to_inds_in_row, A->nnz_in_row, &cur_vals_flat, &cur_inds_flat, max_nnz,nrow);
  // std::cout << "wrap_indicies ";
  // for (int value : wrap_indicies) {
  //   std::cout << value << " ";
  // }
  // std::cout << "\nvals ";
  // for (int value : cur_vals_flat) {
  //   std::cout << value << " ";
  // }
  // std::cout << "\ninds ";
  // for (int value : cur_inds_flat) {
  //   std::cout << value << " ";
  // }
	sycl::default_selector selector;
	sycl::queue q(selector);
  std::cout <<' '<< max_nnz <<' '<< nrow << ' ' << endl;
	std::vector<double> temp_out(max_nnz * nrow, 0);

	{

		cl::sycl::buffer<double, 1> temp_out_sycl(temp_out.data(), temp_out.size());
		cl::sycl::buffer<double, 1> cur_vals_sycl(cur_vals_flat.data(), cur_vals_flat.size());
		cl::sycl::buffer<int, 1> cur_inds_sycl(cur_inds_flat.data(), cur_inds_flat.size());
		cl::sycl::buffer<int, 1> wrap_indicies_sycl(wrap_indicies.data(), wrap_indicies.size());
		cl::sycl::buffer<double, 1> x_sycl(x, cl::sycl::range<1>(nrow));
		cl::sycl::buffer<double, 1> y_sycl(y, cl::sycl::range<1>(nrow));
		q.submit([&](sycl::handler &h)
		{
			sycl::stream out(1024, 256, h);
			auto temp_out_acc = temp_out_sycl.get_access<cl::sycl::access::mode::write>(h);
			auto cur_vals_acc = cur_vals_sycl.get_access<cl::sycl::access::mode::read>(h);
			auto cur_inds_acc = cur_inds_sycl.get_access<cl::sycl::access::mode::read>(h);
			auto wrap_indicies_acc = wrap_indicies_sycl.get_access<cl::sycl::access::mode::read>(h);
			auto x_acc = x_sycl.get_access<cl::sycl::access::mode::read>(h);
			
			cl::sycl::range<1> range = cur_vals_acc.get_range();
			size_t length = range[0];
   
      
			// for each row
			h.parallel_for(sycl::range<1>(length), [=](sycl::id<1> i)
			{
        
	// 			// for val in row
	// 			for (int j=0; j< cur_nnz; j++){
	// //           // Sum = a non zero number in the row * the vector element at that same index
	// 				sum += cur_vals_acc[j]*x_acc[cur_inds_acc[j]];
	// 				counter
	// 			}
// wrap_indicies 0 1 2 3 4 
// vals 1 2 3 4 5 
// inds 0 1 0 2 2 
        // out << "put into index " << wrap_indicies_acc[i] <<' ' <<  cur_vals_acc[i]  << " times by " <<  x_acc[cur_inds_acc[i]] << cl::sycl::endl; 
  
     
				temp_out_acc[wrap_indicies_acc[i]] = cur_vals_acc[i]*x_acc[cur_inds_acc[i]];

			});
		}).wait();
    // Results
    std::cout << "\ntemp_out ";

  for (int value : temp_out) {
    std::cout << value << " ";
  }
 std::cout << std::endl;

		// final
		{
    q.submit([&](sycl::handler &h)
		{
       sycl::stream out(1024, 256, h);
  auto temp_out_acc = temp_out_sycl.get_access<cl::sycl::access::mode::write>(h);
	auto y_acc = y_sycl.get_access<cl::sycl::access::mode::write>(h);
  // cl::sycl::buffer<double, 1> local_result_sycl(local_result, cl::sycl::range<1>(n));

	cl::sycl::range<1> range = temp_out_acc.get_range();
	size_t length = range[0];
 
	h.parallel_for(sycl::nd_range<1>(length, max_nnz), [=](sycl::nd_item<1> item) {
		auto sg = item.get_sub_group();
		int i = item.get_global_id(0);
		int j = item.get_local_id(0);
		
		
		//# Add all elements in sub_group using sub_group algorithm
		int result = sycl::reduce_over_group(sg, temp_out_acc[i], sycl::plus<>());
    // out << " sg.get_local_id()[0]: " << sg.get_local_id()[0] << " i: " << i<< " j: " << j <<" result: "<<result << sycl::endl;
		//# write sub_group sum in first location for each sub_group
		
		y_acc[i/max_nnz] = result;
	
  });
		});

 
  }
	
	}

	return 0;
}



int main() {
  HPC_Sparse_Matrix A;
  int nnz_in_row[5] = {2,2,5,3,1};
  A.nnz_in_row = nnz_in_row;
  double valrow0[2] = {1,2};
  double valrow1[2] = {3,4};
  double valrow2[5] = {5,6,7,8,9};
  double valrow3[3] = {3,4,5};
  double valrow4[1] = {5};
  int indsrow0[2] = {0,1};
  int indsrow1[2] = {0,4};
  int indsrow2[5] = {0,1,2,3,4};
  int indsrow3[3] = {2,3,4};
  int indsrow4[1] = {4};

  double* ptr_to_vals_in_row[5] = {valrow0,valrow1,valrow2,valrow3,valrow4};
  int* ptr_to_inds_in_row[5] = {indsrow0,indsrow1,indsrow2,indsrow3,indsrow4};
  A.ptr_to_vals_in_row = ptr_to_vals_in_row;
  A.ptr_to_inds_in_row = ptr_to_inds_in_row;
 
  A.local_nrow = 5;

  double x[A.local_nrow];
  for (size_t i = 0; i < A.local_nrow; i++)
  {
    x[i] = i;
  }
  
  double y[A.local_nrow];


  HPC_sparsemv_new(&A, x, y);
  for (size_t i = 0; i < A.local_nrow; i++)
  {
    std::cout << y[i] << std::endl;
  }
      
}