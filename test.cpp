// #include<iostream>
// using namespace std;

// #include <CL/sycl.hpp>
// using namespace sycl;
// sycl::default_selector selector;
// sycl::queue q(selector);
// int main()
// {
//     double arr[729] = {540,-16,-16,-10,-16,-10,-10,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     -20,432,-20,-16,-10,-16,-16,-10,-16,-10,-1,-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     -16,540,-10,-16,-10,-16,-1,-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     -20,-16,432,-10,-20,-16,-16,-10,-10,-1,-16,-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
//     -20,-16,-20,-16,270,-16,-20,-16,-20,-16,-10,-16,-10,-1,-10,-16,-10,-16,0,0,0,0,0,0,0,0,0,-16,-20,-10,432,-16,-20,-10,-16,-1,-10,-10,-16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-16,-10,540,-16,-10,-1,-16,-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-16,-10,-16,-20,432,-20,-10,-1,-10,-16,-10,-16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-10,-16,-16,540,-1,-10,-10,-16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-20,-16,-16,-10,432,-10,-10,-1,-20,-16,-16,-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-20,-16,-20,-16,-10,-16,-16,270,-16,-10,-1,-10,-20,-16,-20,-16,-10,-16,0,0,0,0,0,0,0,0,0,-16,-20,-10,-16,-10,432,-1,-10,-16,-20,-10,-16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-20,-16,-16,-10,-20,-16,-16,-10,270,-1,-16,-10,-20,-16,-16,-10,-20,-16,0,0,0,0,0,0,0,0,0,-20,-16,-20,-16,-10,-16,-20,-16,-20,-16,-10,-16,-10,27,-10,-16,-10,-16,-20,-16,-20,-16,-10,-16,-20,-16,-20,-16,-20,-10,-16,-16,-20,-10,-16,-1,270,-10,-16,-16,-20,-10,-16,-16,-20,0,0,0,0,0,0,0,0,0,-16,-10,-20,-16,-10,-1,432,-10,-16,-10,-20,-16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-16,-10,-16,-20,-16,-20,-10,-1,-10,-16,270,-16,-16,-10,-16,-20,-16,-20,0,0,0,0,0,0,0,0,0,-10,-16,-16,-20,-1,-10,-10,432,-10,-16,-16,-20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-16,-10,-10,-1,540,-16,-16,-10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-16,-10,-16,-10,-1,-10,-20,432,-20,-16,-10,-16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-10,-16,-1,-10,-16,540,-10,-16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-16,-10,-10,-1,-16,-10,-20,-16,432,-10,-20,-16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-16,-10,-16,-10,-1,-10,-16,-10,-16,-20,-16,-20,-16,270,-16,-20,-16,-20,0,0,0,0,0,0,0,0,0,-10,-16,-1,-10,-10,-16,-16,-20,-10,432,-16,-20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-10,-1,-16,-10,-16,-10,540,-16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-10,-1,-10,-16,-10,-16,-16,-10,-16,-20,432,-20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,-10,-10,-16,-10,-16,-16,540,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
//     //# initialize data array using usm
//     double y[729];
//     // double y[27];

//     for (size_t i = 0; i < 729; i++)
//     {
//        y[i] = 0;

//     }
    
//     vector<double> temp_out(arr,arr+729);
//   	{
//     cl::sycl::buffer<double, 1> temp_out_sycl(temp_out.data(), temp_out.size());
//     cl::sycl::buffer<double, 1> y_sycl(y, cl::sycl::range<1>(729));
//     q.submit([&](sycl::handler &h)
// 		{
//        sycl::stream out(65535, 65535, h);
//   auto temp_out_acc = temp_out_sycl.get_access<cl::sycl::access::mode::read>(h);
// 	auto y_acc = y_sycl.get_access<cl::sycl::access::mode::write>(h);
// 	cl::sycl::range<1> range = temp_out_acc.get_range();
// 	size_t length = range[0];
// 	std::cout<<"size "<<length<<std::endl;
 
// 	h.parallel_for(sycl::nd_range<1>(729, 27), [=](sycl::nd_item<1> item) {
// 		auto sg = item.get_sub_group();
// 		int i = item.get_global_id(0);
// 		int j = item.get_local_id(0);
		
		
// 		//# Add all elements in sub_group using sub_group algorithm
// 		double result = sycl::reduce_over_group(sg, temp_out_acc[i], sycl::plus<>());
		
// 		// if (j == 0){
// 		// 	y_acc[i/27] = result;
// 		// }
	
//         if (sg.get_local_id()[0] == 0) {
//         y_acc[i] = result;
//         } else {
//         y_acc[i] = result;
//         }
// 		// printf("index: %d result %f",i/max_nnz,result);
// 		out << " i : " << i<< " j : " << j<< " result : " << result   << cl::sycl::endl;
//   });
// 		});

    
// }
// std::cout<<"OUTPUT "<<std::endl;
// for (size_t i = 0; i < 729; i++)
//     {
       
//        std::cout<<y[i]<<' ';
//     }
// }

#include <CL/sycl.hpp>
using namespace sycl;

static constexpr size_t N = 256; // global size
static constexpr size_t B = 64;  // work-group size


int main() {
queue q;
std::cout << "Device : " << q.get_device().get_info<info::device::name>() << "\n";
 //# get all supported sub_group sizes and print
  auto sg_sizes = q.get_device().get_info<info::device::sub_group_sizes>();
  std::cout << "Supported Sub-Group Sizes : ";
  for (int i=0; i<sg_sizes.size(); i++) std::cout << sg_sizes[i] << " "; std::cout << "\n";
    
  //# find out maximum supported sub_group size
  auto max_sg_size = std::max_element(sg_sizes.begin(), sg_sizes.end());
  std::cout << "Max Sub-Group Size        : " << max_sg_size[0] << "\n";
//# initialize data array using usm
int *data = malloc_shared<int>(N, q);
for (int i = 0; i < N; i++) data[i] = i%2 ? i:0;
for (int i = 0; i < N; i++) std::cout << data[i] << " ";
std::cout << "\n\n";

q.parallel_for(nd_range<1>(N, B), [=](nd_item<1> item) {
auto sg = item.get_sub_group();
auto i = item.get_global_id(0);

//# Add all elements in sub_group using sub_group algorithm
int result = reduce_over_group(sg, data[i], plus<>());

//# write sub_group sum in first location for each sub_group
if (sg.get_local_id()[0] == 0) {
    data[i] = result;
} else {
    data[i] = result;
}
}).wait();

for (int i = 0; i < N; i++) std::cout << data[i] << " ";
std::cout << "\n";

free(data, q);
return 0;
}