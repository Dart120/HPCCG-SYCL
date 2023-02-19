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
int sum;
std::vector<int> data{1,1,1,1,1,1,1,1};
{


sycl::buffer<int> sum_buf(&sum,1);
sycl::buffer<int> data_buf(data);

q.submit([&](auto &h) {
      sycl::accessor buf_acc(data_buf, h, sycl::read_only);
     
      auto sumr =sycl::reduction(sum_buf,h, sycl::ext::oneapi::plus<>());

      h.parallel_for(sycl::range<1>{8}, sumr, [=](sycl::id<1> i, auto &sum) {
      
        sum += buf_acc[i];
      });
    });
}
    std::cout <<sum<<std::endl;

}