#include <CL/sycl.hpp>
#include <stdio.h>

constexpr int N = 100;

int main() {
  int n = 33;
  double x[33];
  double y[33];
  for (int i = 0; i < n; i ++){
    x[i] = 1;
    y[i] = 2;
  }
  double local_result = 0.0;
  sycl::default_selector selector;
  auto R = sycl::range<1>(n);
  sycl::queue q(selector);
  int max_group_size = q.get_device().get_info<sycl::info::device::native_vector_width_int>();
   auto local_mem_size = q.get_device().get_info<sycl::info::device::local_mem_size>();
    
    size_t wgroup_size = 32;
    auto part_size = wgroup_size * 2;
    auto n_wgroups = (n + part_size - 1) / part_size;
    std::cout << n_wgroups << std::endl;

  {
    
  cl::sycl::buffer<double, 1> x_sycl(x, cl::sycl::range<1>(n));
  cl::sycl::buffer<double, 1> y_sycl(y, cl::sycl::range<1>(n));
  cl::sycl::buffer<double, 1> local_result_sycl(&local_result, cl::sycl::range<1>(1));

  // cl::sycl::buffer<double, 1> local_result_sycl(local_result, cl::sycl::range<1>(n));



 q.submit([&](sycl::handler& h) {
          auto x_acc = x_sycl.get_access<cl::sycl::access::mode::read>(h);
          auto y_acc = y_sycl.get_access<cl::sycl::access::mode::read>(h);
          auto local_result_acc = local_result_sycl.get_access<cl::sycl::access::mode::read_write>(h);
          auto sumr = sycl::ext::oneapi::reduction(local_result_acc, sycl::ext::oneapi::plus<>());
          h.parallel_for(sycl::nd_range<1>{n_wgroups * wgroup_size, wgroup_size}, sumr, [=](sycl::nd_item<1> item, auto &sumr_arg) {
             int i = item.get_global_id(0);
              sumr_arg += x_acc[i] * y_acc[i];
          }); 
 });
 
  }
  std::cout << local_result << std::endl;


  return(0);
}