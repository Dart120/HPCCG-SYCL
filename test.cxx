#include <CL/sycl.hpp>
#include <stdio.h>

constexpr int N = 100;

int main() {
  int n = 33;
  double x[33];
  double y[33];
  double w[33];
  for (int i = 0; i < n; i ++){
    x[i] = 1;
    y[i] = 2;
    w[i] = 0;
  }
  sycl::default_selector selector;
  sycl::queue q(selector);
  
  

  {
    
  cl::sycl::buffer<double, 1> x_sycl(x, cl::sycl::range<1>(n));
  cl::sycl::buffer<double, 1> y_sycl(y, cl::sycl::range<1>(n));
  cl::sycl::buffer<double, 1> w_sycl(w, cl::sycl::range<1>(n));

  // cl::sycl::buffer<double, 1> local_result_sycl(local_result, cl::sycl::range<1>(n));



 q.submit([&](sycl::handler& h) {
  auto x_acc = x_sycl.get_access<cl::sycl::access::mode::read>(h);
  auto y_acc = y_sycl.get_access<cl::sycl::access::mode::read>(h);
  auto w_acc = w_sycl.get_access<cl::sycl::access::mode::write>(h);
  h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
      
      w_acc[i] = x_acc[i] + y_acc[i];
  }); 
 });
 
  }
  std::cout << w[0] << std::endl;


  return(0);
}