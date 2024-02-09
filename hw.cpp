#include <sycl/sycl.hpp>

#define N 10

int main()
{
    sycl::gpu_selector sel;
    sycl::queue q(sel);

    std::vector<double> x(N);
    for(auto& i : x) i=1.0;

    {
	    sycl::buffer<double,1> x_buf(x.data(), x.size());
	
	    q.submit([&](sycl::handler& cgh)
	    {
	        auto x_d = x_buf.get_access<sycl::access::mode::read_write>(cgh);
	
	        cgh.single_task([=]
	        {
	    		for(int i=0; i<N; i++)
				x_d[i]=2.0;
	        });
	    }).wait();
    }

    for(auto i : x) std::cout << i << " ";
    std::cout << std::endl;
}
