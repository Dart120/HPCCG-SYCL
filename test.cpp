#include <CL/sycl.hpp>
using namespace cl::sycl;
struct MyStruct {
    int value;
};

void manipulateStruct(MyStruct** ptrToDeviceA, queue& q) {
    // Allocate new instance of MyStruct in device memory
    MyStruct* newDeviceA = malloc_device<MyStruct>(1, q);
    
    // Initialize the new instance with a kernel
    q.submit([&](handler& h) {
        h.single_task([=]() {
            newDeviceA->value = 100; 
        });
    }).wait();

    // If ptrToDeviceA is already pointing to a device allocation, free it
    if (*ptrToDeviceA != nullptr) {
        free(*ptrToDeviceA, q);
    }

    // Redirect ptrToDeviceA to the new device memory allocation
    *ptrToDeviceA = newDeviceA;

    size_t bufferSize = 1024;
    size_t maxStatementSize = 256;
    // This causes an error....
    q.submit([&](handler& h) {
    stream out(256, 1024, h);
        h.single_task([=]() {
          out << (*ptrToDeviceA)->value << sycl::endl;
          
        });
    }).wait();
}
int main() {
    queue q;

    // Pointer initially meant for host memory but is nullptr
    MyStruct* deviceA = malloc_device<MyStruct>(1, q);

    // Use manipulateStruct to allocate and initialize the struct in device memory
    manipulateStruct(&deviceA, q);

    //This is completely fine!
    q.submit([&](handler& h) {
    stream out(256, 1024, h);
        h.single_task([=]() {
          out << "This many nnz: " << deviceA->value << sycl::endl;
         
        });
    }).wait();

    free(deviceA, q);
}
