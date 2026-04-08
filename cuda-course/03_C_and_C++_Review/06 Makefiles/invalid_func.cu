#include <iostream>
#include <cuda_runtime.h>

// 一个在 GPU 上执行的子函数
__device__ void real_device_function() {
    printf("This is a real device function.\n");
}

// 定义一个函数指针类型
typedef void(*func_ptr_t)();

// 致命错误操作：在 CPU (Host) 端，试图获取 Device 函数的指针，并直接传给 Kernel
__global__ void trigger_kernel(func_ptr_t fptr) {
    // GPU 试图跳转到一个只有 CPU 才知道的地址，或者未能正确解析的地址
    fptr(); 
}

int main() {
    // 获取设备函数的地址 (如果不使用 cudaGetSymbolAddress 而是直接传名，会导致指针失效)
    func_ptr_t host_side_ptr = real_device_function; 
    
    trigger_kernel<<<1, 1>>>(host_side_ptr);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "抓到错误: " << cudaGetErrorString(err) << std::endl;
    }
    return 0;
}