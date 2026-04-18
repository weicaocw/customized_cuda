#include <iostream>
#include <cuda_runtime.h>

// 1. 定义设备函数
__device__ void real_device_function() {
    printf("终于成功了！This is a real device function.\n");
}

typedef void(*func_ptr_t)();

// 2. 【核心魔法】在设备端（GPU的全局内存中）定义一个函数指针变量，并用函数初始化它！
// 因为初始化发生在设备端，GPU 的编译器和链接器知道它自己真实的地址。
__device__ func_ptr_t d_ptr = real_device_function;

__global__ void trigger_kernel(func_ptr_t fptr) {
    fptr(); 
}

int main() {
    func_ptr_t h_ptr; // 主机端的变量，用来接收真实的 GPU 地址

    // 3. 将设备端查好的真实地址，拷贝到主机端
    // cudaMemcpyFromSymbol 专门用于读取设备的全局变量
    cudaMemcpyFromSymbol(&h_ptr, d_ptr, sizeof(func_ptr_t));
    
    // 4. 发射！现在 h_ptr 里装的是真正的、东京司机能看懂的东京地址了
    trigger_kernel<<<1, 1>>>(h_ptr);
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "抓到错误: " << cudaGetErrorString(err) << std::endl; 
    }
    return 0;
}