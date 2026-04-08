#include <cuda_runtime.h>
#include <iostream>

// 这是一个在 GPU 上执行的核函数
__global__ void hello_from_gpu() {
    printf("Hello from GPU! 我正在底层硬件上运行！\n");
}

int main() {
    std::cout << "--- 程序启动 ---" << std::endl;
    
    // 启动核函数 (1个线程块，1个线程)
    hello_from_gpu<<<1, 1>>>();
    
    // 关键！捕获核函数启动阶段的错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[致命错误] 核函数发射失败: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "[成功] 核函数已成功发射到 GPU！" << std::endl;
    }
    
    // 同步主机和设备，等待 GPU 打印完成，同时捕获运行时的异步错误
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "[同步错误] GPU 运行中断: " << cudaGetErrorString(err) << std::endl;
    }
    
    std::cout << "--- 程序结束 ---" << std::endl;
    return 0;
}