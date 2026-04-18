#include <iostream>
#include <cuda_runtime.h>
#include <thread>
#include <chrono>

__global__ void empty_kernel() {
    // 制造核弹级灾难：试图向空指针写入数据
    // int *null_ptr = nullptr;
    // *null_ptr = 42; 
}

int main() {
    std::cout << "1. 发射正常的空核函数 (建立上下文)..." << std::endl;
    empty_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    
    std::cout << "2. 分配 5GB 巨型显存，作为进程存活的‘物理定海神针’..." << std::endl;
    void* massive_ptr;
    // 分配 5 * 1024 * 1024 * 1024 字节
    cudaError_t err = cudaMalloc(&massive_ptr, 5ULL * 1024 * 1024 * 1024); 
    if (err != cudaSuccess) {
        std::cerr << "分配失败: " << cudaGetErrorString(err) << std::endl;
        // return -1;
    }

    std::cout << "\n==================================================" << std::endl;
    std::cout << ">>> 显存已死死咬住！CPU 休眠 15 秒 <<<" << std::endl;
    std::cout << "请去 nvidia-smi 看 Memory-Usage，一定会暴涨到 5000+ MiB！" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    for (int i = 15; i > 0; --i) {
        std::cout << "倒计时: " << i << " 秒...\r" << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    
    // 释放并结束
    cudaFree(massive_ptr);
    std::cout << "\n程序正常结束，办公室完好无损！" << std::endl;
    return 0;
}