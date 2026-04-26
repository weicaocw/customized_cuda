#include <iostream>
#include <cuda_runtime.h>
#include <thread>  // 引入线程休眠库
#include <chrono>  // 引入时间库

__global__ void crash_kernel() {
    // 制造核弹级灾难：试图向空指针写入数据
    // int *null_ptr = nullptr;
    // *null_ptr = 42; 
}

int main() {
    std::cout << "1. 发射核函数 (制造内存崩溃)..." << std::endl;
    crash_kernel<<<1, 1>>>();
    
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "2. 捕获到致命错误: " << cudaGetErrorString(err) << std::endl; 
    }

    std::cout << "\n==================================================" << std::endl;
    std::cout << ">>> 案发现场已保护，CPU 进程强制暂停 15 秒 <<<" << std::endl;
    std::cout << "请立刻在另一个终端查看 nvidia-smi！" << std::endl;
    std::cout << "==================================================\n" << std::endl;

    // 倒计时 15 秒，让你有充足的时间切屏
    for (int i = 15; i > 0; --i) {
        std::cout << "倒计时: " << i << " 秒...\r" << std::flush;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }

    std::cout << "\n\n暂停结束，验证环境是否彻底崩塌..." << std::endl;
    
    void* test_ptr;
    err = cudaMalloc(&test_ptr, 1);
    if (err != cudaSuccess) {
        std::cerr << "3. 结论：上下文彻底崩塌！再次报错: " << cudaGetErrorString(err) << std::endl;
    } else {
        cudaFree(test_ptr);
    }

    return 0;
}