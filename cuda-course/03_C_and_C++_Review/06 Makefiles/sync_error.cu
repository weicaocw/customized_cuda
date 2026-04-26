#include <iostream>

__global__ void crash_kernel() {
    // 制造一场灾难：试图向空指针写入数据
    int *null_ptr = nullptr;
    *null_ptr = 42; 
}

int main() {
    std::cout << "1. 开始发射..." << std::endl;
    crash_kernel<<<1, 1>>>(); // 发射瞬间是没问题的，因为参数合法
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cout << "发射期捕获: " << cudaGetErrorString(err) << std::endl;
    } else {
        std::cout << "2. 发射成功！CPU继续往下走..." << std::endl;
    }
    
    std::cout << "3. CPU 等待 GPU 完成..." << std::endl;
    // 此时 GPU 已经在里面崩溃了
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // 在这里，等待结束的 CPU 收到了 GPU 的死亡通知
        std::cerr << "4. 运行期捕获到致命错误: " << cudaGetErrorString(err) << std::endl; // unspecified launch failure
    }
    return 0;
}