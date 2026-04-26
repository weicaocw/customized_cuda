#include <iostream>

/**
 * Usage:
 * # 这行命令的意思是：只生成 compute_61 的 PTX，并且把这段 PTX 打包进去。绝不生成任何具体的 sm_xx 机器码！
 * nvcc -arch=compute_86 -code=compute_86 jit_test.cu -o jit_test
 * 
 # 证据 1：证明存在 PTX
 * cuobjdump -ptx jit_test
 * 你会看到一大串虚拟汇编代码，开头写着： .version 7.x  .target sm_61
 * 证据 2：证明不存在 SASS（机器码）
 * cuobjdump -sass jit_test
 * 
 * # 执行程序
 * ./jit_test
 * # 输出：Hello from JIT compiled Kernel!
 * 当你执行 ./jit_test 时，CUDA 驱动发现文件里没有 SASS，
 * 但找到了兼容的 compute_86 PTX。
 * 于是，显卡驱动程序在千分之一秒内，在后台自动调用了驱动内置的编译器（JIT Compiler）
 * 将 PTX 翻译成了你当前 RTX 3060 的 SASS，并缓存在了系统的 ~/.nv/ComputeCache 目录下，然后再发给 GPU 运行
 */
__global__ void jit_kernel() {
    printf("Hello from JIT compiled Kernel!\n");
}
int main() {
    jit_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}