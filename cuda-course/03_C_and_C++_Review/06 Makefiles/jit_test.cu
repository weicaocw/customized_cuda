#include <iostream>
__global__ void jit_kernel() {
    printf("Hello from JIT compiled Kernel!\n");
}
int main() {
    jit_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}