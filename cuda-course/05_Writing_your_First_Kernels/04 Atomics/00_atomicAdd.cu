#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

#define NUM_THREADS 1000
#define NUM_BLOCKS 1000

// Kernel without atomics (incorrect)
__global__ void incrementCounterNonAtomic(unsigned long long* counter) {
    for (int i = 0; i < 10000; i++) {
    // not locked
    int old = *counter;
    int new_value = old + 1;
    // not unlocked
    *counter = new_value;
    }
}

// Kernel with atomics (correct)
__global__ void incrementCounterAtomic(unsigned long long* counter) {
    for (int i = 0; i < 10000; i++) {
    int a = atomicAdd(counter, 1);
    }
}

// Function to measure execution time
double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    unsigned long long h_counterNonAtomic = 0;
    unsigned long long h_counterAtomic = 0;
    unsigned long long *d_counterNonAtomic, *d_counterAtomic;

    // Allocate device memory
    cudaMalloc((void**)&d_counterNonAtomic, sizeof(unsigned long long));
    cudaMalloc((void**)&d_counterAtomic, sizeof(unsigned long long));

    // Copy initial counter values to device
    cudaMemcpy(d_counterNonAtomic, &h_counterNonAtomic, sizeof(unsigned long long), cudaMemcpyHostToDevice);
    cudaMemcpy(d_counterAtomic, &h_counterAtomic, sizeof(unsigned long long), cudaMemcpyHostToDevice);

    // Launch kernels
    double start_time = get_time();
    incrementCounterNonAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterNonAtomic);
    cudaDeviceSynchronize();
    double end_time = get_time();
    double non_atomic_time = end_time - start_time;
    start_time = get_time();
    incrementCounterAtomic<<<NUM_BLOCKS, NUM_THREADS>>>(d_counterAtomic);
    cudaDeviceSynchronize();
    end_time = get_time();
    double atomic_time = end_time - start_time;

    // Copy results back to host
    cudaMemcpy(&h_counterNonAtomic, d_counterNonAtomic, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_counterAtomic, d_counterAtomic, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Print results
    printf("Non-atomic counter value: %llu\n", h_counterNonAtomic);
    printf("Atomic counter value: %llu\n", h_counterAtomic);
    printf("Non-atomic time: %f seconds\n", non_atomic_time); // 编译器做了优化，最后一次才写入显存
    printf("Atomic time: %f seconds\n", atomic_time); 
    printf("Speedup(Non-atomic should be faster): %f\n", atomic_time / non_atomic_time); // 1065.293888

    // Free device memory
    cudaFree(d_counterNonAtomic);
    cudaFree(d_counterAtomic);

    return 0;
}