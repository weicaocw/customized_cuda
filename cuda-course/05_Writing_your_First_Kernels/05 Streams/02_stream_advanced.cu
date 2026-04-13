#include <cuda_runtime.h>
#include <stdio.h>
#include <iostream>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void kernel1(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 2.0f;
    }
}

__global__ void kernel2(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    printf("Stream callback: Operation completed\n");
}

// 死亡回调函数
void CUDART_CB badCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    // 危险操作：强制把 void* 转换回设备指针
    float* bad_ptr = (float*)userData; 
    
    // 💥 致命一击：CPU 试图读取 GPU 显存！
    // 操作系统瞬间判定越权访问，触发 Segmentation fault (core dumped)
    printf("Bad callback read: %f\n", bad_ptr[0]); 
}

// 拯救者回调函数
void CUDART_CB goodCallback(cudaStream_t stream, cudaError_t status, void *userData) {
    // 1. 安全转换：把 void* 转换回主机指针
    float* safe_ptr = (float*)userData; 
    
    // 2. 合法访问：CPU 读取它自己的内存
    printf("[Callback] Stream completed! The first element is now: %f\n", safe_ptr[0]); 
}

int main(void) {
    const int N = 1000000;
    size_t size = N * sizeof(float);
    float *h_data, *d_data;
    cudaStream_t stream1, stream2;
    cudaEvent_t event;

    // Allocate host and device memory
    CHECK_CUDA_ERROR(cudaMallocHost(&h_data, size));  // Pinned memory for faster transfers
    CHECK_CUDA_ERROR(cudaMalloc(&d_data, size));

    // Initialize data
    for (int i = 0; i < N; ++i) {
        h_data[i] = static_cast<float>(i);
    }

    // Create streams with different priorities
    int leastPriority, greatestPriority;
    CHECK_CUDA_ERROR(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream1, cudaStreamNonBlocking, leastPriority));
    CHECK_CUDA_ERROR(cudaStreamCreateWithPriority(&stream2, cudaStreamNonBlocking, greatestPriority));

    // Create event
    CHECK_CUDA_ERROR(cudaEventCreate(&event));
    std::cout << event << std::endl;

    // Asynchronous memory copy and kernel execution in stream1
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream1));
    kernel1<<<(N + 255) / 256, 256, 0, stream1>>>(d_data, N);

    // Record event in stream1
    CHECK_CUDA_ERROR(cudaEventRecord(event, stream1));

    // Make stream2 wait for event
    CHECK_CUDA_ERROR(cudaStreamWaitEvent(stream2, event, 0));

    // Execute kernel in stream2
    kernel2<<<(N + 255) / 256, 256, 0, stream2>>>(d_data, N);

    // Add callback to stream2
    // CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, myStreamCallback, NULL, 0));
    // 在 main 函数中的错误调用：
    // 传进去了 d_data (通过 cudaMalloc 申请的)
    // CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, badCallback, (void*)d_data, 0));
    // 在 main 函数中的正确调用：
    // 传进去了 h_data (通过 cudaMallocHost 申请的)
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, goodCallback, (void*)h_data, 0));

    // Asynchronous memory copy back to host
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream2));
    // 不用额外的同步机制，因为：Stream 是一个严格的 FIFO（First-In, First-Out，先进先出）硬件任务队列
    CHECK_CUDA_ERROR(cudaStreamAddCallback(stream2, goodCallback, (void*)h_data, 0));

    // Synchronize streams
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));

    // Verify result
    for (int i = 0; i < N; ++i) {
        float expected = (static_cast<float>(i) * 2.0f) + 1.0f;
        if (fabs(h_data[i] - expected) > 1e-5) {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }

    printf("Test PASSED\n");

    // Clean up
    CHECK_CUDA_ERROR(cudaFreeHost(h_data));
    CHECK_CUDA_ERROR(cudaFree(d_data));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    CHECK_CUDA_ERROR(cudaEventDestroy(event));

    return 0;
}