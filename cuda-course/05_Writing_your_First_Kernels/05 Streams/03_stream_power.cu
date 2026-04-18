#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define CHECK(call) \
{ \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}

// 一个非常“重”的核函数，用来模拟真实的复杂计算
__global__ void heavyCompute(float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float val = in[idx];
        // 强行增加计算量，让 GPU 的 SM 忙起来
        for (int i = 0; i < 50; i++) {
            val = sinf(val) * cosf(val) + 1.0f;
        }
        out[idx] = val;
    }
}

int main() {
    // 1. 准备数据：1000 万个浮点数 (约 40MB)
    const int N = 10000000;
    const int bytes = N * sizeof(float);
    
    // 划分为 4 个流水线批次
    const int num_streams = 4;
    const int chunk_size = N / num_streams;
    const int chunk_bytes = chunk_size * sizeof(float);

    float *h_in, *h_out;
    float *d_in, *d_out;

    // 必须使用固定内存 (Pinned Memory) 才能实现真·异步
    CHECK(cudaMallocHost((void**)&h_in, bytes));
    CHECK(cudaMallocHost((void**)&h_out, bytes));
    CHECK(cudaMalloc((void**)&d_in, bytes));
    CHECK(cudaMalloc((void**)&d_out, bytes));

    for (int i = 0; i < N; i++) h_in[i] = 1.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float ms_serial, ms_pipeline;

    dim3 block(256);
    dim3 grid((chunk_size + block.x - 1) / block.x);

    // ==========================================
    // 阶段一：单流（串行）笨办法
    // ==========================================
    printf("Starting Single Stream (Serial) Execution...\n");
    cudaEventRecord(start);

    // 1. 一次性把 40MB 全搬过去 (H2D)
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);
    // 2. 一次性把 1000 万数据全算完 (Compute)
    heavyCompute<<<(N + block.x - 1)/block.x, block.x>>>(d_in, d_out, N);
    // 3. 一次性全搬回来 (D2H)
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_serial, start, stop);
    printf("Single Stream Time: %f ms\n\n", ms_serial);

    // ==========================================
    // 阶段二：4流并发（Pipelining 流水线）
    // ==========================================
    printf("Starting Multi-Stream Pipelining Execution...\n");
    cudaStream_t streams[num_streams];
    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
    }

    cudaEventRecord(start);

    // 【核心魔法发生在这里】
    // 遍历每一个分块，为其分配独立的三步流水线
    for (int i = 0; i < num_streams; i++) {
        // 计算当前数据块的指针偏移量
        int offset = i * chunk_size; 

        // 1. 搬运第 i 块数据 (H2D)
        cudaMemcpyAsync(&d_in[offset], &h_in[offset], chunk_bytes, cudaMemcpyHostToDevice, streams[i]);
        
        // 2. 计算第 i 块数据 (Compute)
        heavyCompute<<<grid, block, 0, streams[i]>>>(&d_in[offset], &d_out[offset], chunk_size);
        
        // 3. 传回第 i 块结果 (D2H)
        cudaMemcpyAsync(&h_out[offset], &d_out[offset], chunk_bytes, cudaMemcpyDeviceToHost, streams[i]);
    }

    // 等待所有流完成
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms_pipeline, start, stop);
    
    printf("Pipelining Time:    %f ms\n", ms_pipeline);
    printf(">>> Speedup:        %.2f x <<<\n", ms_serial / ms_pipeline); // >>> Speedup:        1.90 x <<<

    // 释放资源
    for (int i = 0; i < num_streams; i++) cudaStreamDestroy(streams[i]);
    cudaFreeHost(h_in); cudaFreeHost(h_out);
    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}