#include <iostream>
#include "KernelForTest.cuh"
#include "error_handling.h"
#include <cuda_runtime.h>

#define NUM_CALL_KERNEL 100000

int main() {
    cudaStream_t stream_id;
    cudaStreamCreate(&stream_id);
        
    cudaEvent_t start, stop;
    RUNTIME_CHECK(cudaEventCreate(&start));
    RUNTIME_CHECK(cudaEventCreate(&stop));
    

    int N = 4;  
    float A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};  
    float B[] = {16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1}; 
    float C[16]; 
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * N * sizeof(float));
    cudaMalloc(&d_B, N * N * sizeof(float));
    cudaMalloc(&d_C, N * N * sizeof(float));
    cudaMemcpy(d_A, A, N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);


    RUNTIME_CHECK(cudaEventRecord(start, stream_id));       // 利用start事件打上标记
    // ...
    // 启动 CUDA 核函数
    for (int i = 0; i < NUM_CALL_KERNEL; i++) {    
        matrixMultiply<<<blocksPerGrid, threadsPerBlock, 0, stream_id>>>(d_A, d_B, d_C, N);
    }
    // ...

    RUNTIME_CHECK(cudaEventRecord(stop, stream_id));       // 利用stop事件打上标记
    RUNTIME_CHECK(cudaEventSynchronize(stop));             // 阻塞式同步stop事件完成

    // 计算Kernel运行时间
    float milliseconds = 0;
    RUNTIME_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << NUM_CALL_KERNEL <<"次 kernel 用时: " << milliseconds << " ms" << std::endl;
    
    // 传输结果，释放资源等
    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}
