#include <iostream>
#include "KernelForTest.cuh"
#include "error_handling.h"
#include <cuda_runtime.h>


int main() {
    cudaStream_t stream_id = 0;
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


    RUNTIME_CHECK(cudaEventRecord(start, stream_id));
    // ...
    // 启动 CUDA 核函数
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    // ...

    RUNTIME_CHECK(cudaEventRecord(stop, stream_id));
    RUNTIME_CHECK(cudaEventSynchronize(stop));
    float milliseconds = 0;
    RUNTIME_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    std::cout << "Kernel time: " << milliseconds << " ms" << std::endl;



    cudaDeviceSynchronize();
    cudaMemcpy(C, d_C, N * N * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    


    return 0;
}
