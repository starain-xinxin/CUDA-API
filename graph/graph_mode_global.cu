#include <cuda_runtime.h>
#include <iostream>
#include <cassert>  
#include <cstdint>
#include "error_handling.h"
#include "KernelForTest.cuh"

#define MEM_SIZE 1024 * 1024 * 8
#define VEC_SIZE 100

int main(){
    int threadsPerBlock = 1024;
    int Blocks = (MEM_SIZE / sizeof(int64_t) + threadsPerBlock - 1) / threadsPerBlock;

    cudaStream_t stream1;
    RUNTIME_CHECK(cudaStreamCreate(&stream1));

    int64_t* host_ptr;
    int64_t* device_ptr;
    RUNTIME_CHECK(cudaHostAlloc(&host_ptr, MEM_SIZE, cudaHostAllocDefault));
    RUNTIME_CHECK(cudaMalloc(&device_ptr, MEM_SIZE));


    /* ----------------------------- 1. CUDA graph 捕获 --------------------------------- */
    cudaGraph_t graph1;
    RUNTIME_CHECK(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));

    SingleVecKernel_int64_t<<< Blocks, threadsPerBlock, 0, stream1 >>>(device_ptr, VEC_SIZE, 100);
    SingleVecKernel_int64_t<<< Blocks, threadsPerBlock, 0, stream1 >>>(device_ptr + VEC_SIZE, VEC_SIZE, 200);
    SingleVecAddKernel_int64_t<<< Blocks, threadsPerBlock, 0, stream1 >>>(device_ptr, 2 * VEC_SIZE , 100);
   
    RUNTIME_CHECK(cudaStreamEndCapture(stream1, &graph1));

    /* ----------------------------- 1. graph 结束捕获 --------------------------------- */

    /* ----------------------------- 2. CUDA graph 实例化并执行 ------------------------- */
    cudaGraphExec_t graphExec1;
    RUNTIME_CHECK(cudaGraphInstantiate(&graphExec1, graph1, NULL, NULL, 0));
    RUNTIME_CHECK(cudaGraphLaunch(graphExec1, stream1));
    RUNTIME_CHECK(cudaMemcpy(host_ptr, device_ptr, MEM_SIZE, cudaMemcpyDeviceToHost));
    RUNTIME_CHECK(cudaStreamSynchronize(stream1));
    
    /* ------------------------------------ 3. 验证结果 -------------------------------- */
    for(int i = 0; i < VEC_SIZE; i++){
        if (host_ptr[i] != 200){
            std::cout << host_ptr[i] << " ";
            exit(-1);
        }
    }
    std::cout << std::endl;
    for(int i = VEC_SIZE; i < 2 * VEC_SIZE; i++){
        if (host_ptr[i] != 300){
            std::cout << host_ptr[i] << " ";
            exit(-1);
        }
    }

    /* ----------------------------- 4. 释放资源 ----------------------------------- */
    RUNTIME_CHECK(cudaFreeHost(host_ptr));
    RUNTIME_CHECK(cudaFree(device_ptr));
    RUNTIME_CHECK(cudaStreamDestroy(stream1));
    RUNTIME_CHECK(cudaGraphExecDestroy(graphExec1));
    RUNTIME_CHECK(cudaGraphDestroy(graph1));

    return 0;

}