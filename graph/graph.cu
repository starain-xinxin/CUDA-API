#include <cuda_runtime.h>
#include <iostream>
#include <cassert>
#include <cstdint>
#include "error_handling.h"
#include "KernelForTest.cuh"

#define HOST_MEM_SIZE 1024 * 1024 * 8
#define VEC_SIZE 100

int main(){
    /* 0. 线程数，显存等预处理 */
    int threadsPerBlock = 1024;
    int blocks = (HOST_MEM_SIZE / sizeof(int64_t) + threadsPerBlock - 1) / threadsPerBlock;

    int64_t* host_ptr;
    RUNTIME_CHECK(cudaHostAlloc(&host_ptr, HOST_MEM_SIZE, cudaHostAllocDefault));
    int64_t* device_ptr;
    RUNTIME_CHECK(cudaMalloc(&device_ptr, HOST_MEM_SIZE));  

    /* ------------------------------------ 1. 创建图 ------------------------------------ */
    cudaGraph_t graph1;
    // cudaGraphNode_t MallocNode, Kernel_B, C, D, E;
    RUNTIME_CHECK(cudaGraphCreate(&graph1, 0));

    cudaGraphNode_t Kernel_A;
    cudaKernelNodeParams Param_A;
    Param_A.blockDim = dim3(threadsPerBlock);
    Param_A.gridDim = dim3(blocks);
    Param_A.func = (void*)SingleVecKernel_int64_t;
    Param_A.sharedMemBytes = 0; 
    Param_A.extra = NULL; 
    int64_t* A_p_vec = device_ptr; int A_N = VEC_SIZE; int64_t A_number = 100;
    void* KernelParams_A[3] = {&A_p_vec, &A_N, &A_number};
    Param_A.kernelParams = KernelParams_A;
    RUNTIME_CHECK(cudaGraphAddKernelNode(&Kernel_A, graph1, NULL, 0, &Param_A));

    cudaGraphNode_t Kernel_B;
    cudaKernelNodeParams Param_B;
    Param_B.blockDim = dim3(threadsPerBlock);
    Param_B.gridDim = dim3(blocks);
    Param_B.func = (void*)SingleVecKernel_int64_t;
    Param_B.sharedMemBytes = 0; 
    Param_B.extra = NULL; 
    int64_t* B_p_vec = device_ptr + VEC_SIZE; int B_N = VEC_SIZE; int64_t B_number = 200;
    void* KernelParams_B[3] = {&B_p_vec, &B_N, &B_number};
    Param_B.kernelParams = KernelParams_B;
    RUNTIME_CHECK(cudaGraphAddKernelNode(&Kernel_B, graph1, NULL, 0, &Param_B));

    cudaGraphNode_t Kernel_C;
    cudaKernelNodeParams Param_C;
    Param_C.blockDim = dim3(threadsPerBlock);
    Param_C.gridDim = dim3(blocks);
    Param_C.func = (void*)SingleVecAddKernel_int64_t;
    Param_C.sharedMemBytes = 0; 
    Param_C.extra = NULL; 
    int64_t* C_p_vec = device_ptr; int C_N = 2 * VEC_SIZE; int64_t C_number = 100;
    void* KernelParams_C[3] = {&C_p_vec, &C_N, &C_number};
    Param_C.kernelParams = KernelParams_C;
    RUNTIME_CHECK(cudaGraphAddKernelNode(&Kernel_C, graph1, NULL, 0, &Param_C));

    cudaGraphNode_t MemCpy_D;
    RUNTIME_CHECK(cudaGraphAddMemcpyNode1D(&MemCpy_D, graph1, NULL, 0, host_ptr, device_ptr, HOST_MEM_SIZE, cudaMemcpyDeviceToHost));

    // 添加依赖
    RUNTIME_CHECK(cudaGraphAddDependencies(graph1, &Kernel_A, &Kernel_C, 1)); // A -> C
    RUNTIME_CHECK(cudaGraphAddDependencies(graph1, &Kernel_B, &Kernel_C, 1)); // B -> C
    RUNTIME_CHECK(cudaGraphAddDependencies(graph1, &Kernel_C, &MemCpy_D, 1)); // c -> D



     /* ------------------------------------ 2. 实例化图 ------------------------------------ */
    cudaGraphExec_t graphExec1;
    RUNTIME_CHECK(cudaGraphInstantiate(&graphExec1, graph1, NULL, NULL, 0));
    cudaStream_t stream1;
    RUNTIME_CHECK(cudaStreamCreate(&stream1));

    /* ------------------------------------ 3. 执行图 ------------------------------------- */
    cudaGraphLaunch(graphExec1, stream1);
    cudaStreamSynchronize(stream1);

    /* ------------------------------------ 4. 验证结果 ---------- */
    for(int i = 0; i < VEC_SIZE; i++){
        assert(host_ptr[i] == 200 && "Error: The result is not 200");
    }
    for(int i = VEC_SIZE; i < 2 * VEC_SIZE; i++){
        assert(host_ptr[i] == 300 && "Error: The result is not 300");
    }

    /* ------------------------------------ 5. 释放资源 ------------------------------------ */
    RUNTIME_CHECK(cudaGraphExecDestroy(graphExec1));
    RUNTIME_CHECK(cudaStreamDestroy(stream1));
    RUNTIME_CHECK(cudaGraphDestroy(graph1));
    RUNTIME_CHECK(cudaFree(device_ptr));
    RUNTIME_CHECK(cudaFreeHost(host_ptr));


    return 0;
}
