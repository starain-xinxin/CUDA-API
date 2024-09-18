#pragma once
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda.h>

// 运行时库的错误处理
#define RUNTIME_CHECK(call)                           \
do {                                                  \
    cudaError_t error_code = call;                           \
    if (error_code != cudaSuccess) {                         \
        printf("CUDA runtime api error:\n");          \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while(0)  

// 低级驱动库的错误处理
#define DRIVE_CHECK(call)                                           \
do {                                                                \
    CUresult result = call;                                         \
    if (result != CUDA_SUCCESS) {                                   \
        const char *errMsg; cuGetErrorString(result, &errMsg);      \
        printf("CUDA drive api error:\n");                          \
        printf("    File:       %s\n", __FILE__);                   \
        printf("    Line:       %d\n", __LINE__);                   \
        printf("    Error code: %d\n", result);                     \
        printf("    Error text: %s\n", errMsg);                     \
        exit(1);                                                    \
    }                                                               \
} while(0)  
