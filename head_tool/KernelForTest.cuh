#pragma once
#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 16  // CUDA 线程块的大小

// 矩阵乘法的 CUDA 核函数声明
__global__ void matrixMultiply(const float* A, const float* B, float* C, int N);

// 矩阵乘法的接口声明
void matrixMultiplyHost(const float* A, const float* B, float* C, int N);
