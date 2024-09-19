#include <iostream>
#include "KernelForTest.cuh"
#include "error_handling.h"
#include <cuda_runtime.h>
#include <vector>
#include <chrono>
#include <cstdint>

#define NUM_STREAMS 8
#define HOST_MEM_SIZE 1024 * 1024 * 1024 // 1GB

int main() {
    // 创建多个流
    std::vector<cudaStream_t> stream_list(NUM_STREAMS);
    for(int i = 0; i < NUM_STREAMS; i++) {
        RUNTIME_CHECK(cudaStreamCreateWithFlags(&stream_list[i], cudaStreamNonBlocking));
    }

    // 创建一块主机的页锁定内存，并把所有的数据写为 666
    constexpr size_t DATA_SIZE = sizeof(int64_t);
    int64_t* p_host_mem;
    RUNTIME_CHECK(cudaMallocHost(&p_host_mem, HOST_MEM_SIZE));
    for(size_t i = 0; i < HOST_MEM_SIZE / DATA_SIZE; i++) {
        p_host_mem[i] = 666;
    }
    // 测试写入成功与否
    for(size_t i = 0; i < HOST_MEM_SIZE / DATA_SIZE; i++) {
        if(p_host_mem[i] != 666) {
            std::cout << "主机内存写入失败" << std::endl;
            return 0;
        }
    }

    // 创建一块设备的页锁定内存，并把所有的数据写为 666
    int64_t* p_device_mem;
    RUNTIME_CHECK(cudaMalloc(&p_device_mem, HOST_MEM_SIZE));

    // ------------------------------------------------------ 1. 单流执行 ------------------------------------------------------ //
    std::chrono::high_resolution_clock::time_point start_time, end_time;
    start_time = std::chrono::high_resolution_clock::now();
    // 拷贝，执行，拷贝
    RUNTIME_CHECK(cudaMemcpyAsync(p_device_mem, p_host_mem, HOST_MEM_SIZE, 
                                  cudaMemcpyHostToDevice, stream_list[0]));
    int threadsPerBlock = 1024;
    int blocks = (HOST_MEM_SIZE / sizeof(int64_t) + threadsPerBlock - 1) / threadsPerBlock;
    SingleVecKernel_int64_t<<<blocks, threadsPerBlock, 0, stream_list[0]>>>(p_device_mem, HOST_MEM_SIZE / sizeof(int64_t), 777);
    RUNTIME_CHECK(cudaMemcpyAsync(p_host_mem, p_device_mem, HOST_MEM_SIZE,
                                  cudaMemcpyDeviceToHost, stream_list[0]));
    // 同步计时
    RUNTIME_CHECK(cudaStreamSynchronize(stream_list[0]));
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time = end_time - start_time;
    // 核函数报错处理
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    // 验证结果
    bool single_flag = true;
    for(size_t i = 0; i < HOST_MEM_SIZE / DATA_SIZE; i++) {
        if(p_host_mem[i] != 777) 
            single_flag = false;
    }
    if (single_flag)
        std::cout << "单流执行用时: " << elapsed_time.count() << " ms" << std::endl;
    else
        std::cout << "单流执行结果验证失败" << std::endl;
    // ------------------------------------------------------ 1. 单流执行 ------------------------------------------------------ //


    // ------------------------------------------------------ 2. 多流执行 ------------------------------------------------------ //
    start_time = std::chrono::high_resolution_clock::now();
    size_t chunk_size = HOST_MEM_SIZE / NUM_STREAMS; // 每个流处理的数据块大小
    // 拷贝，执行，拷贝
    for(int i = 0; i < NUM_STREAMS; i++) {
        // 计算当前流需要传递的指针
        int64_t* p_host_chunk = p_host_mem + i * chunk_size / DATA_SIZE;  
        int64_t* p_device_chunk = p_device_mem + i * chunk_size / DATA_SIZE;  

        // 异步将数据从主机拷贝到设备
        RUNTIME_CHECK(cudaMemcpyAsync(p_device_chunk, p_host_chunk, chunk_size, 
                                    cudaMemcpyHostToDevice, stream_list[i]));
        
        // 动态计算块数和每块线程数
        int threadsPerBlock = 1024;  // 你可以根据实际情况调整
        int blocks = (chunk_size / sizeof(int64_t) + threadsPerBlock - 1) / threadsPerBlock;

        // 核函数调用，指定每个流的网格和块大小
        SingleVecKernel_int64_t<<<blocks, threadsPerBlock, 0, stream_list[i]>>>(p_device_chunk, chunk_size / DATA_SIZE, 888);

        // 异步将数据从设备拷贝到主机
        RUNTIME_CHECK(cudaMemcpyAsync(p_host_chunk, p_device_chunk, chunk_size,
                                    cudaMemcpyDeviceToHost, stream_list[i]));
    }
    // 同步所有流并计时
    for(int i = 0; i < NUM_STREAMS; i++) {
        RUNTIME_CHECK(cudaStreamSynchronize(stream_list[i]));
    }
    end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed_time2 = end_time - start_time;
    // 核函数报错处理
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    }
    // 验证结果
    bool multi_flag = true;
    for(size_t i = 0; i < HOST_MEM_SIZE / DATA_SIZE; i++) {
        if(p_host_mem[i] != 888) 
            multi_flag = false;
    }
    if (multi_flag)
        std::cout << "多流执行用时: " << elapsed_time2.count() << " ms" << std::endl;
    else{
        std::cout << "多流执行结果验证失败" << std::endl;
        for(size_t i = HOST_MEM_SIZE / DATA_SIZE - 1; i > HOST_MEM_SIZE / DATA_SIZE - 21; i--) {
            std::cout << p_host_mem[i] << " ";
        }
        std::cout << std::endl;
    }
    // ------------------------------------------------------ 2. 多流执行 ------------------------------------------------------ //

    // 释放资源
    RUNTIME_CHECK(cudaFreeHost(p_host_mem));
    RUNTIME_CHECK(cudaFree(p_device_mem));
    for(int i = 0; i < NUM_STREAMS; i++) {
        RUNTIME_CHECK(cudaStreamDestroy(stream_list[i]));
    }

    // 加速比计算
    std::cout << "加速比: " << elapsed_time.count() / elapsed_time2.count() << " 倍" << std::endl;


    return 0;
}
