#include <iostream>
#include <cuda_runtime.h>
#include "error_handling.h"
#include "KernelForTest.cuh"
#include <cstdint>

// 对于某个流，异步上传计算任务
void upload_task(cudaStream_t stream, int64_t* p_device_memory, size_t num_data){
    // 计算需要的线程数
    int num_threads = 128;
    int num_blocks = (num_data + num_threads - 1) / num_threads;
    RUNTIME_CHECK(cudaMalloc(&p_device_memory, num_data * sizeof(int64_t)));
    SingleVecKernel_int64_t <<<num_blocks, num_threads, 0, stream>>> (p_device_memory, num_data, 666);
}

int main() {

    // 获取优先级范围
    int leastPriority = 0;
    int greatestPriority = 0;
    RUNTIME_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    std::cout << "Least priority: " << leastPriority << std::endl;
    std::cout << "Greatest priority: " << greatestPriority << std::endl;

    // 创建四个流,两个最高，两个最低
    cudaStream_t streams[4];
    RUNTIME_CHECK(cudaStreamCreateWithPriority(&streams[0], cudaStreamNonBlocking, greatestPriority));
    RUNTIME_CHECK(cudaStreamCreateWithPriority(&streams[3], cudaStreamNonBlocking, greatestPriority));
    RUNTIME_CHECK(cudaStreamCreateWithPriority(&streams[1], cudaStreamNonBlocking, leastPriority));
    RUNTIME_CHECK(cudaStreamCreateWithPriority(&streams[2], cudaStreamNonBlocking, leastPriority));
    std::cout << "优先级排序：0 = 3 > 1 = 2" << std::endl; 

    // 准备指针与任务配置
    int64_t* p_list[4];
    size_t num_data = 128 * 1024 * 1024;

    // 上传任务
    for (int i = 0; i < 4; i++) {
        upload_task(streams[i], p_list[i], num_data);
    }

    // 非阻塞的询问流是否完成
    bool is_done[4] = {false, false, false, false};
    int priority_sort[4];
    int no_stream = 0;
    cudaError_t error[4];
    while(no_stream < 4){    // 轮询流是否完成
        for (int i = 0; i < 4; i++) {
            if (!is_done[i]){
                error[i] = cudaStreamQuery(streams[i]);
                if (error[i] == cudaSuccess) {
                    std::cout << "Stream " << i << " is done." << std::endl;
                    is_done[i] = true;
                    priority_sort[no_stream] = i;
                    no_stream++;
                }
            }
        }
    }

    // 打印实测优先级排序
    std::cout << "实测优先级排序：" << priority_sort[0] << " " << priority_sort[1] << " " << priority_sort[2] << " " << priority_sort[3] << " "<< std::endl;

    // 释放资源
    for (int i = 0; i < 4; i++) {
        RUNTIME_CHECK(cudaStreamDestroy(streams[i]));
        RUNTIME_CHECK(cudaFree(p_list[i]));
    }
    
    return 0;
}