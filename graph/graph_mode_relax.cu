#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK_ERROR(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    } \
}

__global__ void kernel(float* d_data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 1024) {
        d_data[idx] += 1.0f;
    }
}

void runCudaGraphRelaxed() {
    float* d_data;
    size_t size = 1024 * sizeof(float);

    // 创建 CUDA 流并设置非阻塞标志
    // ！！！！！！
    // 这里不能设置为阻塞模式，否则这将导致捕获出错
    // ！！！！！！
    cudaStream_t stream;
    CUDA_CHECK_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    // 开始捕获流
    CUDA_CHECK_ERROR(cudaStreamBeginCapture(stream, cudaStreamCaptureModeRelaxed));

    // 在捕获图中分配内存
    CUDA_CHECK_ERROR(cudaMalloc(&d_data, size));

    // 初始化数据
    CUDA_CHECK_ERROR(cudaMemset(d_data, 0, size));

    // 捕获内核调用
    kernel<<<4, 256, 0, stream>>>(d_data);

    // 捕获结束
    cudaGraph_t graph;
    CUDA_CHECK_ERROR(cudaStreamEndCapture(stream, &graph));

    // 实例化图
    cudaGraphExec_t graphExec;
    CUDA_CHECK_ERROR(cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0));

    // 重放图
    CUDA_CHECK_ERROR(cudaGraphLaunch(graphExec, stream));
    CUDA_CHECK_ERROR(cudaStreamSynchronize(stream));

    // 清理资源
    CUDA_CHECK_ERROR(cudaFree(d_data));
    CUDA_CHECK_ERROR(cudaGraphExecDestroy(graphExec));
    CUDA_CHECK_ERROR(cudaGraphDestroy(graph));
    CUDA_CHECK_ERROR(cudaStreamDestroy(stream));
}


int main() {
    std::cout << "Running with Relaxed mode..." << std::endl;
    runCudaGraphRelaxed();
    std::cout << "CUDA graph with Relaxed mode executed successfully." << std::endl;
    return 0;
}
