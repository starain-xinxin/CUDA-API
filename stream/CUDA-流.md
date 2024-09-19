# CUDA 流

## 1. 相关API

1. 创建cuda流：`cudaError_t cudaStreamCreate(cudaStream_t* pStream);`
2. 释放cuda流：`cudaError_t cudaStreamDestroy(cudaStram_t stream);`
3. 检查流操作是否完成：
    1. 阻塞主机：`cudaError_t cudaStreamSynchronize(cudaStream_t stream)`
    2. 非阻塞主机：`cudaError_t cudaStreamQuery(cudaStream_t stream)`
4. 流优先级api：
    1. 创建一个特定优先级的流：`cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int priority);`
    2. 查询优先级衡量范围：`cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);`


## 2. 例子：多流调度带来收益

见stream_intro.cu文件。测试结果如下表格如下：
| 流数 | 单流运行时间/ms | 多流运行时间/ms | 多流加速比 |
| ---- | ---------- | ---------- | --------- |
| 2    | 84.5087    | 66.0570    | 1.27933   |
| 4    | 85.6501    | 58.0775    | 1.47476   |
| 8    | 84.5134    | 52.7043    | 1.60354   |
| 16   | 86.3229    | 51.5744    | 1.67376   |

## 3. CUDA流的基本概念

### 3.1 CUDA流调度 

不同的流的任务并不一定是真正并发的，这根据硬件架构和特性决定。

### 3.2 流优先级

流的优先级不会影响传输操作的顺序，只对计算内核有关。


