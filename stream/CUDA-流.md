# CUDA 流

## 1. 相关API

1. 创建cuda流：`cudaError_t cudaStreamCreate(cudaStream_t* pStream);`
2. 释放cuda流：`cudaError_t cudaStreamDestroy(cudaStram_t stream);`
3. 检查流操作是否完成：
    1. 阻塞主机：`cudaError_t cudaStreamSynchronize(cudaStream_t stream)`
    2. 非阻塞主机：`cudaError_t cudaStreamQuery(cudaStream_t stream)`


## 2. 例子：多流调度带来收益

见stream_intro.cu文件。测试结果如下表格如下：
| 流数 | 单流运行时间/ms | 多流运行时间/ms | 多流加速比 |
| ---- | ---------- | ---------- | --------- |
| 2    | 84.5087    | 66.0570    | 1.27933   |
| 4    | 85.6501    | 58.0775    | 1.47476   |
| 8    | 84.5134    | 52.7043    | 1.60354   |
| 16   | 86.3229    | 51.5744    | 1.67376   |

## 