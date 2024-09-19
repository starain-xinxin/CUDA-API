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

### 3.3 流的分类
``` txt
|-- 同步流(默认流/空流)      // 因为默认流的流句柄是空指针，所以称为空流 
|-- 异步流(非空流)
    ｜
    ｜-- 阻塞流 cudaStreamDefault：阻塞流与默认流以及其他阻塞流之间会自动进行隐式同步。这意味着，阻塞流中的操作会等待默认流中的操作完成后再执行，默认流中的操作也会等待阻塞流中的操作完成。
    ｜
    ｜-- 非阻塞流 cudaStreamNonBlocking：非阻塞流不会与默认流或其他流进行隐式同步。它与其他流的操作是独立的，能够并行执行，不等待默认流或其他阻塞流中的任务。
```
什么是自动隐式同步？
例如下面的代码：
```cuda
kernel1<<<1,1,0,stream1>>>();
kernel2<<<1,1>>>();
kernel3<<<1,1,0,stream2>>>();
```
如果stream1和stream2都是阻塞流，则这3个kernel之间会自动进行隐式同步，即kernel2会等待kernel1执行完成后再执行，kernel3会等待kernel2执行完成后再执行。
即，实际上并没有实现物理意义上的并发。
但如果是非阻塞流，就是互不影响，几乎真正的并发。（当然是受硬件决定的并发）。
实际上阻塞流和非阻塞流的区别在于是否受到自动隐式同步的影响；并且二者均可以被显式同步(利用 event 或者强制同步的 API)。

### 3.4 显示同步CUDA流
1. 阻塞主机线程直到设备完成所有的任务：`cudaError_t cudaDeviceSynchronize();`
   这个同步粒度太粗。
2. 检查流API进行单流的操作：
    1. 阻塞主机：`cudaError_t cudaStreamSynchronize(cudaStream_t stream)`
    2. 非阻塞主机：`cudaError_t cudaStreamQuery(cudaStream_t stream)`
    这可以在单流的粒度上进行同步。
3. 检查事件API进行多流的操作：
    1. 阻塞式：`cudaError_t cudaEventSynchronize(cudaEvent_t event);`
    2. 非阻塞式：`cudaError_t cudaEventQuery(cudaEvent_t event);`
    这可以在单流多断点的粒度上进行同步。（实际上也可以实现一些多流同步了）
4. 流间依赖事件API：`cudaError_t cudaStreamWaitEvent(cudaStream_t streamA, cudaEvent_t event_of_streamB, unsigned int flags);`
   这个API的意思是`streamA`需要阻塞，直到`StreamB`的事件`event_of_streamB`被触发，这时候`streamA`才能继续执行。





