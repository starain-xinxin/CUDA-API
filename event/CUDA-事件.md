# CUDA 事件

## 1. CUDA 事件介绍

CUDA事件本质上是一个个CUDA流中的标记，与流内的特定点相关联。
可以用于执行两个基本任务：
1. 同步流执行
2. 监控设备的进展

### 1.1 创建 & 销毁
1. 创建事件：`cudaError_t cudaEventCreate(cudaEvent_t *event);`
2. 销毁事件：`cudaError_t cudaEventDestroy(cudaEvent_t event);`

### 1.2 事件Record & 计算运行时间
1. Record一个事件：`cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = 0);`  
   本质上就是在代码的此处插入一个“ 名为CUDA事件 —— cudaEvent_t ”的操作进入流的任务队列中，打上一个标记。
2. 检查事件是否完成：
   实际上，与流同步一样，检查事件是否完成 --> 在任务队列的标记点(event)出同步
    1. 阻塞式：`cudaError_t cudaEventSynchronize(cudaEvent_t event);`  
    2. 非阻塞式：`cudaError_t cudaEventQuery(cudaEvent_t event);`  
3. 计算两个事件之间运行时间：`cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);`  
   本质上，就是对需要计时的任务流两端打上断点/标记，然后计算事件。

## 2. 利用CUDA事件实现流同步
在/stream/CUDA-流同步.md中，我们介绍了阻塞流是需要隐式同步的。这会带来性能损失。例如：
- 主机锁页内存分配
- 设备内存分配
- 内存拷贝
- 内存初始化
- etc.
我们可以使用非阻塞流，利用时间进行同步。
1. 检查事件API进行多流的操作：
    1. 阻塞式：`cudaError_t cudaEventSynchronize(cudaEvent_t event);`
    2. 非阻塞式：`cudaError_t cudaEventQuery(cudaEvent_t event);`
    这可以在单流多断点的粒度上进行同步。（实际上也可以实现一些多流同步了）
2. 流间依赖事件API：`cudaError_t cudaStreamWaitEvent(cudaStream_t streamA, cudaEvent_t event_of_streamB, unsigned int flags);`
   这个API的意思是`streamA`需要阻塞，直到`StreamB`的事件`event_of_streamB`被触发，这时候`streamA`才能继续执行。
  

## 3. CUDA事件的标志位
1. `cudaEventDefault`：默认，忙等待：CPU 会不断轮询事件是否完成
2. `cudaEventBlockingSync`：阻塞等待：CPU 会睡眠，直到同步事件唤醒；但可能带来延迟。(这里的睡眠是指线程的暂停，而不是实际硬件不用了，操作系统可以调度硬件执行其他线程)
3. `cudaEventDisableTiming`：禁用计时标志位，表示事件不会被计时。这可以提高同步性能。
4. `cudaEventInterprocess`：跨进程同步标志位，表示事件可以跨进程同步。


