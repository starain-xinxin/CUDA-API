# CUDA-Graph

## 0. CUDA-graph API
1. 

## 1. CUDA 图工作流程

1. 定义cuda-graph
2. 实例化可执行图
3. 将实例化的图执行到CUDA流中

## 2. 创建CUDA-graph

### 2.1 显示创建CUDA-graph

### 2.2 捕获方式创建CUDA-graph

捕获的一般性流程：
``` c++
cudaGraph_t graph;
cudaStreamBeginCapture(stream, mode); // 开始捕获
// ...
kernel<<<grid, block>>>(...);
Memcpy...
kernel<<<grid, block>>>(...);
// ...
cudaStreamEndCapture(stream, &graph); // 结束捕获
```

捕获的方式有三种：
1. 全局模式：cudaSreamCaptureModeGlobal
    - 支持多线程的捕获
    - 捕获的操作必须与当前捕获线程的上下文一致
2. 单线程模式：cudaStreamCaptureModeThreadLocal
    - 只支持单线程的捕获，用于防止其他线程干扰
3. 宽松模式：cudaStreamCaptureModeRelaxed
    - 支持多线程，多上下文的捕获


