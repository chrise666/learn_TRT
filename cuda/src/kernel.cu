#include "kernel.cuh"

/***  CUDA 核函数  ***/
// 定义设备函数：处理单个元素
__device__ void kernel(int *data, int idx)
{
    data[idx] *= 2;
}

// 核函数：计算a+b，并通过kernel处理结果
__global__ void addKernel(int *a, int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];  // 计算加法
        kernel(c, i);        // 调用设备函数处理c[i]
    }
}

// 核函数包装函数
void addKernelWrapper(int *a, int *b, int *c, int n)
{
    int *d_a, *d_b, *d_c;
    const int size = n * sizeof(int);

    // 设备端分配内存
    cudaErrorCheck(cudaMalloc(&d_a, size));
    cudaErrorCheck(cudaMalloc(&d_b, size));
    cudaErrorCheck(cudaMalloc(&d_c, size));

    // 创建两个流
    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t event;
    cudaEventCreate(&event);

    /***  流1：传输a和b到GPU，执行加法，传回结果  ***/
    // 将数据从主机复制到设备
    cudaErrorCheck(cudaMemcpyAsync(d_a, a, size, cudaMemcpyHostToDevice, stream1));
    cudaErrorCheck(cudaMemcpyAsync(d_b, b, size, cudaMemcpyHostToDevice, stream1));

    // 启动核函数
    int blockSize = 256;
    int gridSize = (n + blockSize - 1) / blockSize;
    addKernel<<<gridSize, blockSize, 0, stream1>>>(d_a, d_b, d_c, n);

    // 检查核函数是否成功启动
    cudaErrorCheck(cudaGetLastError());

    // 将结果从设备复制回主机
    cudaErrorCheck(cudaMemcpyAsync(c, d_c, size, cudaMemcpyDeviceToHost, stream1));

    // 流1：执行操作后记录事件
    cudaEventRecord(event, stream1);

    /***  流2：类似操作（假设有其他任务）  ***/
    // 流2：等待事件完成后再执行
    cudaStreamWaitEvent(stream2, event, 0);
    // addKernel<<<gridSize, blockSize, 0, stream2>>>(d_a, d_b, d_c, n);

    // 同步所有流
    cudaErrorCheck(cudaStreamSynchronize(stream1));
    cudaErrorCheck(cudaStreamSynchronize(stream2));
    // cudaErrorCheck(cudaDeviceSynchronize());

    // 释放资源
    cudaErrorCheck(cudaFree(d_a));
    cudaErrorCheck(cudaFree(d_b));
    cudaErrorCheck(cudaFree(d_c));
    cudaErrorCheck(cudaStreamDestroy(stream1));
    cudaErrorCheck(cudaStreamDestroy(stream2));
}
