#include "kernel.cuh"


/*******************************  CUDA 核函数  *********************************/
// 向量加法核函数：逐元素加法
template <typename T>
__global__ void vectorAdd_Kernel(T *d_a, T *d_b, T *d_result, int n)
{
    // blockIdx代表block的索引,blockDim代表block的大小，threadIdx代表thread线程的索引，因此对于一维的block和thread索引的计算方式如下
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        d_result[i] = d_a[i] + d_b[i];  // 逐元素相加
    }
}

// 向量乘法核函数：逐元素乘法
template <typename T>
__global__ void vectorMultiply_Kernel(T *d_a, T *d_b, T *d_result, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        d_result[i] = d_a[i] * d_b[i];  // 逐元素相乘
    }
}

// 向量除法核函数：逐元素除法
template <typename T>
__global__ void vectorDivision_Kernel(T *d_a, T *d_b, T *d_result, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        d_result[i] = d_a[i] / d_b[i];  // 逐元素相除
    }
}

// 向量求和核函数：归约求和
template <typename T>
__global__ void vectorSum_Kernel(T *d_vector, T *d_result, int n)
{
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // 将数据加载到共享内存
    sdata[tid] = (idx < n) ? d_vector[idx] : 0.0f;
    __syncthreads();
    
    // 归约求和
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // 将块结果写入全局内存
    if (tid == 0) {
        d_result[blockIdx.x] = sdata[0];
    }
}

// 矩阵加法核函数：
template <typename T>
__global__ void matrixAdd_Kernel(T *d_a, T *d_b, T *d_result, int nx, int ny)
{
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * ny; // 将2D索引转换为1D索引

    if (ix < nx && iy < ny)
    {
        d_result[idx] = d_a[idx] + d_b[idx];
    }
}


/*******************************  核函数包装函数  *********************************/
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
    // addKernel<<<gridSize, blockSize, 0, stream1>>>(d_a, d_b, d_c, n);

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

// 核函数用模板不会报错，模板名字是具有链接的，但它们不能具有C链接，因此不能用在供调用的函数上
float *matAdd(float *a, float *b, int length)
{
    int device = 0;        // 设置使用第0块GPU进行运算
    cudaSetDevice(device); // 设置运算显卡

    cudaDeviceProp devProp;
    cudaGetDeviceProperties(&devProp, device);                    // 获取对应设备属性
    int threadMaxSize = devProp.maxThreadsPerBlock;               // 每个线程块的最大线程数
    int blockSize = (length + threadMaxSize - 1) / threadMaxSize; // 计算Block大小,block一维度是最大的，一般不会溢出

    dim3 thread(threadMaxSize); // 设置thread
    dim3 block(blockSize);      // 设置block

    int size = length * sizeof(float);  // 计算空间大小
    float *sum = (float *)malloc(size); // 开辟动态内存空间

    // 开辟显存空间
    float *sumGPU, *aGPU, *bGPU;
    cudaMalloc((void **)&sumGPU, size);
    cudaMalloc((void **)&aGPU, size);
    cudaMalloc((void **)&bGPU, size);

    // 内存->显存
    cudaMemcpy((void *)aGPU, (void *)a, size, cudaMemcpyHostToDevice);
    cudaMemcpy((void *)bGPU, (void *)b, size, cudaMemcpyHostToDevice);

    // 启动核函数，进行加法运算
    // vectorAdd<float><<<block, thread>>>(aGPU, bGPU, sumGPU, length);

    // 等待GPU完成
    cudaDeviceSynchronize();

    // 显存->内存
    cudaMemcpy(sum, sumGPU, size, cudaMemcpyDeviceToHost);

    // 释放显存
    cudaFree(sumGPU);
    cudaFree(aGPU);
    cudaFree(bGPU);

    return sum;
}
