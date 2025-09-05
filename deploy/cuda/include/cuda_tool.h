#ifndef CUDA_TOOL_H
#define CUDA_TOOL_H

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// CUDA 错误检查宏
#define cudaErrorCheck(ans) { gpuAssert((ans), __FILE__, __LINE__, true); }
// CUDA 错误处理函数
inline void gpuAssert(cudaError_t code, const char *file, int line, int abort) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s in %s line %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code); // 如果 abort 为 true，则终止程序
        }
    }
}

// GPU 信息结构体
typedef struct {
    char name[256];                     // 设备名称
    int computeCapabilityMajor;         // 计算能力主版本号
    int computeCapabilityMinor;         // 计算能力次版本号
    size_t totalGlobalMem;              // 全局内存大小（字节）
    size_t sharedMemPerBlock;           // 每个块的共享内存大小（字节）
    int regsPerBlock;                   // 每个块的寄存器数量
    int warpSize;                       // Warp 大小
    int maxThreadsPerBlock;             // 每个块的最大线程数
    int maxThreadsPerMultiProcessor;    // 每个多处理器的最大线程数
    int multiProcessorCount;            // 多处理器数量
    int maxGridSize[3];                 // 最大网格大小
    int maxThreadsDim[3];               // 最大线程块维度
    int clockRate;                      // GPU 时钟频率（kHz）
    int memoryClockRate;                // 显存时钟频率（kHz）
    int memoryBusWidth;                 // 显存总线宽度（位）
    int l2CacheSize;                    // L2 缓存大小（字节）
    int eccEnabled;                     // ECC 是否启用
    int unifiedAddressing;              // 是否支持统一寻址（UVA）
    int concurrentKernels;              // 是否支持并发内核
    int deviceOverlap;                  // 是否支持设备重叠
    int integrated;                     // 是否是集成 GPU
    int canMapHostMemory;               // 是否可以映射主机内存
    int computeMode;                    // 计算模式
} GPUInfo;

// 获取 GPU 设备信息
inline GPUInfo getGPUInfo(int deviceId) {
    GPUInfo info = {0};  // 初始化结构体
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, deviceId);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to get device properties for device %d: %s\n", deviceId, cudaGetErrorString(err));
        return info; // 返回空结构体
    }

    // 填充 GPUInfo 结构体
    snprintf(info.name, sizeof(info.name), "%s", prop.name);
    info.computeCapabilityMajor = prop.major;
    info.computeCapabilityMinor = prop.minor;
    info.totalGlobalMem = prop.totalGlobalMem;
    info.sharedMemPerBlock = prop.sharedMemPerBlock;
    info.regsPerBlock = prop.regsPerBlock;
    info.warpSize = prop.warpSize;
    info.maxThreadsPerBlock = prop.maxThreadsPerBlock;
    info.maxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    info.multiProcessorCount = prop.multiProcessorCount;
    info.maxGridSize[0] = prop.maxGridSize[0];
    info.maxGridSize[1] = prop.maxGridSize[1];
    info.maxGridSize[2] = prop.maxGridSize[2];
    info.maxThreadsDim[0] = prop.maxThreadsDim[0];
    info.maxThreadsDim[1] = prop.maxThreadsDim[1];
    info.maxThreadsDim[2] = prop.maxThreadsDim[2];
    info.clockRate = prop.clockRate;
    info.memoryClockRate = prop.memoryClockRate;
    info.memoryBusWidth = prop.memoryBusWidth;
    info.l2CacheSize = prop.l2CacheSize;
    info.eccEnabled = prop.ECCEnabled;
    info.unifiedAddressing = prop.unifiedAddressing;
    info.concurrentKernels = prop.concurrentKernels;
    info.deviceOverlap = prop.deviceOverlap;
    info.integrated = prop.integrated;
    info.canMapHostMemory = prop.canMapHostMemory;
    info.computeMode = prop.computeMode;

    return info;
}

// 打印 GPU 信息
inline void printGPUInfo(const GPUInfo *info) {
    printf("Device Name: %s\n", info->name);
    printf("  Compute Capability: %d.%d\n", info->computeCapabilityMajor, info->computeCapabilityMinor);
    printf("  Global Memory: %.2f GB\n", (double)info->totalGlobalMem / (1024 * 1024 * 1024));
    printf("  Shared Memory per Block: %.2f KB\n", (double)info->sharedMemPerBlock / 1024);
    printf("  Registers per Block: %d\n", info->regsPerBlock);
    printf("  Warp Size: %d\n", info->warpSize);
    printf("  Max Threads per Block: %d\n", info->maxThreadsPerBlock);
    printf("  Max Threads per Multiprocessor: %d\n", info->maxThreadsPerMultiProcessor);
    printf("  Multiprocessor Count: %d\n", info->multiProcessorCount);
    printf("  Max Grid Size: [%d, %d, %d]\n", info->maxGridSize[0], info->maxGridSize[1], info->maxGridSize[2]);
    printf("  Max Block Dimensions: [%d, %d, %d]\n", info->maxThreadsDim[0], info->maxThreadsDim[1], info->maxThreadsDim[2]);
    printf("  Clock Rate: %.2f GHz\n", (double)info->clockRate / 1000000);
    printf("  Memory Clock Rate: %.2f GHz\n", (double)info->memoryClockRate / 1000000);
    printf("  Memory Bus Width: %d bits\n", info->memoryBusWidth);
    printf("  L2 Cache Size: %.2f MB\n", (double)info->l2CacheSize / (1024 * 1024));
    printf("  ECC Enabled: %s\n", info->eccEnabled ? "Yes" : "No");
    printf("  Unified Addressing (UVA): %s\n", info->unifiedAddressing ? "Yes" : "No");
    printf("  Concurrent Kernels: %s\n", info->concurrentKernels ? "Yes" : "No");
    printf("  Device Overlap: %s\n", info->deviceOverlap ? "Yes" : "No");
    printf("  Integrated GPU: %s\n", info->integrated ? "Yes" : "No");
    printf("  Can Map Host Memory: %s\n", info->canMapHostMemory ? "Yes" : "No");
    printf("  Compute Mode: ");
    switch (info->computeMode) {
        case cudaComputeModeDefault:
            printf("Default\n");
            break;
        case cudaComputeModeExclusive:
            printf("Exclusive\n");
            break;
        case cudaComputeModeProhibited:
            printf("Prohibited\n");
            break;
        case cudaComputeModeExclusiveProcess:
            printf("Exclusive Process\n");
            break;
        default:
            printf("Unknown\n");
    }
    printf("\n");
}

// CUDA设备检查
inline bool checkCudaDevice()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess) && (device_count > 0);
}

#endif // CUDA_TOOL_H