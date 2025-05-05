#ifndef KERNEL_CUH
#define KERNEL_CUH

#include "cuda_tool.h"

#ifdef __cplusplus
extern "C" {
#endif

// 核函数声明
void addKernelWrapper(int *a, int *b, int *c, int n);

#ifdef __cplusplus
}
#endif

#endif // KERNEL_CUH
