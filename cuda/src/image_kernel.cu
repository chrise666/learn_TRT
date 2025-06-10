#include "image_kernel.cuh"

using namespace cv;

// CUDA设备检查
bool checkCudaDevice()
{
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess) && (device_count > 0);
}

// CUDA核函数实现图像裁剪
__global__ void gpu_crop_kernel(uchar3 *src, int src_width,
                                uchar3 *dst, int dst_width, int dst_height,
                                int roi_x, int roi_y)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < dst_width && y < dst_height)
    {
        int src_x = roi_x + x;
        int src_y = roi_y + y;

        dst[y * dst_width + x] = src[src_y * src_width + src_x];
    }
}

// CUDA加速裁剪实现
Mat cudaCrop(const Mat &src, const Rect &roi)
{
    // 准备输入输出
    Mat dst(roi.height, roi.width, CV_8UC3);

    // 分配设备内存
    uchar3 *d_src = nullptr, *d_dst = nullptr;
    cudaMalloc(&d_src, src.total() * src.elemSize());
    cudaMalloc(&d_dst, roi.area() * 3);

    // 上传数据到设备
    cudaMemcpy(d_src, src.data,
               src.total() * src.elemSize(),
               cudaMemcpyHostToDevice);

    // 启动核函数
    dim3 block(16, 16);
    dim3 grid((roi.width + block.x - 1) / block.x,
              (roi.height + block.y - 1) / block.y);

    gpu_crop_kernel<<<grid, block>>>(d_src, src.cols, d_dst,
                                     roi.width, roi.height,
                                     roi.x, roi.y);

    // 下载数据到主机
    cudaMemcpy(dst.data, d_dst,
               roi.area() * 3,
               cudaMemcpyDeviceToHost);

    // 释放设备内存
    cudaFree(d_src);
    cudaFree(d_dst);

    return dst;
}
