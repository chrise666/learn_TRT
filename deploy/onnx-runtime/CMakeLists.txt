# 设置 ONNX Runtime 路径
set(ONNXRUNTIME_ROOT 
    "E:/workspace/Lib/onnxruntime-win-x64-gpu-1.18.1" 
    CACHE PATH "ONNX Runtime 路径"
)

# 检查路径有效性
if(NOT EXISTS ${ONNXRUNTIME_ROOT})
    message(FATAL_ERROR "ONNX Runtime 路径无效: ${ONNXRUNTIME_ROOT}")
endif()

# 添加库目标
aux_source_directory(src/. SRC_ONNX) # 添加同级目录下的所有源文件
add_library(onnx_inference STATIC ${SRC_ONNX})

# 包含目录
target_include_directories(onnx_inference 
    PUBLIC 
        include
        ${ONNXRUNTIME_ROOT}/include
        ${OpenCV_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(onnx_inference
    PUBLIC
        ${OpenCV_LIBRARIES}
        ${CUDA_LIBRARIES}
        onnxruntime
        onnxruntime_providers_cuda
        cudart
        # cudnn
)

# 链接目录
target_link_directories(onnx_inference
    PUBLIC
        ${ONNXRUNTIME_ROOT}/lib
        ${CUDA_LIBRARY_DIRS}
)
