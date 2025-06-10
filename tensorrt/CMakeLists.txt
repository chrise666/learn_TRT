# 添加项目的依赖库
set(TENSORRT_DIR "E:/workspace/TensorRT-10.9.0.34") # 设置 TensorRT 路径

# 编译列表
aux_source_directory(src/. SRC_TRT) # 添加同级目录下的所有源文件
add_library(trt_lib STATIC ${SRC_TRT})

# 包含头文件目录
target_include_directories(trt_lib PUBLIC
    include
    ${TENSORRT_DIR}/include
)

# 链接库路径
target_link_directories(trt_lib PUBLIC
    ${TENSORRT_DIR}/lib
)

# 链接库
target_link_libraries(trt_lib PUBLIC
    nvinfer_10
    nvonnxparser_10
    cuda_lib
)