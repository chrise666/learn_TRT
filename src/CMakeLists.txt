# 添加项目自身的库和依赖
set(CUDA_TOOLKIT_ROOT_DIR "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8") # 设置 cuda 路径
set(TENSORRT_DIR "E:/workspace/TensorRT-10.9.0.34") # 设置 TensorRT 路径

aux_source_directory(. SRC) # 添加同级目录下的所有源文件

# 添加第三方库
# find_package(OpenCV REQUIRED)

# 包含头文件目录
include_directories(../include)
include_directories(
    ${CUDA_TOOLKIT_ROOT_DIR}/include
    ${TENSORRT_DIR}/include
    ${TENSORRT_DIR}/samples/common
)

# 链接库路径
link_directories(
    ${CUDA_LIBRARIES}
    ${TENSORRT_DIR}/lib
)

# 编译列表
add_executable(${PROJECT_NAME} ${SRC})
# add_library(gpu SHARED ${SRC_CUDA})

# 链接库
target_link_libraries(${PROJECT_NAME}
    nvinfer_10
    nvonnxparser_10
    cudart
    cuda
)
# target_link_libraries(${PROJECT_NAME} gpu)
