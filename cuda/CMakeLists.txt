# 编译列表
aux_source_directory(src/. SRC_CUDA) # 添加同级目录下的所有源文件
add_library(cuda_lib STATIC ${SRC_CUDA})

# 包含头文件目录
target_include_directories(cuda_lib PUBLIC
    include
    ${CUDA_INCLUDE_DIRS}
)

# 链接目录
target_link_directories(cuda_lib PUBLIC
    ${CUDA_LIBRARIES}
)

# 链接库
target_link_libraries(cuda_lib PUBLIC
    CUDA::cudart
    CUDA::cublas
    opencv_lib
)