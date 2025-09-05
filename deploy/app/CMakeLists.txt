# 编译列表
aux_source_directory(. SRC) # 添加同级目录下的所有源文件
add_executable(${PROJECT_NAME} ${SRC})

# 链接库
target_link_libraries(${PROJECT_NAME} PRIVATE
    trt_lib
    cuda_lib
    opencv_lib
    onnx_inference
)

# 设置编译后的程序存放位置
set(EXECUTABLE_OUTPUT_PATH E:/workspace/learn_AI/bin)