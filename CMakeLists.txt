cmake_minimum_required(VERSION 3.20) # 设置Cmake最低版本

project(demo LANGUAGES CXX CUDA) # 激活C++与cuda语言支持

set(CMAKE_CXX_STANDARD 17)  # 设置 C++ 标准
# set(CMAKE_CUDA_STANDARD 17) # 设置 CUDA 标准

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin) # 设置编译后的程序存放位置

# 设置源文件的编码为 UTF-8
add_compile_options(
    "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/source-charset:utf-8>"
    # "$<$<AND:$<COMPILE_LANGUAGE:C>,$<C_COMPILER_ID:MSVC>>:/source-charset:utf-8>"
    # "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler /source-charset:utf-8>"
)

add_subdirectory(./src) # 添加子目录
