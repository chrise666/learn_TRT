cmake_minimum_required(VERSION 3.20) # 设置Cmake最低版本

project(demo LANGUAGES CXX CUDA) # 激活C++与cuda语言支持

# 1. 全局设置
set(CMAKE_CXX_STANDARD 17)              # 设置 C++ 标准
set(CMAKE_CXX_STANDARD_REQUIRED ON)     # 强制使用设置的 C++ 标准
set(CMAKE_CUDA_STANDARD 17)             # 设置 CUDA 标准
set(CMAKE_CUDA_STANDARD_REQUIRED ON)    # 强制使用设置的 CUDA 标准

# 设置源文件的编码为 UTF-8
# add_compile_options(
#     "$<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:/source-charset:utf-8>"
#     # "$<$<AND:$<COMPILE_LANGUAGE:C>,$<C_COMPILER_ID:MSVC>>:/source-charset:utf-8>"
#     # "$<$<AND:$<COMPILE_LANGUAGE:CUDA>,$<CUDA_COMPILER_ID:NVIDIA>>:-Xcompiler /source-charset:utf-8>"
# )
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler /source-charset:utf-8")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /source-charset:utf-8")

# 2. 查找所有依赖库
# OpenCV
set(OpenCV_DIR "D:/AAA_ZJF/Lib/opencv/build") # 手动设置 OpenCV 路径
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
else()
    message(WARNING "OpenCV not found, app will be disabled")
endif()

# CUDA
find_package(CUDAToolkit REQUIRED)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDAToolkit found: ${CUDAToolkit_VERSION}")
else()
    message(WARNING "CUDAToolkit not found, app will be disabled")
endif()

# 3. 添加子目录
add_subdirectory(./app)
add_subdirectory(./cuda)
add_subdirectory(./opencv)
add_subdirectory(./tensorrt)
add_subdirectory(./ort)