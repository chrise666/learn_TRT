# 添加opencv目录下的所有源文件
aux_source_directory(src/. SRC_OpenCV) 
add_library(opencv_lib STATIC ${SRC_OpenCV})

# 添加头文件路径
target_include_directories(opencv_lib PUBLIC
    include
    # ${OpenCV_INCLUDE_DIRS}
)

# 链接库
target_link_libraries(opencv_lib PUBLIC
    ${OpenCV_LIBS} 
)
