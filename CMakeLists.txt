cmake_minimum_required(VERSION 3.20)

project(CUDA_API LANGUAGES CUDA CXX C)

# -------------------目标构建需要的库和头文件路径放置在下面变量中 ------------------ #
#- 链接库 CUDA_API_LINK_LIST
#- 头文件路径 CUDA_API_INCLUDE_LIST
set(CUDA_API_LINK_LIST)
set(CUDA_API_INCLUDE_LIST)

# -------------------- CUDA设置部分 --------------------- #
find_package(CUDAToolkit REQUIRED)
message(STATUS "搜索CUDAToolkit... 版本：${CUDAToolkit_VERSION}  链接库：${CUDAToolkit_LIBRARY_DIR}  头文件路径：${CUDAToolkit_INCLUDE_DIRS}")
# nvcc编译器路径
if(${CMAKE_CUDA_COMPILER})
    message(STATUS "cmake 系统搜索 nvcc 路径: ${CMAKE_CUDA_COMPILER}")
else()
    set(CMAKE_CUDA_COMPILER /usr/local/cuda-12.2/bin/nvcc)
endif()
message(STATUS "nvcc 路径设置: ${CMAKE_CUDA_COMPILER}")
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75) # 75 是指 CUDA Compute Capability 7.5，适用于较新的 GPU
list(APPEND CUDA_API_LINK_LIST "CUDA::cudart")      # 运行时库
list(APPEND CUDA_API_LINK_LIST "CUDA::cuda_driver") # 驱动库
list(APPEND CUDA_API_INCLUDE_LIST "${CUDAToolkit_INCLUDE_DIRS}")   # CUDAToolkit头文件路径

# -----------------引入第三方库 ------------------ #
# GoogleTest测试框架
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.12.1
)
FetchContent_MakeAvailable(googletest)
# TODO: 链接库与头文件包含

# ----------------- 管理子目录 ------------------ #
set(SUBDIR_LIST 
    head_tool
    memory
    event
    stream
)
foreach(subdir ${SUBDIR_LIST})         # 包含所有子目录
    list(APPEND CUDA_API_INCLUDE_LIST "${CMAKE_SOURCE_DIR}/${subdir}")   
    add_subdirectory(${subdir})
endforeach()

# ----------------- 管理需要使用链接库的子目录 ------------------ #
set(SUBDIR_USELIB_LIST
    test
)
foreach(subdir_uselib ${SUBDIR_USELIB_LIST})
    add_subdirectory(${subdir_uselib})
endforeach()




