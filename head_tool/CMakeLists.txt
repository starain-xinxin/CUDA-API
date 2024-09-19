message(STATUS "head_tool build ...")

# add_library(KernelForTest SHARED KernelForTest.cu)
# target_include_directories(KernelForTest PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" CUDA_API_INCLUDE_LIST)    # 这里表明MathLib的头文件路径为当前目录

set(HEAD_TOOL_LINK_LIB_LIST)

foreach(lib ${HEAD_TOOL_LINK_LIB_LIST})
    list(APPEND CUDA_API_LINK_LIST ${lib})                          
    set(CUDA_API_LINK_LIST "${CUDA_API_LINK_LIST}" PARENT_SCOPE)    # 在父文件中share变量
endforeach(lib ${HEAD_TOOL_LINK_LIB_LIST})

