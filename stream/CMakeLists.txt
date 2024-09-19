message(STATUS "stream build ...")

# 需要生成目标文件的源文件列表
set(SOURCE_FILES
    stream_intro.cu
    priority.cu
)
# 对于所有cpp和cu文件，生成对应的目标文件
message(STATUS "头文件包含路径：${CUDA_API_INCLUDE_LIST}  链接库：${CUDA_API_LINK_LIST}")
foreach(SRC ${SOURCE_FILES})
    get_filename_component(TARGET_NAME ${SRC} NAME_WE)  # 获取文件名作为目标名（去掉扩展名）
    add_executable(${TARGET_NAME} ${SRC})
    target_link_libraries(${TARGET_NAME} "${CUDA_API_LINK_LIST}")
    target_include_directories(${TARGET_NAME} PUBLIC "${CUDA_API_INCLUDE_LIST}" ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()

# 生成动态库
# ...
