message(STATUS "event build ...")

# 生成动态库
add_library(EventPool SHARED Event_Pool.cpp)
target_include_directories(EventPool PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CUDA_API_INCLUDE_LIST}") 

# 所有生成的动态库都加入链接库列表
set(EVENT_LINK_LIB_LIST 
    EventPool
)
# 将链接库列表传递给父级作用域
foreach(lib ${EVENT_LINK_LIB_LIST})
    list(APPEND CUDA_API_LINK_LIST ${lib})
    set(CUDA_API_LINK_LIST ${CUDA_API_LINK_LIST} PARENT_SCOPE)
endforeach()


# 需要生成目标文件的源文件列表
set(SOURCE_FILES
    EventForTime.cu
    EventForSyn.cu
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