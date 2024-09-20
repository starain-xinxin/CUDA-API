message(STATUS "build test ...")

# 创建测试目标
# add_executable(test_main test_main.cpp)

# 将测试目标汇总于TEST_TARGET_LIST变量中    
set(TEST_TARGET_LIST

)
foreach(target ${TEST_TARGET_LIST})
    target_link_libraries(${target} "${CUDA_API_LINK_LIST}")
    target_include_directories(${target} PUBLIC "${CUDA_API_INCLUDE_LIST}" ${CMAKE_CURRENT_SOURCE_DIR})
endforeach()
