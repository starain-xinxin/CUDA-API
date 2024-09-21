message(STATUS "build test ...")

# 创建测试目标
enable_testing()
    add_executable(EventPoolTest test_event_pool.cpp)
    # 将测试目标汇总于TEST_TARGET_LIST变量中    
    set(TEST_TARGET_LIST
        EventPoolTest
    )
    # 为测试目标添加依赖和头文件搜索地址
    foreach(test_target ${TEST_TARGET_LIST})
        target_link_libraries(${test_target} gtest gtest_main ${CUDA_API_LINK_LIST})
        target_include_directories(${test_target} PUBLIC "${CUDA_API_INCLUDE_LIST}" ${CMAKE_CURRENT_SOURCE_DIR})
    endforeach()
add_test(NAME EventPoolTest COMMAND EventPoolTest)


