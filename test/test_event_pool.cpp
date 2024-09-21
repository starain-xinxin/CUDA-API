#include "Event_Pool.h"
#include <iostream>
#include <cuda_runtime.h>
#include "gtest/gtest.h"

int main(){
    int device = 0;

    auto& pool = cT::Event::GetPool();
    {
        auto p_event = pool.get(device, cudaEventBlockingSync);
        auto p2 = pool.get(device,cudaEventBlockingSync);
    }
    
    std::cout << "设备" << device << " cudaEventBlockingSync类型的事件池拥有空闲的事件：" << pool.FreeEventCount(device, cudaEventBlockingSync);

    EXPECT_EQ(pool.FreeEventCount(device, cudaEventBlockingSync), 2);

    pool.empty_cache();

    EXPECT_EQ(pool.FreeEventCount(device, cudaEventBlockingSync), 0);
}