/* CUDA 事件池的基本实现
- 事件池架构
    ```text
    - Event_pool
        |-- SingleDevicePool(devide:0)      // 每个设备拥有独自的池
        |-- SingleDevicePool(devide:1) 
                ...
        |-- SingleDevicePool(devide:6) 
        |-- SingleDevicePool(devide:7) 
                    ｜
                    |--- PoolWithFlag(flag:0)           // 四种事件类型分别拥有独自的池
                    |--- PoolWithFlag(flag:1)  
                    |--- PoolWithFlag(flag:2)  
                    |--- PoolWithFlag(flag:3)  
    ```
- 事件在池内：通过普通的unique_ptr管理。
- 事件在池外：通过Event_ptr智能指针管理，这是一个自定义析构函数的独占式智能指针
- 单例模式：lazy初始化，通过局部静态变量实现，C++11保证线程安全

维护：
- 事件池的类型：需要修改NUM_EVENT_CLASS
 */

#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include <functional>
#include <memory>
#include <cassert>
#include <atomic>
#include "error_handling.h" 

#define NUM_EVENT_CLASS 4

namespace cT{
namespace Event{

inline int device_count(){
    int DeviceCount;
    RUNTIME_CHECK(cudaGetDeviceCount(&DeviceCount));
    return DeviceCount;
}

class EventPool{
    public:
        // 自动回收到事件池的cuda事件智能指针
        using Event_ptr = std::unique_ptr<cudaEvent_t, std::function<void(cudaEvent_t*)>>; 
        
        EventPool(): pools(device_count()){}
        ~EventPool() = default;
        EventPool(const EventPool&) = delete;
        EventPool& operator = (const EventPool&) = delete;
        EventPool(EventPool&&) = delete;
        EventPool& operator = (EventPool&&) = delete;

        static EventPool& GetEventPool(){
            static EventPool pool_instance;
            return pool_instance;
        }


        Event_ptr get(int device_id, unsigned int flag) {
            if (device_id < 0 || device_id >= device_count()) {
                throw std::out_of_range("The input of device id is " + std::to_string(device_id) + 
                                        ", valid range is [0, " + std::to_string(device_count() - 1) + "]");
            }
            if (flag >= NUM_EVENT_CLASS)
                throw std::out_of_range("The input of device id is " + std::to_string(flag) + 
                                        ", valid range is [0, " + std::to_string(NUM_EVENT_CLASS));

            auto& pool =  pools[device_id].device_pools[flag];
            auto destructor = [&pool, flag](cudaEvent_t* event){
                std::lock_guard<std::mutex> lock_pool(*pool.pool_lock_ptr.get());
                pool._event_pool.emplace_back(std::unique_ptr<cudaEvent_t> (event));
            };           // 对于用于外部使用的event，用Event_ptr智能指针保护，析构器是将事件指针直接封装成新的unique_ptr进入池中

            // 尝试从池中取空闲事件
            {
                std::lock_guard<std::mutex> lock_pool(*pool.pool_lock_ptr.get());
                if (!pool._event_pool.empty()){
                    auto* event = pool._event_pool.back().release();
                    pool._event_pool.pop_back();
                    return Event_ptr(event, destructor);
                }
            }

            // 没有空闲的就取
            auto new_ptr = std::unique_ptr<cudaEvent_t>(new cudaEvent_t);
            RUNTIME_CHECK(cudaEventCreateWithFlags(new_ptr.get(), flag));
            return Event_ptr(new_ptr.release(), destructor);
        };

        void empty_cache(){
            for (auto& d_pool : pools){
                for (auto& e_pool : d_pool.device_pools){
                    e_pool._event_pool.clear();
                }
            }
        };

        int FreeEventCount(int device_id, unsigned int flag){
            if (device_id < 0 || device_id >= device_count()) {
                throw std::out_of_range("Invalid device id");
            }
            if (flag >= NUM_EVENT_CLASS) {
                throw std::out_of_range("Invalid flag");
            }
            return pools[device_id].device_pools[flag]._event_pool.size();
        }

    private:
        // 单设备的单类型池
        struct PoolWithFlag
        {
            PoolWithFlag(unsigned int flag_): flag(flag_) {}
            std::unique_ptr<std::mutex> pool_lock_ptr;
            std::vector<std::unique_ptr<cudaEvent_t>> _event_pool;
            unsigned int flag;

        };
        // 单设备事件池
        struct SingleDevicePool{
            SingleDevicePool() {
                device_pools.reserve(NUM_EVENT_CLASS);
                for (unsigned int i = 0; i < NUM_EVENT_CLASS; ++i) {
                    device_pools.emplace_back(i);
                    device_pools.back().pool_lock_ptr = std::make_unique<std::mutex>();
                }
            }
            std::vector<PoolWithFlag> device_pools;
        };

        // 全设备池
        std::vector<SingleDevicePool> pools;
};

inline EventPool& GetPool(){
    return EventPool::GetEventPool();
}


} // namespace Event
} // namespace cuTool
