#pragma once
#include <cuda_runtime.h>
#include <vector>
#include <mutex>
#include "error_handling.h"

namespace cuTool{
namespace Event{

struct EventWithFlag
{
    cudaEvent_t event;      
    unsigned int flag;     
};

struct PoolWithFlag
{
    PoolWithFlag(unsigned int flag_): flag(flag_) {}
    std::mutex pool_lock;
    std::vector<EventWithFlag> pool;
    unsigned int flag;

};

class SingleDevicePool{
    public:
        PoolWithFlag Pool_cudaEventDefault{cudaEventDefault};
        PoolWithFlag Pool_cudaEventBlockingSync{cudaEventBlockingSync};
        PoolWithFlag Pool_cudaEventDisableTiming{cudaEventDisableTiming};
        PoolWithFlag Pool_cudaEventInterprocess{cudaEventInterprocess};
};

class EventPool {
    public:
        EventWithFlag get(int device_id);
        void empty_cache();

    private:
        std::vector<SingleDevicePool> pools;
};




} // namespace Event
} // namespace cuTool