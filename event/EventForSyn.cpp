#include <iostream>
#include <cuda_runtime.h>
#include <memory>
#include "error_handling.h"

int main(){
    auto new_ptr = std::unique_ptr<cudaEvent_t>(new cudaEvent_t);
    RUNTIME_CHECK(cudaEventCreateWithFlags(new_ptr.get(), cudaEventBlockingSync));
    return 0;
}