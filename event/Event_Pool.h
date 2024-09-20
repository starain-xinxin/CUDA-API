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