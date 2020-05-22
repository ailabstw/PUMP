#include <string.h>
#include <cuda_runtime.h>
#include <set>
#include <queue>

#include "global.h"
#ifdef ANALYSIS
#include "analysis.h"
#endif

cudaStream_t prefetchStream;

std::map<uint64_t, uint64_t> prefetchCache;
pthread_mutex_t prefetchCacheMutex = PTHREAD_MUTEX_INITIALIZER;

uint64_t gpuMemorySize = 0;

std::queue<std::pair<cudaEvent_t, std::map<uint64_t, uint64_t>>> eventQueue;
std::map<std::pair<uint64_t, uint64_t>, uint64_t> eventQueueCount;
uint64_t eventQueueSize = 0;
pthread_mutex_t eventQueueMutex = PTHREAD_MUTEX_INITIALIZER;

void CUDART_CB dummyHostKernel(void *data) {
}

void prefetchMemoryBlock(uint64_t ptr, uint64_t size) {
  _cudaLaunchHostFunc(prefetchStream, dummyHostKernel, NULL);
  _cudaMemPrefetchAsync((void*) ptr, (size_t) size, 0, prefetchStream);
}

void prefetchStreamInit(void) {
  static bool init = false;
  if (!init) {
    init = true;
    _cudaStreamCreateWithPriority(&prefetchStream, cudaStreamNonBlocking, 1);

    cudaDeviceProp prop;
    _cudaGetDeviceProperties(&prop, 0);
    gpuMemorySize = (uint64_t) prop.totalGlobalMem;
  }
}

void pushToPrefetchCache(uint64_t ptr, uint64_t size) {
  pthread_mutex_lock(&prefetchCacheMutex);
  prefetchCache[ptr] = size;
  pthread_mutex_unlock(&prefetchCacheMutex);
}

uint64_t prefetchSize(void) {
  uint64_t size = eventQueueSize;
  DEBUG("prefetchSize(), size = %llu\n", (unsigned long long int) size);

  for (auto iterator = prefetchCache.begin(); iterator != prefetchCache.end(); ++iterator) {
    auto memoryBlock = std::pair<uint64_t, uint64_t>(iterator->first, iterator->second);
    if (eventQueueCount.find(memoryBlock) == eventQueueCount.end()) {
      size += iterator->second;
      DEBUG("prefetchSize(), size = %llu\n", (unsigned long long int) size);
    } else if (eventQueueCount[memoryBlock] == 0) {
      size += iterator->second;
      DEBUG("prefetchSize(), size = %llu\n", (unsigned long long int) size);
    }
  }

  return size;
}

void prefetchPreKernel(cudaStream_t stream) {
  DEBUG("stream = %llu, default = %llu\n", (unsigned long long int) stream, (unsigned long long int) cudaStreamPerThread);
  bool prefetchIssued = false;
  controlVariables.prefetchIndex++;

  prefetchStreamInit();

  pthread_mutex_lock(&eventQueueMutex);
  pthread_mutex_lock(&prefetchCacheMutex);
  while (prefetchSize() > (gpuMemorySize - RESERVE_GPU_MEMORY) * controlVariables.gpuMemoryFactor / 100 && eventQueue.size() > 0) {
    _cudaStreamWaitEvent(prefetchStream, eventQueue.front().first, 0);
    _cudaEventDestroy(eventQueue.front().first);

    for (auto iterator = eventQueue.front().second.begin(); iterator != eventQueue.front().second.end(); ++iterator) {
      auto memoryBlock = std::pair<uint64_t, uint64_t>(iterator->first, iterator->second);
      if (eventQueueCount.find(memoryBlock) != eventQueueCount.end()) {
        eventQueueCount[memoryBlock]--;
        if (eventQueueCount[memoryBlock] == 0) {
          eventQueueSize -= memoryBlock.second;
        } if (eventQueueCount[memoryBlock] < 0) {
          ERROR("memory block (%llx, %llu) count is negative\n", (unsigned long long int) memoryBlock.first, (unsigned long long int) memoryBlock.second);
        }
      } else {
        ERROR("memory block (%llx, %llu) not found\n", (unsigned long long int) memoryBlock.first, (unsigned long long int) memoryBlock.second);
      }
    }
    eventQueue.pop();
  }

  DEBUG("prefetchSize() = %llu\n", (unsigned long long int) prefetchSize());
  for (auto iterator = prefetchCache.begin(); iterator != prefetchCache.end(); ++iterator) {
    uint64_t ptr = iterator->first;
    uint64_t size = iterator->second;

/*
    auto memoryBlock = std::pair<uint64_t, uint64_t>(iterator->first, iterator->second);
    if (eventQueueCount.find(memoryBlock) != eventQueueCount.end()) {
      if (eventQueueCount[memoryBlock] > 0) {
        continue;
      }
    }
*/

    DEBUG("prefetch block, gpuMemory = %llu, ptr = %llx, size = %llu\n", (unsigned long long int) (gpuMemorySize - RESERVE_GPU_MEMORY) * controlVariables.gpuMemoryFactor / 100, (unsigned long long int) ptr, (unsigned long long int) size);
    if (size > gpuMemorySize) {
      continue;
    }

    prefetchMemoryBlock(ptr, size);
    prefetchIssued = true;
  }
  pthread_mutex_unlock(&prefetchCacheMutex);
  pthread_mutex_unlock(&eventQueueMutex);

  if (prefetchIssued) {
    cudaEvent_t event;
    _cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    _cudaEventRecord(event, prefetchStream);
    _cudaStreamWaitEvent(stream, event, 0);
    _cudaEventDestroy(event);
  }
}

void prefetchPostKernel(cudaStream_t stream) {
  DEBUG("stream = %llu, default = %llu\n", (unsigned long long int) stream, (unsigned long long int) cudaStreamPerThread);
  {
    cudaEvent_t event;
    _cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
    _cudaEventRecord(event, stream);

    pthread_mutex_lock(&eventQueueMutex);
    pthread_mutex_lock(&prefetchCacheMutex);
    eventQueue.push(std::pair<cudaEvent_t, std::map<uint64_t, uint64_t>>(event, prefetchCache));
    for (auto iterator = prefetchCache.begin(); iterator != prefetchCache.end(); ++iterator) {
      auto memoryBlock = std::pair<uint64_t, uint64_t>(iterator->first, iterator->second);
      if (eventQueueCount.find(memoryBlock) == eventQueueCount.end()) {
        eventQueueCount[memoryBlock] = 1;
        eventQueueSize += memoryBlock.second;
      } else {
        if (eventQueueCount[memoryBlock] == 0) {
          eventQueueSize += memoryBlock.second;
        }
        eventQueueCount[memoryBlock]++;
      }
    }

    pthread_mutex_lock(&touchedMemoryBlockFromFrameworkMutex);
    touchedMemoryBlockFromFramework.clear();
    pthread_mutex_unlock(&touchedMemoryBlockFromFrameworkMutex);
#ifdef ANALYSIS
    analysisDumpPrefetchCache(prefetchCache);
    analysisDumpEventQueueCount(eventQueueCount);
#endif
    prefetchCache.clear();

    pthread_mutex_unlock(&prefetchCacheMutex);
    pthread_mutex_unlock(&eventQueueMutex);
  }
}

void invalidateMemoryBlockInPrefetchCache(uint64_t ptr, uint64_t size) {
  pthread_mutex_lock(&prefetchCacheMutex);
  while (true) {
    if (prefetchCache.empty()) {
      break;
    }

    auto iterator = prefetchCache.lower_bound(ptr);
    if (iterator == prefetchCache.end()) {
      auto last = --prefetchCache.end();
      if (last->first + last->second > ptr) {
        prefetchCache.erase(last);
      }
      break;
    }
    if (iterator->first >= ptr + size) {
      break;
    }
    prefetchCache.erase(iterator);
  }
  pthread_mutex_unlock(&prefetchCacheMutex);
}
