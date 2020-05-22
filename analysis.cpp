#include "global.h"

void analysisDumpCudnnCublasApiName(const char *name) {
  INFO("[ANALYSIS] execute: %s\n", name);
}

void analysisDumpCudaKernelName(const char *name) {
  INFO("[ANALYSIS] execute: %s\n", name);
}

void analysisDumpPrefetchCache(std::map<uint64_t, uint64_t> prefetchCache) {
  for (auto iterator = prefetchCache.begin(); iterator != prefetchCache.end(); ++iterator) {
    INFO("[ANALYSIS] prefetch: %llu %llu\n", (unsigned long long int) iterator->first, (unsigned long long int) iterator->second);
  }
}

void analysisDumpEventQueueCount(std::map<std::pair<uint64_t, uint64_t>, uint64_t> eventQueueCount) {
  for (auto iterator = eventQueueCount.begin(); iterator != eventQueueCount.end(); ++iterator) {
    if (iterator->second > 0) {
      auto memoryBlock = iterator->first;
      INFO("[ANALYSIS] gpu: %llu %llu\n", (unsigned long long int) memoryBlock.first, (unsigned long long int) memoryBlock.second);
    }
  }
}

void analysisDumpMemoryBlockActivation(uint64_t ptr, uint64_t size) {
  INFO("[ANALYSIS] activate: %llu %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
}

void analysisDumpMemoryBlockDeactivation(uint64_t ptr, uint64_t size) {
  INFO("[ANALYSIS] deactivate: %llu %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
}

void analysisDumpMemoryBlockAllocation(uint64_t ptr, uint64_t size) {
  INFO("[ANALYSIS] allocate: %llu %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
}

void analysisDumpMemoryBlockDeallocation(uint64_t ptr) {
  INFO("[ANALYSIS] deallocate: %llu\n", (unsigned long long int) ptr);
}

void analysisDumpCudaMalloc(uint64_t ptr, uint64_t size) {
  INFO("[ANALYSIS] cudaMalloc: %llu %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
}

void analysisDumpCudnnKernel(void) {
  INFO("[ANALYSIS] cudnnKernel\n");
}

void analysisDumpCublasKernel(void) {
  INFO("[ANALYSIS] cublasKernel\n");
}

void analysisDumpNativeKernel(void) {
  INFO("[ANALYSIS] nativeKernel\n");
}
