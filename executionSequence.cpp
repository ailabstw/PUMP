#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <set>

#include "global.h"
#include "prefetch.h"
#ifdef ANALYSIS
#include "analysis.h"
#endif

void executionSequenceCudnnCublasApi(const char *name) {
  DEBUG("executionSequenceCudnnCublasApi, name = %s\n", name);
#ifdef ANALYSIS
  analysisDumpCudnnCublasApiName(name);
#endif
}

void executionSequenceCudaKernel(const char *name) {
  DEBUG("executionSequenceCudaKernel, name = %s\n", name);
#ifdef ANALYSIS
  analysisDumpCudaKernelName(name);
#endif
}

void executionSequenceCudnnCublasAccessBlock(const void *ptr, const size_t size, bool fromCudnn) {
  if ((uint64_t) ptr == (uint64_t) 0 || (uint64_t) size == (uint64_t) 0) {
    return;
  }
  DEBUG("Dependency from CUDNN/CUBLAS: ptr = %llx, size = %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
  if (fromCudnn) {
    if (controlVariables.hasCudnn == false) {
      controlVariables.hasCudnn = true;
      updateGpuMemoryFactor();
    }
  }

  if (controlVariables.hasFrameworkActive) {
    return;
  }
  pushToPrefetchCache((uint64_t) ptr, (uint64_t) size);
}

void executionSequenceCudaAccessBlock(const void *ptr, const size_t size) {
  if ((uint64_t) ptr == (uint64_t) 0 || (uint64_t) size == (uint64_t) 0) {
    return;
  }
  DEBUG("Dependency from CUDA: ptr = %llx, size = %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);

  if (controlVariables.hasFrameworkActive) {
    return;
  }
  pushToPrefetchCache((uint64_t) ptr, (uint64_t) size);
}

void executionSequenceFrameworkAccessBlock(const void *ptr, const size_t size) {
  if ((uint64_t) ptr == (uint64_t) 0 || (uint64_t) size == (uint64_t) 0) {
    return;
  }
  DEBUG("Dependency from Framework: ptr = %llx, size = %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);

  pushToPrefetchCache((uint64_t) ptr, (uint64_t) size);
}

void executionSequenceRemoveBlock(uint64_t ptr, uint64_t size) {
  invalidateMemoryBlockInPrefetchCache(ptr, size);
}
