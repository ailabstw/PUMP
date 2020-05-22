#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <map>

#include "global.h"
#include "executionSequence.h"
#ifdef ANALYSIS
#include "analysis.h"
#endif

#define KERNEL_PARAMETER_SCAN_RANGE 8
#define STRING_LENGTH 1024

std::map<uint64_t, uint64_t> memoryUsageTable;
pthread_mutex_t memoryUsageTableMutex = PTHREAD_MUTEX_INITIALIZER;

bool memoryUsageTableUpdateRequired = false;

void updateMemoryUsageTable(void) {
  char fileName[STRING_LENGTH];
  pid_t pid = getpid();
  memset(fileName, 0, sizeof(fileName));
  sprintf(fileName, "/proc/%d/maps", (int) pid);

  FILE *inFile = fopen(fileName, "r");
  if (inFile) {
    pthread_mutex_lock(&memoryUsageTableMutex);
    memoryUsageTable.clear();

    char input[STRING_LENGTH];
    while(fgets(input, sizeof(input), inFile)) {
      char address[STRING_LENGTH], perm[STRING_LENGTH];
      sscanf(input, "%s %s", address, perm);
      if (!strstr(perm, "r")) {
        continue;
      }

      char *sep = strstr(address, "-");
      unsigned long long int startAddr, endAddr;
      sscanf(address, "%llx", &startAddr);
      sscanf(sep + 1, "%llx", &endAddr);

      memoryUsageTable[(uint64_t) startAddr] = (uint64_t) endAddr;
    }
    DEBUG("Memory maps\n");
    for (auto iterator = memoryUsageTable.begin(); iterator != memoryUsageTable.end(); ++iterator) {
      DEBUG("Start = %llx, End = %llx\n", (unsigned long long int) iterator->first, (unsigned long long int) iterator->second);
    }
    pthread_mutex_unlock(&memoryUsageTableMutex);

    fclose(inFile);
  }
}

void fromCudaKernelSymbol(const char *symbolName) {
  DEBUG("CUDA symbolName = %s\n", symbolName);
  executionSequenceCudaKernel(symbolName);
}

void accessMemoryBlock(uint64_t ptr) {
  pthread_mutex_lock(&allocatedMemoryBlockFromFrameworkMutex);
  if (allocatedMemoryBlockFromFramework.find(ptr) != allocatedMemoryBlockFromFramework.end()) {
    uint64_t size = allocatedMemoryBlockFromFramework[ptr];
    pthread_mutex_unlock(&allocatedMemoryBlockFromFrameworkMutex);
    DEBUG("accessMemoryBlock (from framework malloc), addr = %llx, size = %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
    executionSequenceCudaAccessBlock((void*) ptr, (size_t) size);
    return;
  }
  pthread_mutex_unlock(&allocatedMemoryBlockFromFrameworkMutex);

  pthread_mutex_lock(&allMemoryBlockFromCudnnCublasMutex);
  if (!controlVariables.hasFrameworkMalloc && allMemoryBlockFromCudnnCublas.find(ptr) != allMemoryBlockFromCudnnCublas.end()) {
    uint64_t size = allMemoryBlockFromCudnnCublas[ptr];
    pthread_mutex_unlock(&allMemoryBlockFromCudnnCublasMutex);
    DEBUG("accessMemoryBlock (from cudnn/cublas), addr = %llx, size = %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
    executionSequenceCudaAccessBlock((void*) ptr, (size_t) size);
    return;
  }
  pthread_mutex_unlock(&allMemoryBlockFromCudnnCublasMutex);

  pthread_mutex_lock(&allocatedMemoryBlockFromCudaMutex);
  if (!controlVariables.hasFrameworkMalloc && allocatedMemoryBlockFromCuda.find(ptr) != allocatedMemoryBlockFromCuda.end()) {
    uint64_t size = allocatedMemoryBlockFromCuda[ptr];
    pthread_mutex_unlock(&allocatedMemoryBlockFromCudaMutex);
    DEBUG("accessMemoryBlock (from cuda malloc), addr = %llx, size = %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
    executionSequenceCudaAccessBlock((void*) ptr, (size_t) size);
    return;
  }
  pthread_mutex_unlock(&allocatedMemoryBlockFromCudaMutex);
}

std::pair<uint64_t, uint64_t> getAddressBoundaryFromMemoryUsageTable(uint64_t ptr) {
  pthread_mutex_lock(&memoryUsageTableMutex);
  if (memoryUsageTable.size() > 0) {
    auto iterator = memoryUsageTable.upper_bound(ptr);
    if (iterator != memoryUsageTable.begin()) {
      --iterator;
      if (iterator->first <= ptr && ptr <= iterator->second) {
        auto ret = std::pair<uint64_t, uint64_t>(iterator->first, iterator->second);
        pthread_mutex_unlock(&memoryUsageTableMutex);
        return ret;
      }
    }
  }
  pthread_mutex_unlock(&memoryUsageTableMutex);
  return std::pair<uint64_t, uint64_t>(0, 0);
}

void fromCudaKernelParameters(void **kernelParams) {
  if (memoryUsageTableUpdateRequired) {
    updateMemoryUsageTable();
    memoryUsageTableUpdateRequired = false;
  }

  DEBUG("fromCudaKernelParameters, kernelParams = %llx\n", (unsigned long long int) kernelParams);
  uint64_t targetAddr = (uint64_t) kernelParams;
  auto addressBoundary = getAddressBoundaryFromMemoryUsageTable(targetAddr);
  DEBUG("fromCudaKernelParameters, address start = %llx, address end = %llx\n", (unsigned long long int) addressBoundary.first, (unsigned long long int) addressBoundary.second);

  for (int index = 0; index < KERNEL_PARAMETER_SCAN_RANGE; index++) {
    DEBUG("fromCudaKernelParameters, kernelParams[%d] = %llx\n", index, (unsigned long long int) kernelParams[index]);
    if (addressBoundary.first != addressBoundary.second && addressBoundary.first <= (uint64_t) kernelParams[index] && (uint64_t) kernelParams[index] <= addressBoundary.second) {
      DEBUG("fromCudaKernelParameters, try to deref kernelParams[%d]\n", index);
      uint64_t ptr = (uint64_t) *(void**) kernelParams[index];
      DEBUG("fromCudaKernelParameters, deref-kernelParams[%d] = %llx\n", index, (unsigned long long int) ptr);
      accessMemoryBlock(ptr);
    }
  }
}

void addAllocatedMemoryBlockFromCuda(uint64_t ptr, uint64_t size) {
  pthread_mutex_lock(&allocatedMemoryBlockFromCudaMutex);
  allocatedMemoryBlockFromCuda[ptr] = size;
  controlVariables.totalAllocatedMemorySizeFromCuda += size;
  pthread_mutex_unlock(&allocatedMemoryBlockFromCudaMutex);
  DEBUG("totalAllocatedMemorySizeFromCuda = %llu\n", (unsigned long long int) controlVariables.totalAllocatedMemorySizeFromCuda);
}

void invalidateMemoryBlockFromCudnnCublas(uint64_t ptr, uint64_t size) {
  pthread_mutex_lock(&allMemoryBlockFromCudnnCublasMutex);
  while (true) {
    if (allMemoryBlockFromCudnnCublas.empty()) {
      break;
    }

    auto iterator = allMemoryBlockFromCudnnCublas.lower_bound(ptr);
    if (iterator == allMemoryBlockFromCudnnCublas.end()) {
      auto last = --allMemoryBlockFromCudnnCublas.end();
      if (last->first + last->second > ptr) {
        allMemoryBlockFromCudnnCublas.erase(last);
      }
      break;
    }
    if (iterator->first >= ptr + size) {
      break;
    }
    allMemoryBlockFromCudnnCublas.erase(iterator);
  }
  pthread_mutex_unlock(&allMemoryBlockFromCudnnCublasMutex);
}

void delAllocatedMemoryBlockFromCuda(uint64_t ptr) {
  pthread_mutex_lock(&allocatedMemoryBlockFromCudaMutex);
  if (allocatedMemoryBlockFromCuda.find(ptr) != allocatedMemoryBlockFromCuda.end()) {
    invalidateMemoryBlockFromCudnnCublas(ptr, allocatedMemoryBlockFromCuda[ptr]);
    executionSequenceRemoveBlock(ptr, allocatedMemoryBlockFromCuda[ptr]);
    controlVariables.totalAllocatedMemorySizeFromCuda -= allocatedMemoryBlockFromCuda[ptr];
    DEBUG("totalAllocatedMemorySizeFromCuda = %llu\n", (unsigned long long int) controlVariables.totalAllocatedMemorySizeFromCuda);
    allocatedMemoryBlockFromCuda.erase(ptr);
  }
  pthread_mutex_unlock(&allocatedMemoryBlockFromCudaMutex);
}

void fromCudaMalloc(void* ptr, size_t size) {
  DEBUG("Allocate memory block: addr = %llx, size = %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
  addAllocatedMemoryBlockFromCuda((uint64_t) ptr, (uint64_t) size);
#ifdef ANALYSIS
  analysisDumpCudaMalloc((uint64_t) ptr, (uint64_t) size);
#endif
  memoryUsageTableUpdateRequired = true;
}

void fromCudaFree(void* ptr) {
  DEBUG("Deallocate memory block: addr = %llx\n", (unsigned long long int) ptr);
  delAllocatedMemoryBlockFromCuda((uint64_t) ptr);

  memoryUsageTableUpdateRequired = true;
}
