#include <stdio.h>
#include <cstdint>

#include "global.h"
#include "executionSequence.h"
#ifdef ANALYSIS
#include "analysis.h"
#endif

extern "C" void fromFrameworkEnterFunction(void) {
}

extern "C" void fromFrameworkExitFunction(void) {
}

void addActiveMemoryBlockFromFramework(uint64_t ptr, uint64_t size) {
  auto memoryBlock = std::pair<uint64_t, uint64_t>(ptr, size);
  pthread_mutex_lock(&activeMemoryBlockFromFrameworkMutex);
  if (activeMemoryBlockFromFramework.find(memoryBlock) == activeMemoryBlockFromFramework.end()) {
    activeMemoryBlockFromFramework[memoryBlock] = 1;
    executionSequenceFrameworkAccessBlock((void*) ptr, (size_t) size);
  } else {
    activeMemoryBlockFromFramework[memoryBlock]++;
  }
  pthread_mutex_unlock(&activeMemoryBlockFromFrameworkMutex);
}

void subActiveMemoryBlockFromFramework(uint64_t ptr, uint64_t size) {
  auto memoryBlock = std::pair<uint64_t, uint64_t>(ptr, size);
  pthread_mutex_lock(&activeMemoryBlockFromFrameworkMutex);
  if (activeMemoryBlockFromFramework.find(memoryBlock) != activeMemoryBlockFromFramework.end()) {
    activeMemoryBlockFromFramework[memoryBlock]--;
    if (activeMemoryBlockFromFramework[memoryBlock] == 0) {
      activeMemoryBlockFromFramework.erase(memoryBlock);
      executionSequenceRemoveBlock(ptr, size);
    }
  }
  pthread_mutex_unlock(&activeMemoryBlockFromFrameworkMutex);
}

void addTouchedMemoryBlockFromFramework(uint64_t ptr, uint64_t size) {
  /*
  touchedMemoryBlockFromFramework[ptr] = size;
  executionSequenceFrameworkAccessBlock((void*) ptr, (size_t) size);
  */
}

void addAllocatedMemoryBlockFromFramework(uint64_t ptr, uint64_t size) {
  pthread_mutex_lock(&allocatedMemoryBlockFromFrameworkMutex);
  allocatedMemoryBlockFromFramework[ptr] = size;
  pthread_mutex_unlock(&allocatedMemoryBlockFromFrameworkMutex);
  executionSequenceFrameworkAccessBlock((void*) ptr, (size_t) size);
}

void delAllocatedMemoryBlockFromFramework(uint64_t ptr) {
  pthread_mutex_lock(&allocatedMemoryBlockFromFrameworkMutex);
  if (allocatedMemoryBlockFromFramework.find(ptr) != allocatedMemoryBlockFromFramework.end()) {
    executionSequenceRemoveBlock(ptr, allocatedMemoryBlockFromFramework[ptr]);
    allocatedMemoryBlockFromFramework.erase(ptr);
  }
  pthread_mutex_unlock(&allocatedMemoryBlockFromFrameworkMutex);
}

extern "C" void fromFrameworkActivateTensor(uint64_t ptr, uint64_t size) {
  DEBUG("Activate memory block: addr = %llx, size = %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
#ifdef ANALYSIS
  analysisDumpMemoryBlockActivation(ptr, size);
  if (ANALYSIS != 3) {
    return;
  }
#endif

  addActiveMemoryBlockFromFramework(ptr, size);

  if (controlVariables.hasFrameworkActive == false) {
    controlVariables.hasFrameworkActive = true;
    updateGpuMemoryFactor();
  }
}

extern "C" void fromFrameworkDeactivateTensor(uint64_t ptr, uint64_t size) {
  DEBUG("Deactivate memory block: addr = %llx, size = %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
#ifdef ANALYSIS
  analysisDumpMemoryBlockDeactivation(ptr, size);
  if (ANALYSIS != 3) {
    return;
  }
#endif

  subActiveMemoryBlockFromFramework(ptr, size);
}

extern "C" void fromFrameworkTouchTensor(uint64_t ptr, uint64_t size) {
  DEBUG("Touch memory block: addr = %llx, size = %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
#ifdef ANALYSIS
  if (ANALYSIS != 3) {
    return;
  }
#endif

  addTouchedMemoryBlockFromFramework(ptr, size);
}

extern "C" void fromFrameworkMalloc(void* ptr, size_t size) {
  DEBUG("Allocate memory block: addr = %llx, size = %llu\n", (unsigned long long int) ptr, (unsigned long long int) size);
#ifdef ANALYSIS
  analysisDumpMemoryBlockAllocation((uint64_t) ptr, (uint64_t) size);
  if (ANALYSIS != 2 && ANALYSIS != 3) {
    return;
  }
#endif

  addAllocatedMemoryBlockFromFramework((uint64_t) ptr, (uint64_t) size);

  if (controlVariables.hasFrameworkMalloc == false) {
    controlVariables.hasFrameworkMalloc = true;
    updateGpuMemoryFactor();
  }
}

extern "C" void fromFrameworkFree(void* ptr) {
  DEBUG("Deallocate memory block: addr = %llx\n", (unsigned long long int) ptr);
#ifdef ANALYSIS
  analysisDumpMemoryBlockDeallocation((uint64_t) ptr);
  if (ANALYSIS != 2 && ANALYSIS != 3) {
    return;
  }
#endif

  delAllocatedMemoryBlockFromFramework((uint64_t) ptr);
}
