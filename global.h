#include <pthread.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <set>
#include <map>
#include <vector>

/* #define ANALYSIS 3 */

#define ULMS_GPU_MEMORY_FACTOR "ULMS_GPU_MEMORY_FACTOR"
#define ULMS_LOG_LEVEL "ULMS_LOG_LEVEL"

#define GPU_MEMORY_FACTOR_NO_INFO 60
#define GPU_MEMORY_FACTOR_CUDNN   100
#define GPU_MEMORY_FACTOR_MALLOC  80
#define GPU_MEMORY_FACTOR_ACTIVE  100

#define RESERVE_GPU_MEMORY 0

#define ERROR(...)                                                              \
  {                                                                             \
    if (controlVariables.logLevel >= LOG_ERROR) {                               \
      fprintf(stderr, "[ulms] ERROR: %s, line %u, prefetch index = %llu: ",     \
                      __FILE__, __LINE__,                                       \
                      (unsigned long long int) controlVariables.prefetchIndex); \
      fprintf(stderr, __VA_ARGS__);                                             \
    }                                                                           \
  }

#define WARN(...)                                                               \
  {                                                                             \
    if (controlVariables.logLevel >= LOG_WARN) {                                \
      fprintf(stderr, "[ulms] WARN: %s, line %u, prefetch index = %llu: ",      \
                      __FILE__, __LINE__,                                       \
                      (unsigned long long int) controlVariables.prefetchIndex); \
      fprintf(stderr, __VA_ARGS__);                                             \
    }                                                                           \
  }

#define INFO(...)                                                               \
  {                                                                             \
    if (controlVariables.logLevel >= LOG_INFO) {                                \
      fprintf(stderr, "[ulms] INFO: %s, line %u, prefetch index = %llu: ",      \
                      __FILE__, __LINE__,                                       \
                      (unsigned long long int) controlVariables.prefetchIndex); \
      fprintf(stderr, __VA_ARGS__);                                             \
    }                                                                           \
  }

#define DEBUG(...)                                                              \
  {                                                                             \
    if (controlVariables.logLevel >= LOG_DEBUG) {                               \
      fprintf(stderr, "[ulms] DEBUG: %s, line %u, prefetch index = %llu: ",     \
                      __FILE__, __LINE__,                                       \
                      (unsigned long long int) controlVariables.prefetchIndex); \
      fprintf(stderr, __VA_ARGS__);                                             \
    }                                                                           \
  }

enum PrefetchState {
  INITIAL, PREFETCH
};

enum LogLevel {
  LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DEBUG
};

struct controlVariablesStruct {
  PrefetchState prefetchState;

  uint64_t prefetchIndex;
  uint64_t totalAllocatedMemorySizeFromCuda;
  bool customizedGpuMemoryFactor;
  int gpuMemoryFactor;

  bool hasFrameworkMalloc;
  bool hasFrameworkActive;
  bool hasCudnn;

  LogLevel logLevel;
};

extern struct controlVariablesStruct controlVariables;

extern std::map<std::pair<uint64_t, uint64_t>, uint64_t> activeMemoryBlockFromFramework;
extern std::map<uint64_t, uint64_t> touchedMemoryBlockFromFramework;
extern std::map<uint64_t, uint64_t> allocatedMemoryBlockFromFramework;
extern std::map<uint64_t, uint64_t> allocatedMemoryBlockFromCuda;
extern std::map<uint64_t, uint64_t> allMemoryBlockFromCudnnCublas;

extern pthread_mutex_t activeMemoryBlockFromFrameworkMutex;
extern pthread_mutex_t touchedMemoryBlockFromFrameworkMutex;
extern pthread_mutex_t allocatedMemoryBlockFromFrameworkMutex;
extern pthread_mutex_t allocatedMemoryBlockFromCudaMutex;
extern pthread_mutex_t allMemoryBlockFromCudnnCublasMutex;

extern cudaError_t (*_cudaEventCreateWithFlags)(cudaEvent_t*, unsigned int);
extern cudaError_t (*_cudaEventDestroy)(cudaEvent_t);
extern cudaError_t (*_cudaEventRecord)(cudaEvent_t, cudaStream_t);
extern cudaError_t (*_cudaGetDeviceProperties)(cudaDeviceProp*, int);
extern cudaError_t (*_cudaLaunchHostFunc)(cudaStream_t, cudaHostFn_t, void*);
extern cudaError_t (*_cudaMemPrefetchAsync)(const void*, size_t, int, cudaStream_t);
extern cudaError_t (*_cudaStreamCreateWithPriority)(cudaStream_t*, unsigned int, int);
extern cudaError_t (*_cudaStreamWaitEvent)(cudaStream_t, cudaEvent_t, unsigned int);

void updateGpuMemoryFactor(void);
