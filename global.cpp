#include <string.h>

#include "global.h"

struct controlVariablesStruct controlVariables;

std::map<std::pair<uint64_t, uint64_t>, uint64_t> activeMemoryBlockFromFramework;
std::map<uint64_t, uint64_t> touchedMemoryBlockFromFramework;
std::map<uint64_t, uint64_t> allocatedMemoryBlockFromFramework;
std::map<uint64_t, uint64_t> allocatedMemoryBlockFromCuda;
std::map<uint64_t, uint64_t> allMemoryBlockFromCudnnCublas;

pthread_mutex_t activeMemoryBlockFromFrameworkMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t touchedMemoryBlockFromFrameworkMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t allocatedMemoryBlockFromFrameworkMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t allocatedMemoryBlockFromCudaMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t allMemoryBlockFromCudnnCublasMutex = PTHREAD_MUTEX_INITIALIZER;

cudaError_t (*_cudaEventCreateWithFlags)(cudaEvent_t*, unsigned int);
cudaError_t (*_cudaEventDestroy)(cudaEvent_t);
cudaError_t (*_cudaEventRecord)(cudaEvent_t, cudaStream_t);
cudaError_t (*_cudaGetDeviceProperties)(cudaDeviceProp*, int);
cudaError_t (*_cudaLaunchHostFunc)(cudaStream_t, cudaHostFn_t, void*);
cudaError_t (*_cudaMemPrefetchAsync)(const void*, size_t, int, cudaStream_t);
cudaError_t (*_cudaStreamCreateWithPriority)(cudaStream_t*, unsigned int, int);
cudaError_t (*_cudaStreamWaitEvent)(cudaStream_t, cudaEvent_t, unsigned int);

void getGpuMemoryFactor(void) {
  const char *input = getenv(ULMS_GPU_MEMORY_FACTOR);
  controlVariables.gpuMemoryFactor = GPU_MEMORY_FACTOR_NO_INFO;
  if (input) {
    int gpuMemoryFactor = GPU_MEMORY_FACTOR_NO_INFO;
    if (sscanf(input, "%d", &gpuMemoryFactor) == 1) {
      controlVariables.gpuMemoryFactor = gpuMemoryFactor;
      controlVariables.customizedGpuMemoryFactor = true;
    }
  }
}

void getLogLevel(void) {
  const char *input = getenv(ULMS_LOG_LEVEL);
  controlVariables.logLevel = LOG_WARN;
  if (input) {
    if (strcmp(input, "ERROR") == 0) {
      controlVariables.logLevel = LOG_ERROR;
    } else if (strcmp(input, "WARN") == 0) {
      controlVariables.logLevel = LOG_WARN;
    } else if (strcmp(input, "INFO") == 0) {
      controlVariables.logLevel = LOG_INFO;
    } else if (strcmp(input, "DEBUG") == 0) {
      controlVariables.logLevel = LOG_DEBUG;
    }
  }
}

__attribute__((constructor))
void globalInit(void) {
  getGpuMemoryFactor();
  getLogLevel();

  INFO("ULMS enabled\n");
  if (controlVariables.customizedGpuMemoryFactor) {
    INFO("GPU memory factor is set to %d%%\n", controlVariables.gpuMemoryFactor);
  } else {
    INFO("GPU memory factor is set according to program behavior extraction.\n");
  }

  char msg[8];
  if (controlVariables.logLevel == LOG_ERROR) {
    strcpy(msg, "ERROR");
  } else if (controlVariables.logLevel == LOG_WARN) {
    strcpy(msg, "WARN");
  } else if (controlVariables.logLevel == LOG_INFO) {
    strcpy(msg, "INFO");
  } else if (controlVariables.logLevel == LOG_DEBUG) {
    strcpy(msg, "DEBUG");
  }
  INFO("Log level is set to %s\n", msg);
}

void updateGpuMemoryFactor(void) {
  if (controlVariables.customizedGpuMemoryFactor) {
    return;
  }

  if (controlVariables.hasFrameworkActive) {
    controlVariables.gpuMemoryFactor = GPU_MEMORY_FACTOR_ACTIVE;
  } else if (controlVariables.hasFrameworkMalloc) {
    controlVariables.gpuMemoryFactor = GPU_MEMORY_FACTOR_MALLOC;
  } else if (controlVariables.hasCudnn) {
    controlVariables.gpuMemoryFactor = GPU_MEMORY_FACTOR_CUDNN;
  } else {
    controlVariables.gpuMemoryFactor = GPU_MEMORY_FACTOR_NO_INFO;
  }
  INFO("GPU memory factor is set to %d%%\n", controlVariables.gpuMemoryFactor);
}
