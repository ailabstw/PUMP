#include <stdio.h>
#include <cudnn.h>

#include "executionSequence.h"
#include "global.h"

enum AccessType {
  IN, OUT, INOUT
};

void fromCudnnApiName(const char *apiName) {
  DEBUG("CUDNN apiName = %s\n", apiName);
  executionSequenceCudnnCublasApi(apiName);
}

void pushToAllMemoryBlockFromCudnn(uint64_t ptr, uint64_t size) {
  pthread_mutex_lock(&allMemoryBlockFromCudnnCublasMutex);
  allMemoryBlockFromCudnnCublas[ptr] = size;
  pthread_mutex_unlock(&allMemoryBlockFromCudnnCublasMutex);
}

/* tensor part */

std::map<cudnnTensorDescriptor_t, size_t> fromCudnnTensorDescriptors;
pthread_mutex_t fromCudnnTensorDescriptorsMutex = PTHREAD_MUTEX_INITIALIZER;

void fromCudnnCreateTensorDesc(const cudnnTensorDescriptor_t *tensorDesc) {
  pthread_mutex_lock(&fromCudnnTensorDescriptorsMutex);
  fromCudnnTensorDescriptors[*tensorDesc] = (size_t) 0;
  pthread_mutex_unlock(&fromCudnnTensorDescriptorsMutex);
}

void fromCudnnDestroyTensorDesc(const cudnnTensorDescriptor_t tensorDesc) {
  pthread_mutex_lock(&fromCudnnTensorDescriptorsMutex);
  fromCudnnTensorDescriptors.erase(tensorDesc);
  pthread_mutex_unlock(&fromCudnnTensorDescriptorsMutex);
}

void fromCudnnSetTensorDescSize(const cudnnTensorDescriptor_t tensorDesc) {
  if (!tensorDesc) {
    return;
  } else {
    size_t tensorSize;
    if (cudnnGetTensorSizeInBytes(tensorDesc, &tensorSize) != CUDNN_STATUS_SUCCESS) {
      ERROR("Error: cudnnGetTensorSizeInBytes() failed.\n");
      return;
    } else {
      pthread_mutex_lock(&fromCudnnTensorDescriptorsMutex);
      fromCudnnTensorDescriptors[tensorDesc] = tensorSize;
      pthread_mutex_unlock(&fromCudnnTensorDescriptorsMutex);
    }
  }
}

void fromCudnnAccessTensor(const cudnnTensorDescriptor_t desc, const void *tensor, AccessType type) {
  if (!desc || !tensor) {
    return;
  } else {
    size_t tensorSize;
    if (cudnnGetTensorSizeInBytes(desc, &tensorSize) != CUDNN_STATUS_SUCCESS) {
      ERROR("Error: cudnnGetTensorSizeInBytes() failed.\n");
      return;
    } else {
      switch (type) {
        case IN:
          DEBUG("fromCudnnAccessTensor, addr = %llx, size = %llu, type = input\n", (unsigned long long int) tensor, (unsigned long long int) tensorSize);
          break;
        case OUT:
          DEBUG("fromCudnnAccessTensor, addr = %llx, size = %llu, type = output\n", (unsigned long long int) tensor, (unsigned long long int) tensorSize);
          break;
        case INOUT:
          DEBUG("fromCudnnAccessTensor, addr = %llx, size = %llu, type = inout\n", (unsigned long long int) tensor, (unsigned long long int) tensorSize);
          break;
        default:
          break;
      }
      executionSequenceCudnnCublasAccessBlock((void*) tensor, (size_t) tensorSize, true);
      pushToAllMemoryBlockFromCudnn((uint64_t) tensor, (uint64_t) tensorSize);
    }
  }
}

void fromCudnnInputTensor(const cudnnTensorDescriptor_t desc, const void *tensor) {
  fromCudnnAccessTensor(desc, tensor, IN);
}

void fromCudnnOutputTensor(const cudnnTensorDescriptor_t desc, const void *tensor) {
  fromCudnnAccessTensor(desc, tensor, OUT);
}

void fromCudnnInputOutputTensor(const cudnnTensorDescriptor_t desc, const void *tensor) {
  fromCudnnAccessTensor(desc, tensor, INOUT);
}

void fromCudnnInputTensorArray(const cudnnTensorDescriptor_t *desc, const void *tensor, const int length) {
  uint64_t offset = 0;
  for (int index = 0; index < length; index++) {
    size_t tensorSize;
    if (cudnnGetTensorSizeInBytes(desc[index], &tensorSize) != CUDNN_STATUS_SUCCESS) {
      break;
    }

    fromCudnnInputTensor(desc[index], (void*) ((uint64_t) tensor + offset));
    offset += tensorSize;
  }
}

void fromCudnnOutputTensorArray(const cudnnTensorDescriptor_t *desc, const void *tensor, const int length) {
  uint64_t offset = 0;
  for (int index = 0; index < length; index++) {
    size_t tensorSize;
    if (cudnnGetTensorSizeInBytes(desc[index], &tensorSize) != CUDNN_STATUS_SUCCESS) {
      break;
    }

    fromCudnnOutputTensor(desc[index], (void*) ((uint64_t) tensor + offset));
    offset += tensorSize;
  }
}

void fromCudnnInputOutputTensorArray(const cudnnTensorDescriptor_t *desc, const void *tensor, const int length) {
  uint64_t offset = 0;
  for (int index = 0; index < length; index++) {
    size_t tensorSize;
    if (cudnnGetTensorSizeInBytes(desc[index], &tensorSize) != CUDNN_STATUS_SUCCESS) {
      break;
    }

    fromCudnnInputOutputTensor(desc[index], (void*) ((uint64_t) tensor + offset));
    offset += tensorSize;
  }
}

void fromCudnnWorkspace(const void *workspace, size_t bytes) {
  if (!workspace || !bytes) {
    return;
  } else {
    DEBUG("fromCudnnWorkspace, addr = %llx, size = %llu, type = workspace\n", (unsigned long long int) workspace, (unsigned long long int) bytes);
    executionSequenceCudnnCublasAccessBlock(workspace, bytes, true);
    pushToAllMemoryBlockFromCudnn((uint64_t) workspace, (uint64_t) bytes);
  }
}

/* filter part */

std::map<cudnnFilterDescriptor_t, size_t> fromCudnnFilterDescriptors;
pthread_mutex_t fromCudnnFilterDescriptorsMutex = PTHREAD_MUTEX_INITIALIZER;

void fromCudnnCreateFilterDesc(const cudnnFilterDescriptor_t *filterDesc) {
  pthread_mutex_lock(&fromCudnnFilterDescriptorsMutex);
  fromCudnnFilterDescriptors[*filterDesc] = (size_t) 0;
  pthread_mutex_unlock(&fromCudnnFilterDescriptorsMutex);
}

void fromCudnnDestroyFilterDesc(const cudnnFilterDescriptor_t filterDesc) {
  pthread_mutex_lock(&fromCudnnFilterDescriptorsMutex);
  fromCudnnFilterDescriptors.erase(filterDesc);
  pthread_mutex_unlock(&fromCudnnFilterDescriptorsMutex);
}

size_t fromCudnnGetSizeFromDataType(const cudnnDataType_t dataType) {
  switch (dataType) {
    case CUDNN_DATA_FLOAT:
      /* The data is a 32-bit single-precision floating-point (float). */
      return (size_t) 4;
      break;
    case CUDNN_DATA_DOUBLE:
      /* The data is a 64-bit double-precision floating-point (double). */
      return (size_t) 8;
      break;
    case CUDNN_DATA_HALF:
      /* The data is a 16-bit floating-point. */
      return (size_t) 2;
      break;
    case CUDNN_DATA_INT8:
      /* The data is an 8-bit signed integer. */
      return (size_t) 1;
      break;
    case CUDNN_DATA_UINT8:
      /* he data is an 8-bit unsigned integer. */
      return (size_t) 1;
      break;
    case CUDNN_DATA_INT32:
      /* The data is a 32-bit signed integer. */
      return (size_t) 4;
      break;
    case CUDNN_DATA_INT8x4:
      /* The data is 32-bit elements each composed of 4 8-bit signed integers. This data type is only supported with tensor format CUDNN_TENSOR_NCHW_VECT_C. */
      return (size_t) 4;
      break;
    case CUDNN_DATA_INT8x32:
      /* The data is 32-element vectors, each element being an 8-bit signed integer. This data type is only supported with the tensor format CUDNN_TENSOR_NCHW_VECT_C. Moreover, this data type can only be used with algo 1, meaning, CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_???PRECOMP_GEMM. For more information, see cudnnConvolutionFwdAlgo_t. */
      return (size_t) 32;
      break;
    case CUDNN_DATA_UINT8x4:
      /* The data is 32-bit elements each composed of 4 8-bit unsigned integers. This data type is only supported with tensor format CUDNN_TENSOR_NCHW_VECT_C. */
      return (size_t) 4;
      break;
    default:
      fprintf(stderr, "[ulms-access] Error: Unknown dataType in accessSetFilter4dDesc.\n");
      return (size_t) 1;
      break;
  }
  return (size_t) 1;
}

void fromCudnnSetFilterDescSize(const cudnnFilterDescriptor_t filterDesc) {
  if (!filterDesc) {
    return;
  } else {
    size_t filterSize;
    if (cudnnGetFilterSizeInBytes(filterDesc, &filterSize) != CUDNN_STATUS_SUCCESS) {
      ERROR("Error: cudnnGetFilterSizeInBytes() failed.\n");
      return;
    } else {
      pthread_mutex_lock(&fromCudnnFilterDescriptorsMutex);
      fromCudnnFilterDescriptors[filterDesc] = filterSize;
      pthread_mutex_unlock(&fromCudnnFilterDescriptorsMutex);
    }
  }
}

void fromCudnnSetFilter4dDesc(const cudnnFilterDescriptor_t filterDesc,
                         const cudnnDataType_t dataType,
                         const cudnnTensorFormat_t format,
                         const int k,
                         const int c,
                         const int h,
                         const int w) {
  size_t filterSize = fromCudnnGetSizeFromDataType(dataType);
  filterSize *= k;
  filterSize *= c;
  filterSize *= h;
  filterSize *= w;

  pthread_mutex_lock(&fromCudnnFilterDescriptorsMutex);
  fromCudnnFilterDescriptors[filterDesc] = filterSize;
  pthread_mutex_unlock(&fromCudnnFilterDescriptorsMutex);
}

void fromCudnnSetFilterNdDesc(const cudnnFilterDescriptor_t filterDesc,
                         const cudnnDataType_t dataType,
                         const cudnnTensorFormat_t format,
                         const int nbDims,
                         const int filterDimA[]) {
  size_t filterSize = fromCudnnGetSizeFromDataType(dataType);
  for (int index = 0; index < nbDims; index++) {
    filterSize *= filterDimA[index];
  }

  pthread_mutex_lock(&fromCudnnFilterDescriptorsMutex);
  fromCudnnFilterDescriptors[filterDesc] = filterSize;
  pthread_mutex_unlock(&fromCudnnFilterDescriptorsMutex);
}

void fromCudnnAccessFilter(const cudnnFilterDescriptor_t desc, const void *filter, AccessType type) {
  if (!desc || !filter) {
    return;
  } else {
    switch (type) {
      case IN:
        DEBUG("fromCudnnAccessFilter, addr = %llx, size = %llu, type = input\n", (unsigned long long int) filter, (unsigned long long int) fromCudnnFilterDescriptors[desc]);
        break;
      case OUT:
        DEBUG("fromCudnnAccessFilter, addr = %llx, size = %llu, type = output\n", (unsigned long long int) filter, (unsigned long long int) fromCudnnFilterDescriptors[desc]);
        break;
      case INOUT:
        DEBUG("fromCudnnAccessFilter, addr = %llx, size = %llu, type = inout\n", (unsigned long long int) filter, (unsigned long long int) fromCudnnFilterDescriptors[desc]);
        break;
      default:
        break;
    }
    executionSequenceCudnnCublasAccessBlock((void*) filter, (size_t) fromCudnnFilterDescriptors[desc], true);
    pushToAllMemoryBlockFromCudnn((uint64_t) filter, (uint64_t) fromCudnnFilterDescriptors[desc]);
  }
}

void fromCudnnInputFilter(const cudnnFilterDescriptor_t desc, const void *filter) {
  fromCudnnAccessFilter(desc, filter, IN);
}

void fromCudnnOutputFilter(const cudnnFilterDescriptor_t desc, const void *filter) {
  fromCudnnAccessFilter(desc, filter, OUT);
}

void fromCudnnInputOutputFilter(const cudnnFilterDescriptor_t desc, const void *filter) {
  fromCudnnAccessFilter(desc, filter, INOUT);
}
