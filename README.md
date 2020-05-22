# PUMP: Profiling-free Unified Memory Prefetcher for Large Model Deep Learning

PUMP is a profiling-free, framework-independent prefetch engine that improves the execution performance of large models supported by the CUDA unified memory. PUMP is transparent to the models and the frameworks, so that model developers can easily apply PUMP to existing development environments.

## Features

### Framework-Independent Large Model Support

PUMP exploits the CUDA unified memory that allows GPU memory over-subscription to provide large model support transparently to the models and the frameworks by replacing *cudaMalloc()* with *cudaMallocManaged()* through shared library hooking.

There are several advantages:

* Easy to implement: It is easy to replace *cudaMalloc()* with *cudaMallocManaged()* through shared library hooking.
* Easy to verify: It is easy to verify the logical equivalence of replacing *cudaMalloc()* and adding extra *cudaMemPrefetchAsync()*.
* Feasible for large layers: It still works when the memory footprint of a single layer is large than the GPU memory size.

### Profiling-Free Framework-Independent Prefetch

To improve the execution performance of the unified-memory-based LMS in model training and inference, PUMP implements a profiling-free framework-independent prefetch mechanism by exploiting the natures of CUDA and related libraries.

#### Memory Block Extraction

PUMP extracts the information of the accessed memory blocks from the parameters of cuDNN APIs, cuBLAS APIs and CUDA APIs. Here is how PUMP collects the information:

* cuDNN/cuBLAS APIs: The filters (weights), the tensors, and the workspaces accessed in cuDNN APIs and the matrices accessed in cuBLAS APIs are usually explicitly described in the parameters of the cuDNN and cuBLAS APIs. So, the information could be extracted by shared library hooking according to the descriptions in the *cuDNN API reference* (https://docs.nvidia.com/deeplearning/sdk/cudnn-api/index.html), and the API reference guide for cuBLAS (https://docs.nvidia.com/cuda/cublas/index.html).

* CUDA APIs: For the native kernels of the deep learning frameworks, PUMP tries to capture the information of accessed memory blocks of the upcoming kernels from the parameters of *cuLaunchKernel()* by matching the pointers in the parameters and the known memory blocks from cuDNN, cuBLAS and *cudaMalloc()*.
  
#### Memory Block Prefetching

The CUDA kernels are always executed asynchronously, so PUMP exploits the gap between kernel launching and kernel execution, and performs prefetch operations when the kernel launching to bring the required memory blocks into the GPU memory before the kernel execution. There is a bidirectional synchronization mechanism between prefetch and kernel execution to make sure that the required memory blocks are prefetched before the corresponding computation and avoid the GPU memory thrashing.

## Getting Started

### Compile from the Source

* Create a docker container with CUDA support
  ```=
  nvidia-docker run --shm-size 16G -it -d --name <container_name> nvidia/cuda:10.0-cudnn7-devel-ubuntu18.04 /bin/bash
  ```

* Enter the docker container in interactive mode
  ```=
  docker exec -it <container_name> /bin/bash
  ```

* Build the development environment
  ```=
  apt-get update
  apt-get -y install git
  ```

* Git clone this repository
  ```=
  git clone https://github.com/ailabstw/PUMP.git
  ```

* Compile the source code
  ```=
  cd pump
  make
  ```

### Execute programs with *LD_PRELOAD*

* Execute your training or inference jobs with *LD_PRELOAD*
  ```=
  LD_PRELOAD="<path_to_ulms.so>/ulms.so <path_to_libgomp.so>/libgomp.so" python <your_python_file> <your_python arguments>
  ```

### Performance tuning

* Tune *ULMS_GPU_MEMROY_FACTOR* for better execution performance
  ```=
  ULMS_GPU_MEMORY_FACTOR=<factor_number> LD_PRELOAD="<path_to_ulms.so>/ulms.so <path_to_libgomp.so>/libgomp.so" python <your_python_file> <your_python arguments>
  ```

  where *<factor_number>* should be between 0 and 100. The *factor number* stands for the percentage of GPU memory capacity that is expected to keep the prefetched memory blocks. Because there are still some unknown memory blocks that are not extracted through the memory block extraction mechanism of PUMP, the actual memory footprint is usually larger than what PUMP prefetches. In this case, if PUMP prefetches memory blocks to fill 100% GPU memory capacity, the prefetch operations will cause GPU memory thrashing and evict active memory blocks to degrade the execution performance. To limit the prefetch operations to a reasonable fraction of GPU memory, we set the *factor number* from four default values for four different situations heuristically. However, to achieve the best execution performance, setting appropriate *ULMS_GPU_MEMORY_FACTOR* for workloads individually is recommended.

### Debug

* Output debug message

  There are four log levels in PUMP: ERROR, WARN, INFO, and DEBUG. The default log level is WARN. To change the log level to increase or decrease the printed messages, please set the environment variable *ULMS_LOG_LEVEL*, e.g., `ULMS_LOG_LEVEL=DEBUG`.

## Not Supported Features

* Multiple GPU support: For now, PUMP only supports one visible GPU, because the usages of GPUs, including the kernels launched to GPUs and the memory blocks prefetched to GPU memories, need to be tracked according to the target GPUs separately by PUMP in multiple GPU environments. However, many global data structures in PUMP that track the usages of GPU now have only one copy without considering the target GPUs. Multiple GPU support is one of the future works.
