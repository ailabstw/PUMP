#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
using namespace std;

#define CUBLAS_ENTER_API(name)                                                 \
  DEBUG("Enter " STRINGIFY(name) "()\n");                                      \
  typedef decltype(&name) funcType;                                            \
  funcType func = (funcType) actualDlsym(libcublasHandle, STRINGIFY(name));    \

#define CUBLAS_ENTER_COMPUTE_KERNEL(name, handle)                              \
  fromCublasApiName(STRINGIFY(name));                                          \
  cudaStream_t stream;                                                         \
  cublasGetStream(handle, &stream);                                            \
  prefetchPreKernel(stream);                                                   \
  if (inCudnnCublasInvocation) {                                               \
    ERROR("ERROR\n");                                                          \
  }                                                                            \
//  inCudnnCublasInvocation = true;                                              \
  inCublasInvocation = true;                                                   \

#define CUBLAS_LEAVE_COMPUTE_KERNEL(handle)                                    \
//  inCudnnCublasInvocation = false;                                             \
  inCublasInvocation = false;                                                  \
  prefetchPostKernel(stream);                                                  \

cublasStatus_t cublasCreate(cublasHandle_t *handle) {
  CUBLAS_ENTER_API(cublasCreate_v2);
  cublasStatus_t ret = func(handle);
  return ret;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
  CUBLAS_ENTER_API(cublasDestroy_v2);
  cublasStatus_t ret = func(handle);
  return ret;
}

cublasStatus_t cublasGetVersion(cublasHandle_t handle, int *version) {
  CUBLAS_ENTER_API(cublasGetVersion_v2);
  cublasStatus_t ret = func(handle, version);
  return ret;
}

cublasStatus_t cublasGetProperty(libraryPropertyType type, int *value) {
  CUBLAS_ENTER_API(cublasGetProperty);
  cublasStatus_t ret = func(type, value);
  return ret;
}

size_t cublasGetCudartVersion() {
  CUBLAS_ENTER_API(cublasGetCudartVersion);
  size_t ret = func();
  return ret;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t streamId) {
  CUBLAS_ENTER_API(cublasSetStream_v2);
  cublasStatus_t ret = func(handle, streamId);
  return ret;
}

cublasStatus_t cublasGetStream(cublasHandle_t handle, cudaStream_t *streamId) {
  CUBLAS_ENTER_API(cublasGetStream_v2);
  cublasStatus_t ret = func(handle, streamId);
  return ret;
}

cublasStatus_t cublasGetPointerMode(cublasHandle_t handle,
                                    cublasPointerMode_t *mode) {
  CUBLAS_ENTER_API(cublasGetPointerMode_v2);
  cublasStatus_t ret = func(handle, mode);
  return ret;
}

cublasStatus_t cublasSetPointerMode(cublasHandle_t handle,
                                    cublasPointerMode_t mode) {
  CUBLAS_ENTER_API(cublasSetPointerMode_v2);
  cublasStatus_t ret = func(handle, mode);
  return ret;
}

cublasStatus_t cublasGetAtomicsMode(cublasHandle_t handle,
                                    cublasAtomicsMode_t *mode) {
  CUBLAS_ENTER_API(cublasGetAtomicsMode);
  cublasStatus_t ret = func(handle, mode);
  return ret;
}

cublasStatus_t cublasSetAtomicsMode(cublasHandle_t handle,
                                    cublasAtomicsMode_t mode) {
  CUBLAS_ENTER_API(cublasSetAtomicsMode);
  cublasStatus_t ret = func(handle, mode);
  return ret;
}

cublasStatus_t cublasGetMathMode(cublasHandle_t handle, cublasMath_t *mode) {
  CUBLAS_ENTER_API(cublasGetMathMode);
  cublasStatus_t ret = func(handle, mode);
  return ret;
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode) {
  CUBLAS_ENTER_API(cublasSetMathMode);
  cublasStatus_t ret = func(handle, mode);
  return ret;
}

cublasStatus_t cublasLoggerConfigure(int logIsOn, int logToStdOut,
                                     int logToStdErr, const char *logFileName) {
  CUBLAS_ENTER_API(cublasLoggerConfigure);
  cublasStatus_t ret = func(logIsOn, logToStdOut, logToStdErr, logFileName);
  return ret;
}

cublasStatus_t cublasSetLoggerCallback(cublasLogCallback userCallback) {
  CUBLAS_ENTER_API(cublasSetLoggerCallback);
  cublasStatus_t ret = func(userCallback);
  return ret;
}

cublasStatus_t cublasGetLoggerCallback(cublasLogCallback *userCallback) {
  CUBLAS_ENTER_API(cublasGetLoggerCallback);
  cublasStatus_t ret = func(userCallback);
  return ret;
}

cublasStatus_t cublasSetVector(int n, int elemSize, const void *x, int incx,
                               void *devicePtr, int incy) {
  CUBLAS_ENTER_API(cublasSetVector);
  cublasStatus_t ret = func(n, elemSize, x, incx, devicePtr, incy);
  return ret;
}

cublasStatus_t cublasGetVector(int n, int elemSize, const void *x, int incx,
                               void *y, int incy) {
  CUBLAS_ENTER_API(cublasGetVector);
  cublasStatus_t ret = func(n, elemSize, x, incx, y, incy);
  return ret;
}

cublasStatus_t cublasSetMatrix(int rows, int cols, int elemSize, const void *A,
                               int lda, void *B, int ldb) {
  CUBLAS_ENTER_API(cublasSetMatrix);
  cublasStatus_t ret = func(rows, cols, elemSize, A, lda, B, ldb);
  return ret;
}

cublasStatus_t cublasGetMatrix(int rows, int cols, int elemSize, const void *A,
                               int lda, void *B, int ldb) {
  CUBLAS_ENTER_API(cublasGetMatrix);
  cublasStatus_t ret = func(rows, cols, elemSize, A, lda, B, ldb);
  return ret;
}

cublasStatus_t cublasSetVectorAsync(int n, int elemSize, const void *hostPtr,
                                    int incx, void *devicePtr, int incy,
                                    cudaStream_t stream) {
  CUBLAS_ENTER_API(cublasSetVectorAsync);
  cublasStatus_t ret =
      func(n, elemSize, hostPtr, incx, devicePtr, incy, stream);
  return ret;
}

cublasStatus_t cublasGetVectorAsync(int n, int elemSize, const void *devicePtr,
                                    int incx, void *hostPtr, int incy,
                                    cudaStream_t stream) {
  CUBLAS_ENTER_API(cublasGetVectorAsync);
  cublasStatus_t ret =
      func(n, elemSize, devicePtr, incx, hostPtr, incy, stream);
  return ret;
}

cublasStatus_t cublasSetMatrixAsync(int rows, int cols, int elemSize,
                                    const void *A, int lda, void *B, int ldb,
                                    cudaStream_t stream) {
  CUBLAS_ENTER_API(cublasSetMatrixAsync);
  cublasStatus_t ret = func(rows, cols, elemSize, A, lda, B, ldb, stream);
  return ret;
}

cublasStatus_t cublasGetMatrixAsync(int rows, int cols, int elemSize,
                                    const void *A, int lda, void *B, int ldb,
                                    cudaStream_t stream) {
  CUBLAS_ENTER_API(cublasGetMatrixAsync);
  cublasStatus_t ret = func(rows, cols, elemSize, A, lda, B, ldb, stream);
  return ret;
}

void cublasXerbla(const char *srName, int info) {
  CUBLAS_ENTER_API(cublasXerbla);
  func(srName, info);
}

cublasStatus_t cublasNrm2Ex(cublasHandle_t handle, int n, const void *x,
                            cudaDataType xType, int incx, void *result,
                            cudaDataType resultType,
                            cudaDataType executionType) {
  CUBLAS_ENTER_API(cublasNrm2Ex);
  fromCublasInputPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasNrm2Ex, handle);
  cublasStatus_t ret =
      func(handle, n, x, xType, incx, result, resultType, executionType);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSnrm2(cublasHandle_t handle, int n, const float *x,
                           int incx, float *result) {
  CUBLAS_ENTER_API(cublasSnrm2_v2);
  fromCublasInputPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSnrm2_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDnrm2(cublasHandle_t handle, int n, const double *x,
                           int incx, double *result) {
  CUBLAS_ENTER_API(cublasDnrm2_v2);
  fromCublasInputPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDnrm2_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasScnrm2(cublasHandle_t handle, int n, const cuComplex *x,
                            int incx, float *result) {
  CUBLAS_ENTER_API(cublasScnrm2_v2);
  fromCublasInputPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasScnrm2_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDznrm2(cublasHandle_t handle, int n,
                            const cuDoubleComplex *x, int incx,
                            double *result) {
  CUBLAS_ENTER_API(cublasDznrm2_v2);
  fromCublasInputPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDznrm2_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDotEx(cublasHandle_t handle, int n, const void *x,
                           cudaDataType xType, int incx, const void *y,
                           cudaDataType yType, int incy, void *result,
                           cudaDataType resultType,
                           cudaDataType executionType) {
  CUBLAS_ENTER_API(cublasDotEx);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDotEx, handle);
  cublasStatus_t ret = func(handle, n, x, xType, incx, y, yType, incy, result,
                            resultType, executionType);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDotcEx(cublasHandle_t handle, int n, const void *x,
                            cudaDataType xType, int incx, const void *y,
                            cudaDataType yType, int incy, void *result,
                            cudaDataType resultType,
                            cudaDataType executionType) {
  CUBLAS_ENTER_API(cublasDotcEx);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDotcEx, handle);
  cublasStatus_t ret = func(handle, n, x, xType, incx, y, yType, incy, result,
                            resultType, executionType);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSdot(cublasHandle_t handle, int n, const float *x,
                          int incx, const float *y, int incy, float *result) {
  CUBLAS_ENTER_API(cublasSdot_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSdot_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, result);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDdot(cublasHandle_t handle, int n, const double *x,
                          int incx, const double *y, int incy, double *result) {
  CUBLAS_ENTER_API(cublasDdot_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDdot_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, result);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCdotu(cublasHandle_t handle, int n, const cuComplex *x,
                           int incx, const cuComplex *y, int incy,
                           cuComplex *result) {
  CUBLAS_ENTER_API(cublasCdotu_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCdotu_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, result);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCdotc(cublasHandle_t handle, int n, const cuComplex *x,
                           int incx, const cuComplex *y, int incy,
                           cuComplex *result) {
  CUBLAS_ENTER_API(cublasCdotc_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCdotc_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, result);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZdotu(cublasHandle_t handle, int n,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *result) {
  CUBLAS_ENTER_API(cublasZdotu_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZdotu_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, result);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZdotc(cublasHandle_t handle, int n,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *result) {
  CUBLAS_ENTER_API(cublasZdotc_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZdotc_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, result);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasScalEx(cublasHandle_t handle, int n, const void *alpha,
                            cudaDataType alphaType, void *x, cudaDataType xType,
                            int incx, cudaDataType executionType) {
  CUBLAS_ENTER_API(cublasScalEx);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasScalEx, handle);
  cublasStatus_t ret =
      func(handle, n, alpha, alphaType, x, xType, incx, executionType);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSscal(cublasHandle_t handle, int n, const float *alpha,
                           float *x, int incx) {
  CUBLAS_ENTER_API(cublasSscal_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSscal_v2, handle);
  cublasStatus_t ret = func(handle, n, alpha, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDscal(cublasHandle_t handle, int n, const double *alpha,
                           double *x, int incx) {
  CUBLAS_ENTER_API(cublasDscal_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDscal_v2, handle);
  cublasStatus_t ret = func(handle, n, alpha, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCscal(cublasHandle_t handle, int n, const cuComplex *alpha,
                           cuComplex *x, int incx) {
  CUBLAS_ENTER_API(cublasCscal_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCscal_v2, handle);
  cublasStatus_t ret = func(handle, n, alpha, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCsscal(cublasHandle_t handle, int n, const float *alpha,
                            cuComplex *x, int incx) {
  CUBLAS_ENTER_API(cublasCsscal_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCsscal_v2, handle);
  cublasStatus_t ret = func(handle, n, alpha, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZscal(cublasHandle_t handle, int n,
                           const cuDoubleComplex *alpha, cuDoubleComplex *x,
                           int incx) {
  CUBLAS_ENTER_API(cublasZscal_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZscal_v2, handle);
  cublasStatus_t ret = func(handle, n, alpha, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZdscal(cublasHandle_t handle, int n, const double *alpha,
                            cuDoubleComplex *x, int incx) {
  CUBLAS_ENTER_API(cublasZdscal_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZdscal_v2, handle);
  cublasStatus_t ret = func(handle, n, alpha, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasAxpyEx(cublasHandle_t handle, int n, const void *alpha,
                            cudaDataType alphaType, const void *x,
                            cudaDataType xType, int incx, void *y,
                            cudaDataType yType, int incy,
                            cudaDataType executiontype) {
  CUBLAS_ENTER_API(cublasAxpyEx);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasAxpyEx, handle);
  cublasStatus_t ret = func(handle, n, alpha, alphaType, x, xType, incx, y,
                            yType, incy, executiontype);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSaxpy(cublasHandle_t handle, int n, const float *alpha,
                           const float *x, int incx, float *y, int incy) {
  CUBLAS_ENTER_API(cublasSaxpy_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSaxpy_v2, handle);
  cublasStatus_t ret = func(handle, n, alpha, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDaxpy(cublasHandle_t handle, int n, const double *alpha,
                           const double *x, int incx, double *y, int incy) {
  CUBLAS_ENTER_API(cublasDaxpy_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDaxpy_v2, handle);
  cublasStatus_t ret = func(handle, n, alpha, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCaxpy(cublasHandle_t handle, int n, const cuComplex *alpha,
                           const cuComplex *x, int incx, cuComplex *y,
                           int incy) {
  CUBLAS_ENTER_API(cublasCaxpy_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCaxpy_v2, handle);
  cublasStatus_t ret = func(handle, n, alpha, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZaxpy(cublasHandle_t handle, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex *y, int incy) {
  CUBLAS_ENTER_API(cublasZaxpy_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZaxpy_v2, handle);
  cublasStatus_t ret = func(handle, n, alpha, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCopyEx(cublasHandle_t handle, int n, const void *x,
                            cudaDataType xType, int incx, void *y,
                            cudaDataType yType, int incy) {
  CUBLAS_ENTER_API(cublasCopyEx);
  fromCublasInputPointerWithSize(x, n);
  fromCublasOutputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCopyEx, handle);
  cublasStatus_t ret = func(handle, n, x, xType, incx, y, yType, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasScopy(cublasHandle_t handle, int n, const float *x,
                           int incx, float *y, int incy) {
  CUBLAS_ENTER_API(cublasScopy_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasOutputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasScopy_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDcopy(cublasHandle_t handle, int n, const double *x,
                           int incx, double *y, int incy) {
  CUBLAS_ENTER_API(cublasDcopy_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasOutputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDcopy_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCcopy(cublasHandle_t handle, int n, const cuComplex *x,
                           int incx, cuComplex *y, int incy) {
  CUBLAS_ENTER_API(cublasCcopy_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasOutputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCcopy_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZcopy(cublasHandle_t handle, int n,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex *y, int incy) {
  CUBLAS_ENTER_API(cublasZcopy_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasOutputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZcopy_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSswap(cublasHandle_t handle, int n, float *x, int incx,
                           float *y, int incy) {
  CUBLAS_ENTER_API(cublasSswap_v2);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSswap_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDswap(cublasHandle_t handle, int n, double *x, int incx,
                           double *y, int incy) {
  CUBLAS_ENTER_API(cublasDswap_v2);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDswap_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCswap(cublasHandle_t handle, int n, cuComplex *x, int incx,
                           cuComplex *y, int incy) {
  CUBLAS_ENTER_API(cublasCswap_v2);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCswap_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZswap(cublasHandle_t handle, int n, cuDoubleComplex *x,
                           int incx, cuDoubleComplex *y, int incy) {
  CUBLAS_ENTER_API(cublasZswap_v2);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZswap_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSwapEx(cublasHandle_t handle, int n, void *x,
                            cudaDataType xType, int incx, void *y,
                            cudaDataType yType, int incy) {
  CUBLAS_ENTER_API(cublasSwapEx);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSwapEx, handle);
  cublasStatus_t ret = func(handle, n, x, xType, incx, y, yType, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasIsamax(cublasHandle_t handle, int n, const float *x,
                            int incx, int *result) {
  CUBLAS_ENTER_API(cublasIsamax_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasIdamax(cublasHandle_t handle, int n, const double *x,
                            int incx, int *result) {
  CUBLAS_ENTER_API(cublasIdamax_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasIcamax(cublasHandle_t handle, int n, const cuComplex *x,
                            int incx, int *result) {
  CUBLAS_ENTER_API(cublasIcamax_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasIzamax(cublasHandle_t handle, int n,
                            const cuDoubleComplex *x, int incx, int *result) {
  CUBLAS_ENTER_API(cublasIzamax_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasIamaxEx(cublasHandle_t handle, int n, const void *x,
                             cudaDataType xType, int incx, int *result) {
  CUBLAS_ENTER_API(cublasIamaxEx);
  cublasStatus_t ret = func(handle, n, x, xType, incx, result);
  return ret;
}

cublasStatus_t cublasIsamin(cublasHandle_t handle, int n, const float *x,
                            int incx, int *result) {
  CUBLAS_ENTER_API(cublasIsamin_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasIdamin(cublasHandle_t handle, int n, const double *x,
                            int incx, int *result) {
  CUBLAS_ENTER_API(cublasIdamin_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasIcamin(cublasHandle_t handle, int n, const cuComplex *x,
                            int incx, int *result) {
  CUBLAS_ENTER_API(cublasIcamin_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasIzamin(cublasHandle_t handle, int n,
                            const cuDoubleComplex *x, int incx, int *result) {
  CUBLAS_ENTER_API(cublasIzamin_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasIaminEx(cublasHandle_t handle, int n, const void *x,
                             cudaDataType xType, int incx, int *result) {
  CUBLAS_ENTER_API(cublasIaminEx);
  cublasStatus_t ret = func(handle, n, x, xType, incx, result);
  return ret;
}

cublasStatus_t cublasAsumEx(cublasHandle_t handle, int n, const void *x,
                            cudaDataType xType, int incx, void *result,
                            cudaDataType resultType,
                            cudaDataType executiontype) {
  CUBLAS_ENTER_API(cublasAsumEx);
  cublasStatus_t ret =
      func(handle, n, x, xType, incx, result, resultType, executiontype);
  return ret;
}

cublasStatus_t cublasSasum(cublasHandle_t handle, int n, const float *x,
                           int incx, float *result) {
  CUBLAS_ENTER_API(cublasSasum_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasDasum(cublasHandle_t handle, int n, const double *x,
                           int incx, double *result) {
  CUBLAS_ENTER_API(cublasDasum_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasScasum(cublasHandle_t handle, int n, const cuComplex *x,
                            int incx, float *result) {
  CUBLAS_ENTER_API(cublasScasum_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasDzasum(cublasHandle_t handle, int n,
                            const cuDoubleComplex *x, int incx,
                            double *result) {
  CUBLAS_ENTER_API(cublasDzasum_v2);
  cublasStatus_t ret = func(handle, n, x, incx, result);
  return ret;
}

cublasStatus_t cublasSrot(cublasHandle_t handle, int n, float *x, int incx,
                          float *y, int incy, const float *c, const float *s) {
  CUBLAS_ENTER_API(cublasSrot_v2);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSrot_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, c, s);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDrot(cublasHandle_t handle, int n, double *x, int incx,
                          double *y, int incy, const double *c,
                          const double *s) {
  CUBLAS_ENTER_API(cublasDrot_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDrot_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, c, s);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCrot(cublasHandle_t handle, int n, cuComplex *x, int incx,
                          cuComplex *y, int incy, const float *c,
                          const cuComplex *s) {
  CUBLAS_ENTER_API(cublasCrot_v2);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCrot_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, c, s);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCsrot(cublasHandle_t handle, int n, cuComplex *x, int incx,
                           cuComplex *y, int incy, const float *c,
                           const float *s) {
  CUBLAS_ENTER_API(cublasCsrot_v2);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCsrot_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, c, s);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZrot(cublasHandle_t handle, int n, cuDoubleComplex *x,
                          int incx, cuDoubleComplex *y, int incy,
                          const double *c, const cuDoubleComplex *s) {
  CUBLAS_ENTER_API(cublasZrot_v2);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZrot_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, c, s);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZdrot(cublasHandle_t handle, int n, cuDoubleComplex *x,
                           int incx, cuDoubleComplex *y, int incy,
                           const double *c, const double *s) {
  CUBLAS_ENTER_API(cublasZdrot_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZdrot_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, c, s);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasRotEx(cublasHandle_t handle, int n, void *x,
                           cudaDataType xType, int incx, void *y,
                           cudaDataType yType, int incy, const void *c,
                           const void *s, cudaDataType csType,
                           cudaDataType executiontype) {
  CUBLAS_ENTER_API(cublasRotEx);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasRotEx, handle);
  cublasStatus_t ret = func(handle, n, x, xType, incx, y, yType, incy, c, s,
                            csType, executiontype);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSrotg(cublasHandle_t handle, float *a, float *b, float *c,
                           float *s) {
  CUBLAS_ENTER_API(cublasSrotg_v2);
  cublasStatus_t ret = func(handle, a, b, c, s);
  return ret;
}

cublasStatus_t cublasDrotg(cublasHandle_t handle, double *a, double *b,
                           double *c, double *s) {
  CUBLAS_ENTER_API(cublasDrotg_v2);
  cublasStatus_t ret = func(handle, a, b, c, s);
  return ret;
}

cublasStatus_t cublasCrotg(cublasHandle_t handle, cuComplex *a, cuComplex *b,
                           float *c, cuComplex *s) {
  CUBLAS_ENTER_API(cublasCrotg_v2);
  cublasStatus_t ret = func(handle, a, b, c, s);
  return ret;
}

cublasStatus_t cublasZrotg(cublasHandle_t handle, cuDoubleComplex *a,
                           cuDoubleComplex *b, double *c, cuDoubleComplex *s) {
  CUBLAS_ENTER_API(cublasZrotg_v2);
  cublasStatus_t ret = func(handle, a, b, c, s);
  return ret;
}

cublasStatus_t cublasRotgEx(cublasHandle_t handle, void *a, void *b,
                            cudaDataType abType, void *c, void *s,
                            cudaDataType csType, cudaDataType executiontype) {
  CUBLAS_ENTER_API(cublasRotgEx);
  cublasStatus_t ret = func(handle, a, b, abType, c, s, csType, executiontype);
  return ret;
}

cublasStatus_t cublasSrotm(cublasHandle_t handle, int n, float *x, int incx,
                           float *y, int incy, const float *param) {
  CUBLAS_ENTER_API(cublasSrotm_v2);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSrotm_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, param);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDrotm(cublasHandle_t handle, int n, double *x, int incx,
                           double *y, int incy, const double *param) {
  CUBLAS_ENTER_API(cublasDrotm_v2);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDrotm_v2, handle);
  cublasStatus_t ret = func(handle, n, x, incx, y, incy, param);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasRotmEx(cublasHandle_t handle, int n, void *x,
                            cudaDataType xType, int incx, void *y,
                            cudaDataType yType, int incy, const void *param,
                            cudaDataType paramType,
                            cudaDataType executiontype) {
  CUBLAS_ENTER_API(cublasRotmEx);
  fromCublasInoutPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasRotmEx, handle);
  cublasStatus_t ret = func(handle, n, x, xType, incx, y, yType, incy, param,
                            paramType, executiontype);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSrotmg(cublasHandle_t handle, float *d1, float *d2,
                            float *x1, const float *y1, float *param) {
  CUBLAS_ENTER_API(cublasSrotmg_v2);
  cublasStatus_t ret = func(handle, d1, d2, x1, y1, param);
  return ret;
}

cublasStatus_t cublasDrotmg(cublasHandle_t handle, double *d1, double *d2,
                            double *x1, const double *y1, double *param) {
  CUBLAS_ENTER_API(cublasDrotmg_v2);
  cublasStatus_t ret = func(handle, d1, d2, x1, y1, param);
  return ret;
}

cublasStatus_t cublasRotmgEx(cublasHandle_t handle, void *d1,
                             cudaDataType d1Type, void *d2, cudaDataType d2Type,
                             void *x1, cudaDataType x1Type, const void *y1,
                             cudaDataType y1Type, void *param,
                             cudaDataType paramType,
                             cudaDataType executiontype) {
  CUBLAS_ENTER_API(cublasRotmgEx);
  cublasStatus_t ret = func(handle, d1, d1Type, d2, d2Type, x1, x1Type, y1,
                            y1Type, param, paramType, executiontype);
  return ret;
}

cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const float *alpha, const float *A,
                           int lda, const float *x, int incx, const float *beta,
                           float *y, int incy) {
  CUBLAS_ENTER_API(cublasSgemv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(x, (1 + (n - 1) * abs(incx)));
  } else {
    fromCublasInputPointerWithSize(x, (1 + (m - 1) * abs(incx)));
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInoutPointerWithSize(y, (1 + (m - 1) * abs(incy)));
  } else {
    fromCublasInoutPointerWithSize(y, (1 + (n - 1) * abs(incy)));
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSgemv_v2, handle);
  cublasStatus_t ret =
      func(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const double *alpha, const double *A,
                           int lda, const double *x, int incx,
                           const double *beta, double *y, int incy) {
  CUBLAS_ENTER_API(cublasDgemv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(x, (1 + (n - 1) * abs(incx)));
  } else {
    fromCublasInputPointerWithSize(x, (1 + (m - 1) * abs(incx)));
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInoutPointerWithSize(y, (1 + (m - 1) * abs(incy)));
  } else {
    fromCublasInoutPointerWithSize(y, (1 + (n - 1) * abs(incy)));
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDgemv_v2, handle);
  cublasStatus_t ret =
      func(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const cuComplex *alpha,
                           const cuComplex *A, int lda, const cuComplex *x,
                           int incx, const cuComplex *beta, cuComplex *y,
                           int incy) {
  CUBLAS_ENTER_API(cublasCgemv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(x, (1 + (n - 1) * abs(incx)));
  } else {
    fromCublasInputPointerWithSize(x, (1 + (m - 1) * abs(incx)));
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInoutPointerWithSize(y, (1 + (m - 1) * abs(incy)));
  } else {
    fromCublasInoutPointerWithSize(y, (1 + (n - 1) * abs(incy)));
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgemv_v2, handle);
  cublasStatus_t ret =
      func(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta, cuDoubleComplex *y,
                           int incy) {
  CUBLAS_ENTER_API(cublasZgemv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(x, (1 + (n - 1) * abs(incx)));
  } else {
    fromCublasInputPointerWithSize(x, (1 + (m - 1) * abs(incx)));
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInoutPointerWithSize(y, (1 + (m - 1) * abs(incy)));
  } else {
    fromCublasInoutPointerWithSize(y, (1 + (n - 1) * abs(incy)));
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgemv_v2, handle);
  cublasStatus_t ret =
      func(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSgbmv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, int kl, int ku, const float *alpha,
                           const float *A, int lda, const float *x, int incx,
                           const float *beta, float *y, int incy) {
  CUBLAS_ENTER_API(cublasSgbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(x, n);
  } else {
    fromCublasInputPointerWithSize(x, m);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInoutPointerWithSize(y, m);
  } else {
    fromCublasInoutPointerWithSize(y, n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSgbmv_v2, handle);
  cublasStatus_t ret =
      func(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDgbmv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, int kl, int ku, const double *alpha,
                           const double *A, int lda, const double *x, int incx,
                           const double *beta, double *y, int incy) {
  CUBLAS_ENTER_API(cublasDgbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(x, n);
  } else {
    fromCublasInputPointerWithSize(x, m);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInoutPointerWithSize(y, m);
  } else {
    fromCublasInoutPointerWithSize(y, n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDgbmv_v2, handle);
  cublasStatus_t ret =
      func(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgbmv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, int kl, int ku, const cuComplex *alpha,
                           const cuComplex *A, int lda, const cuComplex *x,
                           int incx, const cuComplex *beta, cuComplex *y,
                           int incy) {
  CUBLAS_ENTER_API(cublasCgbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(x, n);
  } else {
    fromCublasInputPointerWithSize(x, m);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInoutPointerWithSize(y, m);
  } else {
    fromCublasInoutPointerWithSize(y, n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgbmv_v2, handle);
  cublasStatus_t ret =
      func(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgbmv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n, int kl, int ku,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta, cuDoubleComplex *y,
                           int incy) {
  CUBLAS_ENTER_API(cublasZgbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(x, n);
  } else {
    fromCublasInputPointerWithSize(x, m);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInoutPointerWithSize(y, m);
  } else {
    fromCublasInoutPointerWithSize(y, n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgbmv_v2, handle);
  cublasStatus_t ret =
      func(handle, trans, m, n, kl, ku, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasStrmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const float *A, int lda, float *x, int incx) {
  CUBLAS_ENTER_API(cublasStrmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasStrmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDtrmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const double *A, int lda, double *x,
                           int incx) {
  CUBLAS_ENTER_API(cublasDtrmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDtrmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCtrmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const cuComplex *A, int lda, cuComplex *x,
                           int incx) {
  CUBLAS_ENTER_API(cublasCtrmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCtrmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZtrmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const cuDoubleComplex *A, int lda,
                           cuDoubleComplex *x, int incx) {
  CUBLAS_ENTER_API(cublasZtrmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZtrmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasStbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, int k, const float *A, int lda, float *x,
                           int incx) {
  CUBLAS_ENTER_API(cublasStbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInputPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasStbmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, k, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDtbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, int k, const double *A, int lda, double *x,
                           int incx) {
  CUBLAS_ENTER_API(cublasDtbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDtbmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, k, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCtbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, int k, const cuComplex *A, int lda,
                           cuComplex *x, int incx) {
  CUBLAS_ENTER_API(cublasCtbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCtbmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, k, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZtbmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, int k, const cuDoubleComplex *A, int lda,
                           cuDoubleComplex *x, int incx) {
  CUBLAS_ENTER_API(cublasZtbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZtbmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, k, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasStpmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const float *AP, float *x, int incx) {
  CUBLAS_ENTER_API(cublasStpmv_v2);
  fromCublasInputPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasStpmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, AP, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDtpmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const double *AP, double *x, int incx) {
  CUBLAS_ENTER_API(cublasDtpmv_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDtpmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, AP, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCtpmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const cuComplex *AP, cuComplex *x, int incx) {
  CUBLAS_ENTER_API(cublasCtpmv_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCtpmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, AP, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZtpmv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const cuDoubleComplex *AP, cuDoubleComplex *x,
                           int incx) {
  CUBLAS_ENTER_API(cublasZtpmv_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZtpmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, AP, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasStrsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const float *A, int lda, float *x, int incx) {
  CUBLAS_ENTER_API(cublasStrsv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasStrsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const double *A, int lda, double *x,
                           int incx) {
  CUBLAS_ENTER_API(cublasDtrsv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDtrsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const cuComplex *A, int lda, cuComplex *x,
                           int incx) {
  CUBLAS_ENTER_API(cublasCtrsv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCtrsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZtrsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const cuDoubleComplex *A, int lda,
                           cuDoubleComplex *x, int incx) {
  CUBLAS_ENTER_API(cublasZtrsv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZtrsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasStpsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const float *AP, float *x, int incx) {
  CUBLAS_ENTER_API(cublasStpsv_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasStpsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, AP, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDtpsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const double *AP, double *x, int incx) {
  CUBLAS_ENTER_API(cublasDtpsv_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDtpsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, AP, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCtpsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const cuComplex *AP, cuComplex *x, int incx) {
  CUBLAS_ENTER_API(cublasCtpsv_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCtpsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, AP, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZtpsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, const cuDoubleComplex *AP, cuDoubleComplex *x,
                           int incx) {
  CUBLAS_ENTER_API(cublasZtpsv_v2);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZtpsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, AP, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasStbsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, int k, const float *A, int lda, float *x,
                           int incx) {
  CUBLAS_ENTER_API(cublasStbsv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasStbsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, k, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDtbsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, int k, const double *A, int lda, double *x,
                           int incx) {
  CUBLAS_ENTER_API(cublasDtbsv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDtbsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, k, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCtbsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, int k, const cuComplex *A, int lda,
                           cuComplex *x, int incx) {
  CUBLAS_ENTER_API(cublasCtbsv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCtbsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, k, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZtbsv(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, cublasDiagType_t diag,
                           int n, int k, const cuDoubleComplex *A, int lda,
                           cuDoubleComplex *x, int incx) {
  CUBLAS_ENTER_API(cublasZtbsv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInoutPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZtbsv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, trans, diag, n, k, A, lda, x, incx);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const float *alpha, const float *A, int lda,
                           const float *x, int incx, const float *beta,
                           float *y, int incy) {
  CUBLAS_ENTER_API(cublasSsymv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSsymv_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const double *alpha, const double *A, int lda,
                           const double *x, int incx, const double *beta,
                           double *y, int incy) {
  CUBLAS_ENTER_API(cublasDsymv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDsymv_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *x, int incx, const cuComplex *beta,
                           cuComplex *y, int incy) {
  CUBLAS_ENTER_API(cublasCsymv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCsymv_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZsymv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta, cuDoubleComplex *y,
                           int incy) {
  CUBLAS_ENTER_API(cublasZsymv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZsymv_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasChemv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *x, int incx, const cuComplex *beta,
                           cuComplex *y, int incy) {
  CUBLAS_ENTER_API(cublasChemv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasChemv_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZhemv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta, cuDoubleComplex *y,
                           int incy) {
  CUBLAS_ENTER_API(cublasZhemv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZhemv_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, n, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           int k, const float *alpha, const float *A, int lda,
                           const float *x, int incx, const float *beta,
                           float *y, int incy) {
  CUBLAS_ENTER_API(cublasSsbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSsbmv_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDsbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           int k, const double *alpha, const double *A, int lda,
                           const double *x, int incx, const double *beta,
                           double *y, int incy) {
  CUBLAS_ENTER_API(cublasDsbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDsbmv_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasChbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           int k, const cuComplex *alpha, const cuComplex *A,
                           int lda, const cuComplex *x, int incx,
                           const cuComplex *beta, cuComplex *y, int incy) {
  CUBLAS_ENTER_API(cublasChbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasChbmv_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZhbmv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           int k, const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *beta, cuDoubleComplex *y,
                           int incy) {
  CUBLAS_ENTER_API(cublasZhbmv_v2);
  fromCublasInputPointerWithSize(A, lda * n);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZhbmv_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, n, k, alpha, A, lda, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const float *alpha, const float *AP, const float *x,
                           int incx, const float *beta, float *y, int incy) {
  CUBLAS_ENTER_API(cublasSspmv_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSspmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDspmv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const double *alpha, const double *AP,
                           const double *x, int incx, const double *beta,
                           double *y, int incy) {
  CUBLAS_ENTER_API(cublasDspmv_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDspmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasChpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuComplex *alpha, const cuComplex *AP,
                           const cuComplex *x, int incx, const cuComplex *beta,
                           cuComplex *y, int incy) {
  CUBLAS_ENTER_API(cublasChpmv_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasChpmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZhpmv(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *AP, const cuDoubleComplex *x,
                           int incx, const cuDoubleComplex *beta,
                           cuDoubleComplex *y, int incy) {
  CUBLAS_ENTER_API(cublasZhpmv_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZhpmv_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, AP, x, incx, beta, y, incy);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSger(cublasHandle_t handle, int m, int n,
                          const float *alpha, const float *x, int incx,
                          const float *y, int incy, float *A, int lda) {
  CUBLAS_ENTER_API(cublasSger_v2);
  fromCublasInputPointerWithSize(x, m);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSger_v2, handle);
  cublasStatus_t ret = func(handle, m, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDger(cublasHandle_t handle, int m, int n,
                          const double *alpha, const double *x, int incx,
                          const double *y, int incy, double *A, int lda) {
  CUBLAS_ENTER_API(cublasDger_v2);
  fromCublasInputPointerWithSize(x, m);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDger_v2, handle);
  cublasStatus_t ret = func(handle, m, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgeru(cublasHandle_t handle, int m, int n,
                           const cuComplex *alpha, const cuComplex *x, int incx,
                           const cuComplex *y, int incy, cuComplex *A,
                           int lda) {
  CUBLAS_ENTER_API(cublasCgeru_v2);
  fromCublasInputPointerWithSize(x, m);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgeru_v2, handle);
  cublasStatus_t ret = func(handle, m, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgerc(cublasHandle_t handle, int m, int n,
                           const cuComplex *alpha, const cuComplex *x, int incx,
                           const cuComplex *y, int incy, cuComplex *A,
                           int lda) {
  CUBLAS_ENTER_API(cublasCgerc_v2);
  fromCublasInputPointerWithSize(x, m);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgerc_v2, handle);
  cublasStatus_t ret = func(handle, m, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgeru(cublasHandle_t handle, int m, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *A, int lda) {
  CUBLAS_ENTER_API(cublasZgeru_v2);
  fromCublasInputPointerWithSize(x, m);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgeru_v2, handle);
  cublasStatus_t ret = func(handle, m, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgerc(cublasHandle_t handle, int m, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *A, int lda) {
  CUBLAS_ENTER_API(cublasZgerc_v2);
  fromCublasInputPointerWithSize(x, m);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgerc_v2, handle);
  cublasStatus_t ret = func(handle, m, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                          const float *alpha, const float *x, int incx,
                          float *A, int lda) {
  CUBLAS_ENTER_API(cublasSsyr_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSsyr_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                          const double *alpha, const double *x, int incx,
                          double *A, int lda) {
  CUBLAS_ENTER_API(cublasDsyr_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDsyr_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                          const cuComplex *alpha, const cuComplex *x, int incx,
                          cuComplex *A, int lda) {
  CUBLAS_ENTER_API(cublasCsyr_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCsyr_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZsyr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                          const cuDoubleComplex *alpha,
                          const cuDoubleComplex *x, int incx,
                          cuDoubleComplex *A, int lda) {
  CUBLAS_ENTER_API(cublasZsyr_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZsyr_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCher(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                          const float *alpha, const cuComplex *x, int incx,
                          cuComplex *A, int lda) {
  CUBLAS_ENTER_API(cublasCher_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCher_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZher(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                          const double *alpha, const cuDoubleComplex *x,
                          int incx, cuDoubleComplex *A, int lda) {
  CUBLAS_ENTER_API(cublasZher_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZher_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSspr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                          const float *alpha, const float *x, int incx,
                          float *AP) {
  CUBLAS_ENTER_API(cublasSspr_v2);
  fromCublasInputPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSspr_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDspr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                          const double *alpha, const double *x, int incx,
                          double *AP) {
  CUBLAS_ENTER_API(cublasDspr_v2);
  fromCublasInputPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDspr_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasChpr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                          const float *alpha, const cuComplex *x, int incx,
                          cuComplex *AP) {
  CUBLAS_ENTER_API(cublasChpr_v2);
  fromCublasInputPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasChpr_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZhpr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                          const double *alpha, const cuDoubleComplex *x,
                          int incx, cuDoubleComplex *AP) {
  CUBLAS_ENTER_API(cublasZhpr_v2);
  fromCublasInputPointerWithSize(x, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZhpr_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const float *alpha, const float *x, int incx,
                           const float *y, int incy, float *A, int lda) {
  CUBLAS_ENTER_API(cublasSsyr2_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSsyr2_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const double *alpha, const double *x, int incx,
                           const double *y, int incy, double *A, int lda) {
  CUBLAS_ENTER_API(cublasDsyr2_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDsyr2_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuComplex *alpha, const cuComplex *x, int incx,
                           const cuComplex *y, int incy, cuComplex *A,
                           int lda) {
  CUBLAS_ENTER_API(cublasCsyr2_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCsyr2_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZsyr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *A, int lda) {
  CUBLAS_ENTER_API(cublasZsyr2_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZsyr2_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCher2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuComplex *alpha, const cuComplex *x, int incx,
                           const cuComplex *y, int incy, cuComplex *A,
                           int lda) {
  CUBLAS_ENTER_API(cublasCher2_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCher2_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZher2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *A, int lda) {
  CUBLAS_ENTER_API(cublasZher2_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  fromCublasInoutPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZher2_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, y, incy, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const float *alpha, const float *x, int incx,
                           const float *y, int incy, float *AP) {
  CUBLAS_ENTER_API(cublasSspr2_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSspr2_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, y, incy, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDspr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const double *alpha, const double *x, int incx,
                           const double *y, int incy, double *AP) {
  CUBLAS_ENTER_API(cublasDspr2_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDspr2_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, y, incy, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasChpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuComplex *alpha, const cuComplex *x, int incx,
                           const cuComplex *y, int incy, cuComplex *AP) {
  CUBLAS_ENTER_API(cublasChpr2_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasChpr2_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, y, incy, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZhpr2(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *x, int incx,
                           const cuDoubleComplex *y, int incy,
                           cuDoubleComplex *AP) {
  CUBLAS_ENTER_API(cublasZhpr2_v2);
  fromCublasInputPointerWithSize(x, n);
  fromCublasInputPointerWithSize(y, n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZhpr2_v2, handle);
  cublasStatus_t ret = func(handle, uplo, n, alpha, x, incx, y, incy, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {
  CUBLAS_ENTER_API(cublasSgemm_v2);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSgemm_v2, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, A, lda, B,
                            ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb, const double *beta,
                           double *C, int ldc) {
  CUBLAS_ENTER_API(cublasDgemm_v2);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDgemm_v2, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, A, lda, B,
                            ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *B, int ldb, const cuComplex *beta,
                           cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasCgemm_v2);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgemm_v2, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, A, lda, B,
                            ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgemm3m(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const cuComplex *alpha, const cuComplex *A,
                             int lda, const cuComplex *B, int ldb,
                             const cuComplex *beta, cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasCgemm3m);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgemm3m, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, A, lda, B,
                            ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgemm3mEx(cublasHandle_t handle, cublasOperation_t transa,
                               cublasOperation_t transb, int m, int n, int k,
                               const cuComplex *alpha, const void *A,
                               cudaDataType Atype, int lda, const void *B,
                               cudaDataType Btype, int ldb,
                               const cuComplex *beta, void *C,
                               cudaDataType Ctype, int ldc) {
  CUBLAS_ENTER_API(cublasCgemm3mEx);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgemm3mEx, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, A, Atype,
                            lda, B, Btype, ldb, beta, C, Ctype, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *B, int ldb,
                           const cuDoubleComplex *beta, cuDoubleComplex *C,
                           int ldc) {
  CUBLAS_ENTER_API(cublasZgemm_v2);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgemm_v2, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, A, lda, B,
                            ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgemm3m(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const cuDoubleComplex *alpha,
                             const cuDoubleComplex *A, int lda,
                             const cuDoubleComplex *B, int ldb,
                             const cuDoubleComplex *beta, cuDoubleComplex *C,
                             int ldc) {
  CUBLAS_ENTER_API(cublasZgemm3m);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgemm3m, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, A, lda, B,
                            ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasHgemm(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n, int k,
                           const __half *alpha, const __half *A, int lda,
                           const __half *B, int ldb, const __half *beta,
                           __half *C, int ldc) {
  CUBLAS_ENTER_API(cublasHgemm);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasHgemm, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, A, lda, B,
                            ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const float *alpha, const void *A,
                             cudaDataType Atype, int lda, const void *B,
                             cudaDataType Btype, int ldb, const float *beta,
                             void *C, cudaDataType Ctype, int ldc) {
  CUBLAS_ENTER_API(cublasSgemmEx);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSgemmEx, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, A, Atype,
                            lda, B, Btype, ldb, beta, C, Ctype, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa,
                            cublasOperation_t transb, int m, int n, int k,
                            const void *alpha, const void *A,
                            cudaDataType Atype, int lda, const void *B,
                            cudaDataType Btype, int ldb, const void *beta,
                            void *C, cudaDataType Ctype, int ldc,
                            cudaDataType computeType, cublasGemmAlgo_t algo) {
  CUBLAS_ENTER_API(cublasGemmEx);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasGemmEx, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb,
           beta, C, Ctype, ldc, computeType, algo);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgemmEx(cublasHandle_t handle, cublasOperation_t transa,
                             cublasOperation_t transb, int m, int n, int k,
                             const cuComplex *alpha, const void *A,
                             cudaDataType Atype, int lda, const void *B,
                             cudaDataType Btype, int ldb, const cuComplex *beta,
                             void *C, cudaDataType Ctype, int ldc) {
  CUBLAS_ENTER_API(cublasCgemmEx);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgemmEx, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, A, Atype,
                            lda, B, Btype, ldb, beta, C, Ctype, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasUint8gemmBias(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    cublasOperation_t transc, int m, int n, int k, const unsigned char *A,
    int A_bias, int lda, const unsigned char *B, int B_bias, int ldb,
    unsigned char *C, int C_bias, int ldc, int C_mult, int C_shift) {
  CUBLAS_ENTER_API(cublasUint8gemmBias);
  cublasStatus_t ret =
      func(handle, transa, transb, transc, m, n, k, A, A_bias, lda, B, B_bias,
           ldb, C, C_bias, ldc, C_mult, C_shift);
  return ret;
}

cublasStatus_t cublasSsyrk(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, int n, int k,
                           const float *alpha, const float *A, int lda,
                           const float *beta, float *C, int ldc) {
  CUBLAS_ENTER_API(cublasSsyrk_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSsyrk_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDsyrk(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, int n, int k,
                           const double *alpha, const double *A, int lda,
                           const double *beta, double *C, int ldc) {
  CUBLAS_ENTER_API(cublasDsyrk_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDsyrk_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCsyrk(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, int n, int k,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *beta, cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasCsyrk_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCsyrk_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZsyrk(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, int n, int k,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *beta, cuDoubleComplex *C,
                           int ldc) {
  CUBLAS_ENTER_API(cublasZsyrk_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZsyrk_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCsyrkEx(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, int n, int k,
                             const cuComplex *alpha, const void *A,
                             cudaDataType Atype, int lda, const cuComplex *beta,
                             void *C, cudaDataType Ctype, int ldc) {
  CUBLAS_ENTER_API(cublasCsyrkEx);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCsyrkEx, handle);
  cublasStatus_t ret = func(handle, uplo, trans, n, k, alpha, A, Atype, lda,
                            beta, C, Ctype, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCsyrk3mEx(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               const cuComplex *alpha, const void *A,
                               cudaDataType Atype, int lda,
                               const cuComplex *beta, void *C,
                               cudaDataType Ctype, int ldc) {
  CUBLAS_ENTER_API(cublasCsyrk3mEx);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCsyrk3mEx, handle);
  cublasStatus_t ret = func(handle, uplo, trans, n, k, alpha, A, Atype, lda,
                            beta, C, Ctype, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCherk(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, int n, int k,
                           const float *alpha, const cuComplex *A, int lda,
                           const float *beta, cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasCherk_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCherk_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZherk(cublasHandle_t handle, cublasFillMode_t uplo,
                           cublasOperation_t trans, int n, int k,
                           const double *alpha, const cuDoubleComplex *A,
                           int lda, const double *beta, cuDoubleComplex *C,
                           int ldc) {
  CUBLAS_ENTER_API(cublasZherk_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZherk_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCherkEx(cublasHandle_t handle, cublasFillMode_t uplo,
                             cublasOperation_t trans, int n, int k,
                             const float *alpha, const void *A,
                             cudaDataType Atype, int lda, const float *beta,
                             void *C, cudaDataType Ctype, int ldc) {
  CUBLAS_ENTER_API(cublasCherkEx);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCherkEx, handle);
  cublasStatus_t ret = func(handle, uplo, trans, n, k, alpha, A, Atype, lda,
                            beta, C, Ctype, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCherk3mEx(cublasHandle_t handle, cublasFillMode_t uplo,
                               cublasOperation_t trans, int n, int k,
                               const float *alpha, const void *A,
                               cudaDataType Atype, int lda, const float *beta,
                               void *C, cudaDataType Ctype, int ldc) {
  CUBLAS_ENTER_API(cublasCherk3mEx);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCherk3mEx, handle);
  cublasStatus_t ret = func(handle, uplo, trans, n, k, alpha, A, Atype, lda,
                            beta, C, Ctype, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSsyr2k(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const float *alpha, const float *A, int lda,
                            const float *B, int ldb, const float *beta,
                            float *C, int ldc) {
  CUBLAS_ENTER_API(cublasSsyr2k_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * k);
  } else {
    fromCublasInputPointerWithSize(B, ldb * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSsyr2k_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDsyr2k(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const double *alpha, const double *A, int lda,
                            const double *B, int ldb, const double *beta,
                            double *C, int ldc) {
  CUBLAS_ENTER_API(cublasDsyr2k_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * k);
  } else {
    fromCublasInputPointerWithSize(B, ldb * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDsyr2k_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCsyr2k(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuComplex *alpha, const cuComplex *A, int lda,
                            const cuComplex *B, int ldb, const cuComplex *beta,
                            cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasCsyr2k_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * k);
  } else {
    fromCublasInputPointerWithSize(B, ldb * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCsyr2k_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZsyr2k(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb,
                            const cuDoubleComplex *beta, cuDoubleComplex *C,
                            int ldc) {
  CUBLAS_ENTER_API(cublasZsyr2k_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * k);
  } else {
    fromCublasInputPointerWithSize(B, ldb * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZsyr2k_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCher2k(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuComplex *alpha, const cuComplex *A, int lda,
                            const cuComplex *B, int ldb, const float *beta,
                            cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasCher2k_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * k);
  } else {
    fromCublasInputPointerWithSize(B, ldb * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCher2k_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZher2k(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb,
                            const double *beta, cuDoubleComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasZher2k_v2);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * k);
  } else {
    fromCublasInputPointerWithSize(B, ldb * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZher2k_v2, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const float *alpha, const float *A, int lda,
                            const float *B, int ldb, const float *beta,
                            float *C, int ldc) {
  CUBLAS_ENTER_API(cublasSsyrkx);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * k);
  } else {
    fromCublasInputPointerWithSize(B, ldb * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSsyrkx, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const double *alpha, const double *A, int lda,
                            const double *B, int ldb, const double *beta,
                            double *C, int ldc) {
  CUBLAS_ENTER_API(cublasDsyrkx);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * k);
  } else {
    fromCublasInputPointerWithSize(B, ldb * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDsyrkx, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuComplex *alpha, const cuComplex *A, int lda,
                            const cuComplex *B, int ldb, const cuComplex *beta,
                            cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasCsyrkx);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCsyrkx, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZsyrkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb,
                            const cuDoubleComplex *beta, cuDoubleComplex *C,
                            int ldc) {
  CUBLAS_ENTER_API(cublasZsyrkx);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * k);
  } else {
    fromCublasInputPointerWithSize(B, ldb * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZsyrkx, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCherkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuComplex *alpha, const cuComplex *A, int lda,
                            const cuComplex *B, int ldb, const float *beta,
                            cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasCherkx);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCherkx, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZherkx(cublasHandle_t handle, cublasFillMode_t uplo,
                            cublasOperation_t trans, int n, int k,
                            const cuDoubleComplex *alpha,
                            const cuDoubleComplex *A, int lda,
                            const cuDoubleComplex *B, int ldb,
                            const double *beta, cuDoubleComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasZherkx);
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  if (trans == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * k);
  } else {
    fromCublasInputPointerWithSize(B, ldb * n);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZherkx, handle);
  cublasStatus_t ret =
      func(handle, uplo, trans, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSsymm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, int m, int n,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, const float *beta, float *C,
                           int ldc) {
  CUBLAS_ENTER_API(cublasSsymm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInputPointerWithSize(B, ldb * n);
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSsymm_v2, handle);
  cublasStatus_t ret =
      func(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDsymm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, int m, int n,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb, const double *beta,
                           double *C, int ldc) {
  CUBLAS_ENTER_API(cublasDsymm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInputPointerWithSize(B, ldb * n);
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDsymm_v2, handle);
  cublasStatus_t ret =
      func(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCsymm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, int m, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *B, int ldb, const cuComplex *beta,
                           cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasCsymm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInputPointerWithSize(B, ldb * n);
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCsymm_v2, handle);
  cublasStatus_t ret =
      func(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasZsymm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
            int m, int n, const cuDoubleComplex *alpha,
            const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
            int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasZsymm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInputPointerWithSize(B, ldb * n);
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZsymm_v2, handle);
  cublasStatus_t ret =
      func(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasChemm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, int m, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *B, int ldb, const cuComplex *beta,
                           cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasChemm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInputPointerWithSize(B, ldb * n);
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasChemm_v2, handle);
  cublasStatus_t ret =
      func(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasZhemm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
            int m, int n, const cuDoubleComplex *alpha,
            const cuDoubleComplex *A, int lda, const cuDoubleComplex *B,
            int ldb, const cuDoubleComplex *beta, cuDoubleComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasZhemm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInputPointerWithSize(B, ldb * n);
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZhemm_v2, handle);
  cublasStatus_t ret =
      func(handle, side, uplo, m, n, alpha, A, lda, B, ldb, beta, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasStrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n,
                           const float *alpha, const float *A, int lda,
                           float *B, int ldb) {
  CUBLAS_ENTER_API(cublasStrsm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(B, ldb * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasStrsm_v2, handle);
  cublasStatus_t ret =
      func(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n,
                           const double *alpha, const double *A, int lda,
                           double *B, int ldb) {
  CUBLAS_ENTER_API(cublasDtrsm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(B, ldb * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDtrsm_v2, handle);
  cublasStatus_t ret =
      func(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           cuComplex *B, int ldb) {
  CUBLAS_ENTER_API(cublasCtrsm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(B, ldb * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCtrsm_v2, handle);
  cublasStatus_t ret =
      func(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZtrsm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           cuDoubleComplex *B, int ldb) {
  CUBLAS_ENTER_API(cublasZtrsm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInoutPointerWithSize(B, ldb * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZtrsm_v2, handle);
  cublasStatus_t ret =
      func(handle, side, uplo, trans, diag, m, n, alpha, A, lda, B, ldb);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasStrmm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n,
                           const float *alpha, const float *A, int lda,
                           const float *B, int ldb, float *C, int ldc) {
  CUBLAS_ENTER_API(cublasStrmm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInputPointerWithSize(B, ldb * n);
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasStrmm_v2, handle);
  cublasStatus_t ret = func(handle, side, uplo, trans, diag, m, n, alpha, A,
                            lda, B, ldb, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDtrmm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n,
                           const double *alpha, const double *A, int lda,
                           const double *B, int ldb, double *C, int ldc) {
  CUBLAS_ENTER_API(cublasDtrmm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInputPointerWithSize(B, ldb * n);
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDtrmm_v2, handle);
  cublasStatus_t ret = func(handle, side, uplo, trans, diag, m, n, alpha, A,
                            lda, B, ldb, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCtrmm(cublasHandle_t handle, cublasSideMode_t side,
                           cublasFillMode_t uplo, cublasOperation_t trans,
                           cublasDiagType_t diag, int m, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *B, int ldb, cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasCtrmm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInputPointerWithSize(B, ldb * n);
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCtrmm_v2, handle);
  cublasStatus_t ret = func(handle, side, uplo, trans, diag, m, n, alpha, A,
                            lda, B, ldb, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasZtrmm(cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
            cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
            const cuDoubleComplex *alpha, const cuDoubleComplex *A, int lda,
            const cuDoubleComplex *B, int ldb, cuDoubleComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasZtrmm_v2);
  if (side == CUBLAS_SIDE_LEFT) {
    fromCublasInputPointerWithSize(A, lda * m);
  } else {
    fromCublasInputPointerWithSize(A, lda * n);
  }
  fromCublasInputPointerWithSize(B, ldb * n);
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZtrmm_v2, handle);
  cublasStatus_t ret = func(handle, side, uplo, trans, diag, m, n, alpha, A,
                            lda, B, ldb, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasHgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const __half *alpha, const __half *const Aarray[], int lda,
                   const __half *const Barray[], int ldb, const __half *beta,
                   __half *const Carray[], int ldc, int batchCount) {
  CUBLAS_ENTER_API(cublasHgemmBatched);
  for (int i = 0; i < lda; i++) {
    if (transa == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Aarray[i], lda * k);
    } else {
      fromCublasInputPointerWithSize(Aarray[i], lda * m);
    }
  }
  for (int i = 0; i < ldb; i++) {
    if (transb == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Barray[i], ldb * n);
    } else {
      fromCublasInputPointerWithSize(Barray[i], ldb * k);
    }
  }
  for (int i = 0; i < ldc; i++) {
    fromCublasInoutPointerWithSize(Carray[i], ldc * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasHgemmBatched, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasSgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const float *alpha, const float *const Aarray[], int lda,
                   const float *const Barray[], int ldb, const float *beta,
                   float *const Carray[], int ldc, int batchCount) {
  CUBLAS_ENTER_API(cublasSgemmBatched);
  for (int i = 0; i < lda; i++) {
    if (transa == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Aarray[i], lda * k);
    } else {
      fromCublasInputPointerWithSize(Aarray[i], lda * m);
    }
  }
  for (int i = 0; i < ldb; i++) {
    if (transb == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Barray[i], ldb * n);
    } else {
      fromCublasInputPointerWithSize(Barray[i], ldb * k);
    }
  }
  for (int i = 0; i < ldc; i++) {
    fromCublasInoutPointerWithSize(Carray[i], ldc * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSgemmBatched, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasDgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const double *alpha, const double *const Aarray[], int lda,
                   const double *const Barray[], int ldb, const double *beta,
                   double *const Carray[], int ldc, int batchCount) {
  CUBLAS_ENTER_API(cublasDgemmBatched);
  for (int i = 0; i < lda; i++) {
    if (transa == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Aarray[i], lda * k);
    } else {
      fromCublasInputPointerWithSize(Aarray[i], lda * m);
    }
  }
  for (int i = 0; i < ldb; i++) {
    if (transb == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Barray[i], ldb * n);
    } else {
      fromCublasInputPointerWithSize(Barray[i], ldb * k);
    }
  }
  for (int i = 0; i < ldc; i++) {
    fromCublasInoutPointerWithSize(Carray[i], ldc * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDgemmBatched, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasCgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const cuComplex *alpha, const cuComplex *const Aarray[],
                   int lda, const cuComplex *const Barray[], int ldb,
                   const cuComplex *beta, cuComplex *const Carray[], int ldc,
                   int batchCount) {
  CUBLAS_ENTER_API(cublasCgemmBatched);
  for (int i = 0; i < lda; i++) {
    if (transa == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Aarray[i], lda * k);
    } else {
      fromCublasInputPointerWithSize(Aarray[i], lda * m);
    }
  }
  for (int i = 0; i < ldb; i++) {
    if (transb == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Barray[i], ldb * n);
    } else {
      fromCublasInputPointerWithSize(Barray[i], ldb * k);
    }
  }
  for (int i = 0; i < ldc; i++) {
    fromCublasInoutPointerWithSize(Carray[i], ldc * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgemmBatched, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasCgemm3mBatched(cublasHandle_t handle, cublasOperation_t transa,
                     cublasOperation_t transb, int m, int n, int k,
                     const cuComplex *alpha, const cuComplex *const Aarray[],
                     int lda, const cuComplex *const Barray[], int ldb,
                     const cuComplex *beta, cuComplex *const Carray[], int ldc,
                     int batchCount) {
  CUBLAS_ENTER_API(cublasCgemm3mBatched);
  for (int i = 0; i < lda; i++) {
    if (transa == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Aarray[i], lda * k);
    } else {
      fromCublasInputPointerWithSize(Aarray[i], lda * m);
    }
  }
  for (int i = 0; i < ldb; i++) {
    if (transb == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Barray[i], ldb * n);
    } else {
      fromCublasInputPointerWithSize(Barray[i], ldb * k);
    }
  }
  for (int i = 0; i < ldc; i++) {
    fromCublasInoutPointerWithSize(Carray[i], ldc * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgemm3mBatched, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuDoubleComplex *alpha,
    const cuDoubleComplex *const Aarray[], int lda,
    const cuDoubleComplex *const Barray[], int ldb, const cuDoubleComplex *beta,
    cuDoubleComplex *const Carray[], int ldc, int batchCount) {
  CUBLAS_ENTER_API(cublasZgemmBatched);
  for (int i = 0; i < lda; i++) {
    if (transa == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Aarray[i], lda * k);
    } else {
      fromCublasInputPointerWithSize(Aarray[i], lda * m);
    }
  }
  for (int i = 0; i < ldb; i++) {
    if (transb == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Barray[i], ldb * n);
    } else {
      fromCublasInputPointerWithSize(Barray[i], ldb * k);
    }
  }
  for (int i = 0; i < ldc; i++) {
    fromCublasInoutPointerWithSize(Carray[i], ldc * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgemmBatched, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasGemmBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha, const void *const Aarray[],
    cudaDataType Atype, int lda, const void *const Barray[], cudaDataType Btype,
    int ldb, const void *beta, void *const Carray[], cudaDataType Ctype,
    int ldc, int batchCount, cudaDataType computeType, cublasGemmAlgo_t algo) {
  CUBLAS_ENTER_API(cublasGemmBatchedEx);
  for (int i = 0; i < lda; i++) {
    if (transa == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Aarray[i], lda * k);
    } else {
      fromCublasInputPointerWithSize(Aarray[i], lda * m);
    }
  }
  for (int i = 0; i < ldb; i++) {
    if (transb == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(Barray[i], ldb * n);
    } else {
      fromCublasInputPointerWithSize(Barray[i], ldb * k);
    }
  }
  for (int i = 0; i < ldc; i++) {
    fromCublasInoutPointerWithSize(Carray[i], ldc * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasGemmBatchedEx, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, k, alpha, Aarray, Atype, lda, Barray,
           Btype, ldb, beta, Carray, Ctype, ldc, batchCount, computeType, algo);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasGemmStridedBatchedEx(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const void *alpha, const void *A, cudaDataType Atype,
    int lda, long long int strideA, const void *B, cudaDataType Btype, int ldb,
    long long int strideB, const void *beta, void *C, cudaDataType Ctype,
    int ldc, long long int strideC, int batchCount, cudaDataType computeType,
    cublasGemmAlgo_t algo) {
  CUBLAS_ENTER_API(cublasGemmStridedBatchedEx);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasGemmStridedBatchedEx, handle);
  cublasStatus_t ret = func(handle, transa, transb, m, n, k, alpha, A, Atype,
                            lda, strideA, B, Btype, ldb, strideB, beta, C,
                            Ctype, ldc, strideC, batchCount, computeType, algo);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasSgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const float *alpha, const float *A, int lda,
                          long long int strideA, const float *B, int ldb,
                          long long int strideB, const float *beta, float *C,
                          int ldc, long long int strideC, int batchCount) {
  CUBLAS_ENTER_API(cublasSgemmStridedBatched);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSgemmStridedBatched, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
           strideB, beta, C, ldc, strideC, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasDgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const double *alpha, const double *A, int lda,
                          long long int strideA, const double *B, int ldb,
                          long long int strideB, const double *beta, double *C,
                          int ldc, long long int strideC, int batchCount) {
  CUBLAS_ENTER_API(cublasDgemmStridedBatched);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDgemmStridedBatched, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
           strideB, beta, C, ldc, strideC, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda,
    long long int strideA, const cuComplex *B, int ldb, long long int strideB,
    const cuComplex *beta, cuComplex *C, int ldc, long long int strideC,
    int batchCount) {
  CUBLAS_ENTER_API(cublasCgemmStridedBatched);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgemmStridedBatched, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
           strideB, beta, C, ldc, strideC, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgemm3mStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuComplex *alpha, const cuComplex *A, int lda,
    long long int strideA, const cuComplex *B, int ldb, long long int strideB,
    const cuComplex *beta, cuComplex *C, int ldc, long long int strideC,
    int batchCount) {
  CUBLAS_ENTER_API(cublasCgemm3mStridedBatched);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgemm3mStridedBatched, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
           strideB, beta, C, ldc, strideC, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgemmStridedBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const cuDoubleComplex *alpha, const cuDoubleComplex *A,
    int lda, long long int strideA, const cuDoubleComplex *B, int ldb,
    long long int strideB, const cuDoubleComplex *beta, cuDoubleComplex *C,
    int ldc, long long int strideC, int batchCount) {
  CUBLAS_ENTER_API(cublasZgemmStridedBatched);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgemmStridedBatched, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
           strideB, beta, C, ldc, strideC, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasHgemmStridedBatched(cublasHandle_t handle, cublasOperation_t transa,
                          cublasOperation_t transb, int m, int n, int k,
                          const __half *alpha, const __half *A, int lda,
                          long long int strideA, const __half *B, int ldb,
                          long long int strideB, const __half *beta, __half *C,
                          int ldc, long long int strideC, int batchCount) {
  CUBLAS_ENTER_API(cublasHgemmStridedBatched);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * k);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * k);
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasHgemmStridedBatched, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, k, alpha, A, lda, strideA, B, ldb,
           strideB, beta, C, ldc, strideC, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const float *alpha, const float *A, int lda,
                           const float *beta, const float *B, int ldb, float *C,
                           int ldc) {
  CUBLAS_ENTER_API(cublasSgeam);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * n);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * m);
  }
  fromCublasOutputPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSgeam, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const double *alpha, const double *A, int lda,
                           const double *beta, const double *B, int ldb,
                           double *C, int ldc) {
  CUBLAS_ENTER_API(cublasDgeam);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * n);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * m);
  }
  fromCublasOutputPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDgeam, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const cuComplex *alpha, const cuComplex *A, int lda,
                           const cuComplex *beta, const cuComplex *B, int ldb,
                           cuComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasCgeam);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * n);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * m);
  }
  fromCublasOutputPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgeam, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgeam(cublasHandle_t handle, cublasOperation_t transa,
                           cublasOperation_t transb, int m, int n,
                           const cuDoubleComplex *alpha,
                           const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *beta,
                           const cuDoubleComplex *B, int ldb,
                           cuDoubleComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasZgeam);
  if (transa == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(A, lda * n);
  } else {
    fromCublasInputPointerWithSize(A, lda * m);
  }
  if (transb == CUBLAS_OP_N) {
    fromCublasInputPointerWithSize(B, ldb * n);
  } else {
    fromCublasInputPointerWithSize(B, ldb * m);
  }
  fromCublasOutputPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgeam, handle);
  cublasStatus_t ret =
      func(handle, transa, transb, m, n, alpha, A, lda, beta, B, ldb, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSgetrfBatched(cublasHandle_t handle, int n,
                                   float *const A[], int lda, int *P, int *info,
                                   int batchSize) {
  CUBLAS_ENTER_API(cublasSgetrfBatched);
  cublasStatus_t ret = func(handle, n, A, lda, P, info, batchSize);
  return ret;
}

cublasStatus_t cublasDgetrfBatched(cublasHandle_t handle, int n,
                                   double *const A[], int lda, int *P,
                                   int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasDgetrfBatched);
  cublasStatus_t ret = func(handle, n, A, lda, P, info, batchSize);
  return ret;
}

cublasStatus_t cublasCgetrfBatched(cublasHandle_t handle, int n,
                                   cuComplex *const A[], int lda, int *P,
                                   int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasCgetrfBatched);
  cublasStatus_t ret = func(handle, n, A, lda, P, info, batchSize);
  return ret;
}

cublasStatus_t cublasZgetrfBatched(cublasHandle_t handle, int n,
                                   cuDoubleComplex *const A[], int lda, int *P,
                                   int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasZgetrfBatched);
  cublasStatus_t ret = func(handle, n, A, lda, P, info, batchSize);
  return ret;
}

cublasStatus_t cublasSgetriBatched(cublasHandle_t handle, int n,
                                   const float *const A[], int lda,
                                   const int *P, float *const C[], int ldc,
                                   int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasSgetriBatched);
  cublasStatus_t ret = func(handle, n, A, lda, P, C, ldc, info, batchSize);
  return ret;
}

cublasStatus_t cublasDgetriBatched(cublasHandle_t handle, int n,
                                   const double *const A[], int lda,
                                   const int *P, double *const C[], int ldc,
                                   int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasDgetriBatched);
  cublasStatus_t ret = func(handle, n, A, lda, P, C, ldc, info, batchSize);
  return ret;
}

cublasStatus_t cublasCgetriBatched(cublasHandle_t handle, int n,
                                   const cuComplex *const A[], int lda,
                                   const int *P, cuComplex *const C[], int ldc,
                                   int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasCgetriBatched);
  cublasStatus_t ret = func(handle, n, A, lda, P, C, ldc, info, batchSize);
  return ret;
}

cublasStatus_t cublasZgetriBatched(cublasHandle_t handle, int n,
                                   const cuDoubleComplex *const A[], int lda,
                                   const int *P, cuDoubleComplex *const C[],
                                   int ldc, int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasZgetriBatched);
  cublasStatus_t ret = func(handle, n, A, lda, P, C, ldc, info, batchSize);
  return ret;
}

cublasStatus_t cublasSgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans, int n, int nrhs,
                                   const float *const Aarray[], int lda,
                                   const int *devIpiv, float *const Barray[],
                                   int ldb, int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasSgetrsBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(Aarray[i], n * n);
  }
  fromCublasInputPointerWithSize(devIpiv, n * batchSize);
  for (int i = 0; i < ldb; i++) {
    fromCublasInoutPointerWithSize(Barray[i], n * nrhs);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSgetrsBatched, handle);
  cublasStatus_t ret = func(handle, trans, n, nrhs, Aarray, lda, devIpiv,
                            Barray, ldb, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans, int n, int nrhs,
                                   const double *const Aarray[], int lda,
                                   const int *devIpiv, double *const Barray[],
                                   int ldb, int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasDgetrsBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(Aarray[i], n * n);
  }
  fromCublasInputPointerWithSize(devIpiv, n * batchSize);
  for (int i = 0; i < ldb; i++) {
    fromCublasInoutPointerWithSize(Barray[i], n * nrhs);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDgetrsBatched, handle);
  cublasStatus_t ret = func(handle, trans, n, nrhs, Aarray, lda, devIpiv,
                            Barray, ldb, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans, int n, int nrhs,
                                   const cuComplex *const Aarray[], int lda,
                                   const int *devIpiv,
                                   cuComplex *const Barray[], int ldb,
                                   int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasCgetrsBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(Aarray[i], n * n);
  }
  fromCublasInputPointerWithSize(devIpiv, n * batchSize);
  for (int i = 0; i < ldb; i++) {
    fromCublasInoutPointerWithSize(Barray[i], n * nrhs);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgetrsBatched, handle);
  cublasStatus_t ret = func(handle, trans, n, nrhs, Aarray, lda, devIpiv,
                            Barray, ldb, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgetrsBatched(cublasHandle_t handle,
                                   cublasOperation_t trans, int n, int nrhs,
                                   const cuDoubleComplex *const Aarray[],
                                   int lda, const int *devIpiv,
                                   cuDoubleComplex *const Barray[], int ldb,
                                   int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasZgetrsBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(Aarray[i], n * n);
  }
  fromCublasInputPointerWithSize(devIpiv, n * batchSize);
  for (int i = 0; i < ldb; i++) {
    fromCublasInoutPointerWithSize(Barray[i], n * nrhs);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgetrsBatched, handle);
  cublasStatus_t ret = func(handle, trans, n, nrhs, Aarray, lda, devIpiv,
                            Barray, ldb, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasStrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n,
                                  const float *alpha, const float *const A[],
                                  int lda, float *const B[], int ldb,
                                  int batchCount) {
  CUBLAS_ENTER_API(cublasStrsmBatched);
  for (int i = 0; i < lda; i++) {
    if (trans == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(A[i], lda * m);
    } else {
      fromCublasInputPointerWithSize(A[i], lda * n);
    }
  }
  for (int i = 0; i < ldb; i++) {
    fromCublasInoutPointerWithSize(B[i], ldb * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasStrsmBatched, handle);
  cublasStatus_t ret = func(handle, side, uplo, trans, diag, m, n, alpha, A,
                            lda, B, ldb, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                                  cublasFillMode_t uplo,
                                  cublasOperation_t trans,
                                  cublasDiagType_t diag, int m, int n,
                                  const double *alpha, const double *const A[],
                                  int lda, double *const B[], int ldb,
                                  int batchCount) {
  CUBLAS_ENTER_API(cublasDtrsmBatched);
  for (int i = 0; i < lda; i++) {
    if (trans == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(A[i], lda * m);
    } else {
      fromCublasInputPointerWithSize(A[i], lda * n);
    }
  }
  for (int i = 0; i < ldb; i++) {
    fromCublasInoutPointerWithSize(B[i], ldb * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDtrsmBatched, handle);
  cublasStatus_t ret = func(handle, side, uplo, trans, diag, m, n, alpha, A,
                            lda, B, ldb, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t
cublasCtrsmBatched(cublasHandle_t handle, cublasSideMode_t side,
                   cublasFillMode_t uplo, cublasOperation_t trans,
                   cublasDiagType_t diag, int m, int n, const cuComplex *alpha,
                   const cuComplex *const A[], int lda, cuComplex *const B[],
                   int ldb, int batchCount) {
  CUBLAS_ENTER_API(cublasCtrsmBatched);
  for (int i = 0; i < lda; i++) {
    if (trans == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(A[i], lda * m);
    } else {
      fromCublasInputPointerWithSize(A[i], lda * n);
    }
  }
  for (int i = 0; i < ldb; i++) {
    fromCublasInoutPointerWithSize(B[i], ldb * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCtrsmBatched, handle);
  cublasStatus_t ret = func(handle, side, uplo, trans, diag, m, n, alpha, A,
                            lda, B, ldb, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZtrsmBatched(
    cublasHandle_t handle, cublasSideMode_t side, cublasFillMode_t uplo,
    cublasOperation_t trans, cublasDiagType_t diag, int m, int n,
    const cuDoubleComplex *alpha, const cuDoubleComplex *const A[], int lda,
    cuDoubleComplex *const B[], int ldb, int batchCount) {
  CUBLAS_ENTER_API(cublasZtrsmBatched);
  for (int i = 0; i < lda; i++) {
    if (trans == CUBLAS_OP_N) {
      fromCublasInputPointerWithSize(A[i], lda * m);
    } else {
      fromCublasInputPointerWithSize(A[i], lda * n);
    }
  }
  for (int i = 0; i < ldb; i++) {
    fromCublasInoutPointerWithSize(B[i], ldb * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZtrsmBatched, handle);
  cublasStatus_t ret = func(handle, side, uplo, trans, diag, m, n, alpha, A,
                            lda, B, ldb, batchCount);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSmatinvBatched(cublasHandle_t handle, int n,
                                    const float *const A[], int lda,
                                    float *const Ainv[], int lda_inv, int *info,
                                    int batchSize) {
  CUBLAS_ENTER_API(cublasSmatinvBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(A[i], n * n);
  }
  for (int i = 0; i < lda_inv; i++) {
    fromCublasOutputPointerWithSize(Ainv[i], n * n);
  }
  fromCublasOutputPointerWithSize(info, batchSize);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSmatinvBatched, handle);
  cublasStatus_t ret = func(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDmatinvBatched(cublasHandle_t handle, int n,
                                    const double *const A[], int lda,
                                    double *const Ainv[], int lda_inv,
                                    int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasDmatinvBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(A[i], n * n);
  }
  for (int i = 0; i < lda_inv; i++) {
    fromCublasOutputPointerWithSize(Ainv[i], n * n);
  }
  fromCublasOutputPointerWithSize(info, batchSize);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDmatinvBatched, handle);
  cublasStatus_t ret = func(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCmatinvBatched(cublasHandle_t handle, int n,
                                    const cuComplex *const A[], int lda,
                                    cuComplex *const Ainv[], int lda_inv,
                                    int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasCmatinvBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(A[i], n * n);
  }
  for (int i = 0; i < lda_inv; i++) {
    fromCublasOutputPointerWithSize(Ainv[i], n * n);
  }
  fromCublasOutputPointerWithSize(info, batchSize);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCmatinvBatched, handle);
  cublasStatus_t ret = func(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZmatinvBatched(cublasHandle_t handle, int n,
                                    const cuDoubleComplex *const A[], int lda,
                                    cuDoubleComplex *const Ainv[], int lda_inv,
                                    int *info, int batchSize) {
  CUBLAS_ENTER_API(cublasZmatinvBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(A[i], n * n);
  }
  for (int i = 0; i < lda_inv; i++) {
    fromCublasOutputPointerWithSize(Ainv[i], n * n);
  }
  fromCublasOutputPointerWithSize(info, batchSize);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZmatinvBatched, handle);
  cublasStatus_t ret = func(handle, n, A, lda, Ainv, lda_inv, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSgeqrfBatched(cublasHandle_t handle, int m, int n,
                                   float *const Aarray[], int lda,
                                   float *const TauArray[], int *info,
                                   int batchSize) {
  CUBLAS_ENTER_API(cublasSgeqrfBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(Aarray[i], m * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSgeqrfBatched, handle);
  cublasStatus_t ret =
      func(handle, m, n, Aarray, lda, TauArray, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDgeqrfBatched(cublasHandle_t handle, int m, int n,
                                   double *const Aarray[], int lda,
                                   double *const TauArray[], int *info,
                                   int batchSize) {
  CUBLAS_ENTER_API(cublasDgeqrfBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(Aarray[i], m * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDgeqrfBatched, handle);
  cublasStatus_t ret =
      func(handle, m, n, Aarray, lda, TauArray, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgeqrfBatched(cublasHandle_t handle, int m, int n,
                                   cuComplex *const Aarray[], int lda,
                                   cuComplex *const TauArray[], int *info,
                                   int batchSize) {
  CUBLAS_ENTER_API(cublasCgeqrfBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(Aarray[i], m * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgeqrfBatched, handle);
  cublasStatus_t ret =
      func(handle, m, n, Aarray, lda, TauArray, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgeqrfBatched(cublasHandle_t handle, int m, int n,
                                   cuDoubleComplex *const Aarray[], int lda,
                                   cuDoubleComplex *const TauArray[], int *info,
                                   int batchSize) {
  CUBLAS_ENTER_API(cublasZgeqrfBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInputPointerWithSize(Aarray[i], m * n);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgeqrfBatched, handle);
  cublasStatus_t ret =
      func(handle, m, n, Aarray, lda, TauArray, info, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSgelsBatched(cublasHandle_t handle,
                                  cublasOperation_t trans, int m, int n,
                                  int nrhs, float *const Aarray[], int lda,
                                  float *const Carray[], int ldc, int *info,
                                  int *devInfoArray, int batchSize) {
  CUBLAS_ENTER_API(cublasSgelsBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInoutPointerWithSize(Aarray[i], m * n);
  }
  for (int i = 0; i < ldc; i++) {
    fromCublasInoutPointerWithSize(Carray[i], n * nrhs);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSgelsBatched, handle);
  cublasStatus_t ret = func(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc,
                            info, devInfoArray, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDgelsBatched(cublasHandle_t handle,
                                  cublasOperation_t trans, int m, int n,
                                  int nrhs, double *const Aarray[], int lda,
                                  double *const Carray[], int ldc, int *info,
                                  int *devInfoArray, int batchSize) {
  CUBLAS_ENTER_API(cublasDgelsBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInoutPointerWithSize(Aarray[i], m * n);
  }
  for (int i = 0; i < ldc; i++) {
    fromCublasInoutPointerWithSize(Carray[i], n * nrhs);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDgelsBatched, handle);
  cublasStatus_t ret = func(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc,
                            info, devInfoArray, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCgelsBatched(cublasHandle_t handle,
                                  cublasOperation_t trans, int m, int n,
                                  int nrhs, cuComplex *const Aarray[], int lda,
                                  cuComplex *const Carray[], int ldc, int *info,
                                  int *devInfoArray, int batchSize) {
  CUBLAS_ENTER_API(cublasCgelsBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInoutPointerWithSize(Aarray[i], m * n);
  }
  for (int i = 0; i < ldc; i++) {
    fromCublasInoutPointerWithSize(Carray[i], n * nrhs);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCgelsBatched, handle);
  cublasStatus_t ret = func(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc,
                            info, devInfoArray, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZgelsBatched(cublasHandle_t handle,
                                  cublasOperation_t trans, int m, int n,
                                  int nrhs, cuDoubleComplex *const Aarray[],
                                  int lda, cuDoubleComplex *const Carray[],
                                  int ldc, int *info, int *devInfoArray,
                                  int batchSize) {
  CUBLAS_ENTER_API(cublasZgelsBatched);
  for (int i = 0; i < lda; i++) {
    fromCublasInoutPointerWithSize(Aarray[i], m * n);
  }
  for (int i = 0; i < ldc; i++) {
    fromCublasInoutPointerWithSize(Carray[i], n * nrhs);
  }
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZgelsBatched, handle);
  cublasStatus_t ret = func(handle, trans, m, n, nrhs, Aarray, lda, Carray, ldc,
                            info, devInfoArray, batchSize);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m,
                           int n, const float *A, int lda, const float *x,
                           int incx, float *C, int ldc) {
  CUBLAS_ENTER_API(cublasSdgmm);
  fromCublasInputPointerWithSize(A, lda * n);
  if (mode == CUBLAS_SIDE_LEFT) {
  } else {
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasSdgmm, handle);
  cublasStatus_t ret = func(handle, mode, m, n, A, lda, x, incx, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m,
                           int n, const double *A, int lda, const double *x,
                           int incx, double *C, int ldc) {
  CUBLAS_ENTER_API(cublasDdgmm);
  fromCublasInputPointerWithSize(A, lda * n);
  if (mode == CUBLAS_SIDE_LEFT) {
  } else {
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDdgmm, handle);
  cublasStatus_t ret = func(handle, mode, m, n, A, lda, x, incx, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m,
                           int n, const cuComplex *A, int lda,
                           const cuComplex *x, int incx, cuComplex *C,
                           int ldc) {
  CUBLAS_ENTER_API(cublasCdgmm);
  fromCublasInputPointerWithSize(A, lda * n);
  if (mode == CUBLAS_SIDE_LEFT) {
  } else {
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCdgmm, handle);
  cublasStatus_t ret = func(handle, mode, m, n, A, lda, x, incx, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZdgmm(cublasHandle_t handle, cublasSideMode_t mode, int m,
                           int n, const cuDoubleComplex *A, int lda,
                           const cuDoubleComplex *x, int incx,
                           cuDoubleComplex *C, int ldc) {
  CUBLAS_ENTER_API(cublasZdgmm);
  fromCublasInputPointerWithSize(A, lda * n);
  if (mode == CUBLAS_SIDE_LEFT) {
  } else {
  }
  fromCublasInoutPointerWithSize(C, ldc * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZdgmm, handle);
  cublasStatus_t ret = func(handle, mode, m, n, A, lda, x, incx, C, ldc);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasStpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const float *AP, float *A, int lda) {
  CUBLAS_ENTER_API(cublasStpttr);
  fromCublasOutputPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasStpttr, handle);
  cublasStatus_t ret = func(handle, uplo, n, AP, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const double *AP, double *A, int lda) {
  CUBLAS_ENTER_API(cublasDtpttr);
  fromCublasOutputPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDtpttr, handle);
  cublasStatus_t ret = func(handle, uplo, n, AP, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const cuComplex *AP, cuComplex *A, int lda) {
  CUBLAS_ENTER_API(cublasCtpttr);
  fromCublasOutputPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCtpttr, handle);
  cublasStatus_t ret = func(handle, uplo, n, AP, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZtpttr(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const cuDoubleComplex *AP, cuDoubleComplex *A,
                            int lda) {
  CUBLAS_ENTER_API(cublasZtpttr);
  fromCublasOutputPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZtpttr, handle);
  cublasStatus_t ret = func(handle, uplo, n, AP, A, lda);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasStrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const float *A, int lda, float *AP) {
  CUBLAS_ENTER_API(cublasStrttp);
  fromCublasInputPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasStrttp, handle);
  cublasStatus_t ret = func(handle, uplo, n, A, lda, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasDtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const double *A, int lda, double *AP) {
  CUBLAS_ENTER_API(cublasDtrttp);
  fromCublasInputPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasDtrttp, handle);
  cublasStatus_t ret = func(handle, uplo, n, A, lda, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasCtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const cuComplex *A, int lda, cuComplex *AP) {
  CUBLAS_ENTER_API(cublasCtrttp);
  fromCublasInputPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasCtrttp, handle);
  cublasStatus_t ret = func(handle, uplo, n, A, lda, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasZtrttp(cublasHandle_t handle, cublasFillMode_t uplo, int n,
                            const cuDoubleComplex *A, int lda,
                            cuDoubleComplex *AP) {
  CUBLAS_ENTER_API(cublasZtrttp);
  fromCublasInputPointerWithSize(A, lda * n);
  CUBLAS_ENTER_COMPUTE_KERNEL(cublasZtrttp, handle);
  cublasStatus_t ret = func(handle, uplo, n, A, lda, AP);
  CUBLAS_LEAVE_COMPUTE_KERNEL(handle);
  return ret;
}

cublasStatus_t cublasInit() {
  CUBLAS_ENTER_API(cublasInit);
  cublasStatus_t ret = func();
  return ret;
}

cublasStatus_t cublasShutdown() {
  CUBLAS_ENTER_API(cublasShutdown);
  cublasStatus_t ret = func();
  return ret;
}

cublasStatus_t cublasGetError() {
  CUBLAS_ENTER_API(cublasGetError);
  cublasStatus_t ret = func();
  return ret;
}

cublasStatus_t cublasAlloc(int n, int elemSize, void **devicePtr) {
  CUBLAS_ENTER_API(cublasAlloc);
  cublasStatus_t ret = func(n, elemSize, devicePtr);
  return ret;
}

cublasStatus_t cublasFree(void *devicePtr) {
  CUBLAS_ENTER_API(cublasFree);
  cublasStatus_t ret = func(devicePtr);
  return ret;
}

cublasStatus_t cublasSetKernelStream(cudaStream_t stream) {
  CUBLAS_ENTER_API(cublasSetKernelStream);
  cublasStatus_t ret = func(stream);
  return ret;
}
