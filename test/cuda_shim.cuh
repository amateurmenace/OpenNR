// Minimal CUDA shim so clang can SYNTAX-CHECK CudaKernel.cu on a Mac with no
// CUDA toolkit (-x cuda --cuda-host-only -nocudainc). Declarations only —
// nothing here executes; the goal is catching undeclared identifiers, type
// errors and brace slips in the textual port.
#pragma once
// -nocudainc also drops clang's CUDA wrapper header, so declare the
// attribute macros it would normally provide
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __forceinline__ inline
typedef unsigned int uint;
typedef unsigned long long ulonglong;

struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct dim3 {
    unsigned x, y, z;
    __host__ __device__ dim3(unsigned vx = 1, unsigned vy = 1, unsigned vz = 1) : x(vx), y(vy), z(vz) {}
};
struct uint3 { unsigned x, y, z; };

__device__ uint3 threadIdx;
__device__ uint3 blockIdx;
__device__ uint3 blockDim;   // uint3, not dim3: device globals cannot have
__device__ uint3 gridDim;    // dynamic initializers (dim3 has a ctor)

__device__ inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
__device__ inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }

__device__ unsigned atomicAdd(unsigned* p, unsigned v);
__device__ float __uint_as_float(unsigned u);
__device__ unsigned __float_as_uint(float f);
__device__ float fminf(float, float);
__device__ float fmaxf(float, float);
__device__ float fabsf(float);
__device__ float sqrtf(float);
__device__ float expf(float);
__device__ float exp2f(float);      // Speak: nvcc provides these __device__
__device__ float logf(float);
__device__ float log2f(float);
__device__ float powf(float, float);
__device__ float log10f(float);
__device__ float floorf(float);
__device__ unsigned max(unsigned, unsigned);
__device__ int max(int, int);
__device__ int min(int, int);
__device__ unsigned long long max(unsigned long long, unsigned long long);

typedef struct CUstream_st* cudaStream_t;
typedef int cudaError_t;
cudaError_t cudaMalloc(void** p, unsigned long n);
template <typename T> cudaError_t cudaMalloc(T** p, unsigned long n) { return cudaMalloc((void**)p, n); }
cudaError_t cudaFree(void* p);
cudaError_t cudaMemsetAsync(void* p, int v, unsigned long n, cudaStream_t s);
// v3.5 R1: the render-boost history copy
enum cudaMemcpyKind { cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2,
                      cudaMemcpyDeviceToDevice = 3 };
cudaError_t cudaMemcpyAsync(void* dst, const void* src, unsigned long n,
                            cudaMemcpyKind kind, cudaStream_t s);

__device__ int abs(int);

// kernel-launch plumbing clang expects when parsing <<<...>>> host-only
// (without CUDA headers clang defaults to the legacy launch API)
extern "C" cudaError_t cudaConfigureCall(dim3 gridDim, dim3 blockDim,
                                         unsigned long sharedMem = 0,
                                         cudaStream_t stream = 0);
extern "C" unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                                unsigned long sharedMem = 0,
                                                void* stream = 0);
