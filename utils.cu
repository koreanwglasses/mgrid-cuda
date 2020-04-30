#include <stdio.h>
#include "utils.h"

// Optimized reduction adapted from:
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template <unsigned int blockSize>
__device__ void fmin_warpReduce(volatile float *sdata, unsigned int tid) {
  if (blockSize >= 64) sdata[tid] = fmin(sdata[tid], sdata[tid + 32]);
  if (blockSize >= 32) sdata[tid] = fmin(sdata[tid], sdata[tid + 16]);
  if (blockSize >= 16) sdata[tid] = fmin(sdata[tid], sdata[tid +  8]);
  if (blockSize >= 8)  sdata[tid] = fmin(sdata[tid], sdata[tid +  4]);
  if (blockSize >= 4)  sdata[tid] = fmin(sdata[tid], sdata[tid +  2]);
  if (blockSize >= 2)  sdata[tid] = fmin(sdata[tid], sdata[tid +  1]);
}

template <unsigned int blockSize>
__global__ void fmin_kernel(float *g_idata, float *g_odata, unsigned int n) {
  extern __shared__ float sdata[];
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockSize*2) + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;

  sdata[tid] = INFINITY;
  while (i < n) { 
    float next = i + blockSize < n ? g_idata[i+blockSize] : INFINITY; // Make sure we don't run over array
    sdata[tid] = fmin(sdata[tid], fmin(g_idata[i], next)); 
    i += gridSize; 
  }
  __syncthreads();
  if (blockSize >= 1024) { if (tid < 512) { sdata[tid] = fmin(sdata[tid], sdata[tid + 512]); } __syncthreads(); }
  if (blockSize >=  512) { if (tid < 256) { sdata[tid] = fmin(sdata[tid], sdata[tid + 256]); } __syncthreads(); }
  if (blockSize >=  256) { if (tid < 128) { sdata[tid] = fmin(sdata[tid], sdata[tid + 128]); } __syncthreads(); }
  if (blockSize >=  128) { if (tid <  64) { sdata[tid] = fmin(sdata[tid], sdata[tid +  64]); } __syncthreads(); }
  if (tid < 32) fmin_warpReduce<blockSize>(sdata, tid);
  if (tid == 0) { 
    g_odata[blockIdx.x] = sdata[0];
  }
}

void launch_fmin_kernel (
  float * d_idata, 
  float * d_odata, 
  int n,
  int threads,
  int blocks
) {
  int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

  switch (threads)
  {
    case 1024:
      fmin_kernel<1024><<< blocks, threads, smemSize >>>(d_idata, d_odata, n); break;
    case 512:
      fmin_kernel< 512><<< blocks, threads, smemSize >>>(d_idata, d_odata, n); break;
    case 256:
      fmin_kernel< 256><<< blocks, threads, smemSize >>>(d_idata, d_odata, n); break;
    case 128:
      fmin_kernel< 128><<< blocks, threads, smemSize >>>(d_idata, d_odata, n); break;
    case 64:
      fmin_kernel<  64><<< blocks, threads, smemSize >>>(d_idata, d_odata, n); break;
    case 32:
      fmin_kernel<  32><<< blocks, threads, smemSize >>>(d_idata, d_odata, n); break;
    case 16:
      fmin_kernel<  16><<< blocks, threads, smemSize >>>(d_idata, d_odata, n); break;
    case 8:
      fmin_kernel<   8><<< blocks, threads, smemSize >>>(d_idata, d_odata, n); break;
    case 4:
      fmin_kernel<   4><<< blocks, threads, smemSize >>>(d_idata, d_odata, n); break;
    case 2:
      fmin_kernel<   2><<< blocks, threads, smemSize >>>(d_idata, d_odata, n); break;
    case 1:
      fmin_kernel<   1><<< blocks, threads, smemSize >>>(d_idata, d_odata, n); break;
  }
}