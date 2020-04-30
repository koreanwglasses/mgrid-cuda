#pragma once

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <unsigned int blockSize>
__global__ void fmin_kernel(float *g_idata, float *g_odata, unsigned int n);

void launch_fmin_kernel (
  float * d_idata, 
  float * d_odata, 
  int n,
  int threads,
  int blocks
);