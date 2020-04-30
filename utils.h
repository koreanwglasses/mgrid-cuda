#pragma once

#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>


template <unsigned int blockSize>
__global__ void fmin_kernel(float *g_idata, float *g_odata, unsigned int n);

// Computes the min of values in d_idata and stores the result (per block) in
// d_odata
// d_idata should have `n` floats
// d_odata should have enough memory for `blocks` floats
void launch_fmin_kernel (
  float * d_idata, 
  float * d_odata, 
  int n,
  int threads,
  int blocks
);