#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

typedef struct SYNCZR_PARAMS {
  float total_bandwidth; // <- Maximum network bandwidth of the synchronizer

} SYNCZR_PARAMS;

__global__ void synczr_kernel(SYNCZR_PARAMS sp);