#include "synczr.h"

float synczr_min_tc(CLIENT_PARAMS cp) {
  float* odata;
  cudaMallocManaged(&odata, sizeof(float));
  launch_fmin_kernel(cp.Tc, odata, cp.m, 1024, 1);
  cudaDeviceSynchronize();
  return *odata;
}