#include "client.h"

void alloc_client_params(int m, CLIENT_PARAMS* cp) {
  cp->t = 0;
  cp->m = m;
  cudaMallocManaged(&cp->B,  m * sizeof(float));
  cudaMallocManaged(&cp->C,  m * sizeof(float));
  cudaMallocManaged(&cp->L,  m * sizeof(float));
  cudaMallocManaged(&cp->Q,  m * sizeof(int));
  cudaMallocManaged(&cp->E,  m * sizeof(float));
  cudaMallocManaged(&cp->I,  m * sizeof(float));
  cudaMallocManaged(&cp->R,  m * sizeof(float));
  cudaMallocManaged(&cp->Tc, m * sizeof(float));
  cudaMallocManaged(&cp->K,  m * sizeof(int));
}

void init_client_params_zeros(CLIENT_PARAMS* cp) {
  int i;
  cp->t = 0;

  for(i = 0; i < cp->m; i++) {
    cp->B[i]  = 0;
    cp->C[i]  = 0;
    cp->L[i]  = 0;
    cp->Q[i]  = 0;
    cp->E[i]  = 0;
    cp->I[i]  = 0;
    cp->R[i]  = 0;
    cp->Tc[i] = 0;
    cp->K[i]  = 0;
  }
}

void free_client_params(CLIENT_PARAMS* cp) {
  cudaFree(cp->B);
  cudaFree(cp->C);
  cudaFree(cp->L);
  cudaFree(cp->Q);
  cudaFree(cp->E);
  cudaFree(cp->I);
  cudaFree(cp->R);
  cudaFree(cp->Tc);
  cudaFree(cp->K);
}

__global__ void client_kernel(CLIENT_PARAMS cp) {
  // See CLIENT_PARAMS struct definition for explanation of variables below
  const float  t = cp.t;
  const int    m = cp.m;
  const float* B = cp.B; 
  const float* C = cp.C; // might not be used in the client kernel, may only
                         // be needed for synchronizer (?)
  const float* L = cp.L; 
  const int*   Q = cp.Q;
  const float* E = cp.E; // ditto
  const float* I = cp.I;
  const float* R = cp.R; 
  
  float* Tc = cp.Tc;
  int*    K = cp.K;

  unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;

  float t_exec;

  while(k < m) {
    // Compute time to execute task (might not be used)
    t_exec = L[k] + I[k] * B[k] + I[k] * R[k]; // + computation time (?)
          //  ^          ^             ^
          // Latency     ^       upload results
          //      download task
      
    // Add this execution time to the next completioan time, if applicable
    K[k] += (t <= Tc[k] && Q[k]);
    Tc[k] += (t <= Tc[k] && Q[k]) * t_exec;

    k += blockDim.x * gridDim.x;
  }
}