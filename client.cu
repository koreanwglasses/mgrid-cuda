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

__global__ void client_step_kernel(CLIENT_PARAMS cp) {
  // See CLIENT_PARAMS struct definition for explanation of variables below
  const float  t = cp.t;
  const int    m = cp.m;
  const float* B = cp.B; 
  const float* C = cp.C;
  const float* L = cp.L; 
  const int*   Q = cp.Q;
  const float* E = cp.E;
  const float* I = cp.I;
  const float* R = cp.R; 
  
  float* Tc = cp.Tc;
  int*    K = cp.K;

  unsigned int k = blockIdx.x*blockDim.x + threadIdx.x;

  while(k < m) {
    // Compute time to execute task (might not be used)
    float t_latency = L[k]; // latency
    float t_download = I[k] * B[k]; // download time
    float t_compute = E[k]; // computation time (?) not sure how to compute this
    float t_upload = I[k] * R[k]; // upload time
    float t_exec = t_latency + t_download + t_compute + t_upload; 
      
    // Add this execution time to the next completioan time, if applicable
    int is_job_complete = t <= Tc[k]; // is current/last job complete
    int is_job_available = Q[k]; // is new job available
    // (?) specifically for game of life: have neighbors completed their tasks?
    int are_deps_complete = K[k] >= K[(k - 1 + m) % m] && K[k] >= K[(k + 1) % m]; 
    int is_free = is_job_complete && is_job_available && are_deps_complete;

    K[k] += is_free;
    Tc[k] += is_free * t_exec;

    k += blockDim.x * gridDim.x;
  }
}