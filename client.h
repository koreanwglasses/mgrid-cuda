#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

// A struct is used to keep function prototypes concise
typedef struct CLIENT_PARAMS {
  float   t; // t        <- current time
  int     m; // m        <- number of devices in the cluster
  float*  B; // B[k]     <- available bandwidth at kth device
  float*  C; // C[k]     <- shared processing capacity of the kth device
  float*  L; // C[k]     <- latency between the synchronizer and the kth
             //             device
  int*    Q; // Q[k]     <- is there a task in queue for the kth device
  float*  E; // E[k]     <- the execution load of the next task for the 
             //             kth device
  float*  I; // I[k]     <- size of the transferable input data and
             //             executable code of the next task for the kth
             //             device
  float*  R; // R[k]     <- size of the results of the next task for the 
             //             kth device
  float* Tc; // Tc[k]    -> completion time of current/last task at kth
             //             device
  int*    K; // K[k]     -> number of tasks loaded by kth device (used by
             //             synchronizer to determine next task)
} CLIENT_PARAMS;

void alloc_client_params(int m, CLIENT_PARAMS* cp);
void init_client_params_zeros(CLIENT_PARAMS* cp);
void free_client_params(CLIENT_PARAMS* cp);

// Simulates the execution of a task on the cluster by computing the completion
// time for the task performed. Assumes bandwidth is only limited by
// synchronizer and that latency is constant. At most one task is queued up for
// when the node becomes free. When t > Tc[k], the new task is loaded, and the
// appropriate variables changed.
__global__ void client_kernel(CLIENT_PARAMS cp);