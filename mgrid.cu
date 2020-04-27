#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "client.h"

#define N_THREADS 1024


int main() {
  CLIENT_PARAMS cp;
  alloc_client_params(1048576, &cp);
  init_client_params_zeros(&cp);

  client_step_kernel<<<(cp.m + N_THREADS - 1) / N_THREADS, N_THREADS>>>(cp);

  free_client_params(&cp);

  printf("Done!\n");
  return 0;
}
