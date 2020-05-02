#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "client.h"
#include "utils.h"

// This module only contains helper functions: the bulk of the work is done in
// client.cu

float synczr_min_tc(CLIENT_PARAMS cp); 