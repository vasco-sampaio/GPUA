#pragma once

#include <cuda/std/atomic>

void scan(int* input, int* output, int n);