#pragma once

#include <cuda/std/atomic>

void scan(int* input, int* output, cuda::std::atomic<char> *flags);