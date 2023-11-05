#pragma once 

#include <cuda_runtime.h>
#include <cuda/atomic>


__global__
void predicate_kernel(int* predicate, const int* buffer, const int size);

__global__
void scatter_kernel(const int* buffer, int* output, const int* predicate, const int size);

__global__
void map_kernel(int* buffer, const int size);

__global__
void histogram_kernel(int* histogram, const int* buffer, const int size);

__global__
void histogram_equalization_kernel(int* buffer, const int* histogram, const int size, const int min, const int max);

__global__
void scan_kernel(const int *input, int *output, cuda::std::atomic<int> *sums, const int size, bool INCLUSIVE=true);