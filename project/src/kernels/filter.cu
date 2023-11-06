#include "filter.cuh"

#include <cuda/atomic>

#include "utils.h"


__global__
void predicate_kernel(int* predicate_buffer, const int* buffer, const int size) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size)
        predicate_buffer[i] += (buffer[i] != -27);
}


__global__
void scatter_kernel(const int* buffer, int* output, const int* predicate, const int size) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size)
        if (buffer[i] != -27)
            output[predicate[i]] = buffer[i];
}


__global__
void map_kernel(int* buffer, const int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        if (i % 4 == 0)
            buffer[i] += 1;
        else if (i % 4 == 1)
            buffer[i] -= 5;
        else if (i % 4 == 2)
            buffer[i] += 3;
        else if (i % 4 == 3)
            buffer[i] -= 8;
    }
}

void predicate(int* predicate_buffer, const int* buffer, const int size) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    predicate_kernel<<<grid_size, block_size, 0>>>(predicate_buffer, buffer, size);
}


void scatter(int* buffer, int* output, const int* predicate, const int size) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    scatter_kernel<<<grid_size, block_size, 0>>>(buffer, output, predicate, size);
}


void map(int* buffer, const int size) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    map_kernel<<<grid_size, block_size, 0>>>(buffer, size);
}

