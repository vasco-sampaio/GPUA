#include "filter.cuh"

#include <cuda/atomic>

#include "utils.cuh"


__global__
void predicate_kernel(int* predicate_buffer, const int* buffer, const int size) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size && buffer[i] != -27)
        predicate_buffer[i] = 1;
}


__global__
void scatter_kernel(int* buffer, int* output, const int* predicate, const int size) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size) {
        int val = buffer[i];
        if (val != -27)
            output[predicate[i] - 1] = val;
    }
}


__global__
void map_kernel(int* buffer, const int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        int val = buffer[i];
        if (i % 4 == 0) {
            val += 1;
            buffer[i] = max(0, min(255, val));  // Clamp between 0 and 255
        }
        else if (i % 4 == 1) {
            val -= 5;
            buffer[i] = max(0, min(255, val));
        }
        else if (i % 4 == 2) {
            val += 3;
            buffer[i] = max(0, min(255, val));
        }
        else if (i % 4 == 3) {
            val -= 8;
            buffer[i] = max(0, min(255, val));
        }
    }
}


void predicate(int* predicate, const int* buffer, const int size) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    predicate_kernel<<<grid_size, block_size>>>(predicate, buffer, size);
}


void scatter(int* buffer, int* output, const int* predicate, const int size) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    scatter_kernel<<<grid_size, block_size>>>(buffer, output, predicate, size);
}


void map(int* buffer, const int size) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    map_kernel<<<grid_size, block_size>>>(buffer, size);
}
