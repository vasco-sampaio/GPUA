#include "histogram.cuh"

#include <cuda/atomic>

#include "utils.cuh"


__global__
void histogram_kernel(int* histogram, const int* buffer, const int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        atomicAdd(&histogram[buffer[i]], 1);
}


__global__
void histogram_equalization_kernel(int* buffer, const int* histogram, const int size, const int min, const int max) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        buffer[i] = (buffer[i] - min) * (max - min) / (size - 1);
}


void histogram(int* histogram, const int* buffer, const int size, cudaStream_t* stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    CUDA_CALL(cudaMemset(histogram, 0, 256 * sizeof(int)));

    histogram_kernel<<<grid_size, block_size, 0, *stream>>>(histogram, buffer, size);

    CUDA_CALL(cudaDeviceSynchronize());
}


void histogram_equalization(int* buffer, const int* histogram, const int size, const int min, const int max, cudaStream_t* stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    histogram_equalization_kernel<<<grid_size, block_size, 0, *stream>>>(buffer, histogram, size, min, max);

    CUDA_CALL(cudaDeviceSynchronize());
}
