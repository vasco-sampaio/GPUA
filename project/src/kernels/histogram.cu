#include "histogram.cuh"

#include <cuda/atomic>

#include "utils.cuh"


__global__
void histogram_kernel(int *histogram, const int *buffer, const int size) {
    __shared__ int sdata[256];
    sdata[threadIdx.x] = 0;
    __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;
    while (i < size) {
        atomicAdd(&sdata[buffer[i]], 1);
        i += offset;
    }
    __syncthreads();
    atomicAdd(&histogram[threadIdx.x], sdata[threadIdx.x]);
}

__global__
void histogram_equalization_kernel(int* buffer, const int* histogram, const int size, const int cdf_min) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size) {
        float normalized_pixel = (float)(histogram[buffer[tid]] - cdf_min) / (float)(size - cdf_min);
        buffer[tid] = static_cast<int>(normalized_pixel * 255.0f);
    }
}


void histogram(int* histogram, const int* buffer, const int size, cudaStream_t& stream) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    CUDA_CALL(cudaMemsetAsync(histogram, 0, 256 * sizeof(int), stream));

    histogram_kernel<<<grid_size, block_size, 0, stream>>>(histogram, buffer, size);

}


void histogram_equalization(int* buffer, const int* histogram, const int size, const int cdf_min, cudaStream_t& stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    histogram_equalization_kernel<<<grid_size, block_size, 0, stream>>>(buffer, histogram, size, cdf_min);
}
