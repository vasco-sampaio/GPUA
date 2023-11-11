#include "histogram.cuh"

#include <cuda/atomic>

#include "../utils.cuh"


__global__
void histogram_kernel(int *histogram, const int *buffer, const int size, int* cdf_min) {
    __shared__ int sdata[256];
    sdata[threadIdx.x] = 0;
    __syncthreads();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int offset = blockDim.x * gridDim.x;

    while (i < size) {
        const int val = buffer[i];
        atomicAdd(&sdata[val], 1);
        atomicMin(cdf_min, val);

        i += offset;
    }
    __syncthreads();
    atomicAdd(&histogram[threadIdx.x], sdata[threadIdx.x]);
}

__global__
void histogram_equalization_kernel(int* buffer, const int* histogram, const int size, const int cdf_min) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size)
        buffer[tid] = roundf(((histogram[buffer[tid]] - cdf_min) / static_cast<float>(size - cdf_min)) * 255.0f);
}


int histogram(int* histogram, const int* buffer, const int size, cudaStream_t& stream) {
    const int block_size = 256;
    const int grid_size = (size + block_size - 1) / block_size;

    int *d_cdf;
    CUDA_CALL(cudaMalloc(&d_cdf, sizeof(int)));

    /*
     * __host__ cudaError_t cudaMemset ( void* devPtr, int  value, size_t count )
     * value: Value to set FOR EACH BYTE of specified memory
     * that is why cudaMemset(d_cdf, 256, sizeof(int)) does not work
     */

    int initial_cdf = 256;
    CUDA_CALL(cudaMemcpyAsync(d_cdf, &initial_cdf, sizeof(int), cudaMemcpyHostToDevice, stream));

    histogram_kernel<<<grid_size, block_size, 0, stream>>>(histogram, buffer, size, d_cdf);

    int cdf;
    CUDA_CALL(cudaMemcpyAsync(&cdf, d_cdf, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaFree(d_cdf));

    return cdf;
}


void histogram_equalization(int* buffer, const int* histogram, const int size, const int cdf_min_idx, cudaStream_t& stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    int cdf_min;
    CUDA_CALL(cudaMemcpyAsync(&cdf_min, &histogram[cdf_min_idx], sizeof(int), cudaMemcpyDeviceToHost, stream));

    histogram_equalization_kernel<<<grid_size, block_size, 0, stream>>>(buffer, histogram, size, cdf_min);
}
