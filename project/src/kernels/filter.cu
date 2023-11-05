#include "filter.cuh"

#include <cuda/atomic>

#include "utils.cuh"


__global__
void predicate_kernel(int* predicate, const int* buffer, const int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        predicate[i] = buffer[i] != -27 ? 1 : 0;
}


__global__
void scatter_kernel(int* buffer, const int* predicate, const int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size && buffer[i] != -27)
        buffer[predicate[i]] = buffer[i];
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


__global__
void find_if_kernel(int* buffer, predicate_t fct, int* result, const int size) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < size && fct(buffer[tid])) {
        int old = atomicCAS(result, -1, tid);
        if (old != -1)
            atomicMin(result, tid);
    }
}


void predicate(int* predicate, const int* buffer, const int size, cudaStream_t* stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    predicate_kernel<<<grid_size, block_size, 0, *stream>>>(predicate, buffer, size);

    CUDA_CALL(cudaDeviceSynchronize());
}


void scatter(int* buffer, const int* predicate, const int size, cudaStream_t* stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    scatter_kernel<<<grid_size, block_size, 0, *stream>>>(buffer, predicate, size);

    CUDA_CALL(cudaDeviceSynchronize());
}


void map(int* buffer, const int size, cudaStream_t* stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    map_kernel<<<grid_size, block_size, 0, *stream>>>(buffer, size);

    CUDA_CALL(cudaDeviceSynchronize());
}

int find_if(int* buffer, predicate_t fct, const int size, cudaStream_t* stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    int* result;
    CUDA_CALL(cudaMallocManaged(&result, sizeof(int)));
    CUDA_CALL(cudaMemset(result, -1, sizeof(int)));

    find_if_kernel<<<grid_size, block_size, 0, *stream>>>(buffer, fct, result, size);

    CUDA_CALL(cudaDeviceSynchronize());

    int res = *result;
    CUDA_CALL(cudaFree(result));

    return res;
}
