#include "filter.cuh"

#include <cuda/atomic>

#include "utils.cuh"


__global__
void predicate_kernel(int* predicate_buffer, const int* buffer, const int size) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size)
        predicate_buffer[i] += (buffer[i] != -27);
}


__global__
void scatter_kernel(int* buffer, int* output, const int* predicate, const int size) {
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

template <FindType F>
__global__
void find_first_value(const int *data, const int size, const int valueToFind, int *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        if constexpr (F == FindType::SMALLER) {
            if (data[tid] < valueToFind) {
                if (atomicCAS(result, -1, tid) != -1)
                    atomicMin(result, tid);
            }
        }
        else if constexpr (F == FindType::EQUAL) {
            if (data[tid] == valueToFind) {
                if (atomicCAS(result, -1, tid) != -1)
                    atomicMin(result, tid);
            }
        }
        else if constexpr (F == FindType::BIGGER) {
            if (data[tid] > valueToFind) {
                if (atomicCAS(result, -1, tid) != -1)
                    atomicMin(result, tid);
            }
        }
    }
}


void predicate(int* predicate, const int* buffer, const int size, cudaStream_t& stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    predicate_kernel<<<grid_size, block_size, 0, stream>>>(predicate, buffer, size);
}


void scatter(int* buffer, int* output, const int* predicate, const int size, cudaStream_t& stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    scatter_kernel<<<grid_size, block_size, 0, stream>>>(buffer, output, predicate, size);
}


void map(int* buffer, const int size, cudaStream_t& stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    map_kernel<<<grid_size, block_size, 0, stream>>>(buffer, size);
}

template <FindType F>
int find_index(const int* buffer, const int size, const int value, cudaStream_t& stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    int* result;
    CUDA_CALL(cudaMallocManaged(&result, sizeof(int)));
    CUDA_CALL(cudaMemset(result, -1, sizeof(int)));

    find_first_value<F><<<grid_size, block_size, 0, stream>>>(buffer, size, value, result);
    CUDA_CALL(cudaStreamSynchronize(stream));

    int res = *result;
    CUDA_CALL(cudaFree(result));

    return res;
}

template int find_index<FindType::SMALLER>(const int*, const int, const int, cudaStream_t&);
template int find_index<FindType::EQUAL>(const int*, const int, const int, cudaStream_t&);
template int find_index<FindType::BIGGER>(const int*, const int, const int, cudaStream_t&);
