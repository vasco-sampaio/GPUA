#include "scan.cuh"

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <iostream>
#include <stdio.h>

#define CUDA_CALL(x) cudaCheckError((x), __FILE__, __LINE__)

inline cudaError_t cudaCheckError(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
    return result;
}

__device__
int scan_warp(int* data, const int tid) {
    const int lane = tid & 31; // index within the warp

    // avoid race conditions
    int temp;

    if (lane >= 1) {
        temp = data[tid - 1] + data[tid];
        data[tid] = temp;
    }
    __syncwarp();

    if (lane >= 2) {
        temp = data[tid - 2] + data[tid];
        data[tid] = temp;
    }
    __syncwarp();

    if (lane >= 4) {
        temp = data[tid - 4] + data[tid];
        data[tid] = temp;
    }
    __syncwarp();

    if (lane >= 8) {
        temp = data[tid - 8] + data[tid];
        data[tid] = temp;
    }
    __syncwarp();

    if (lane >= 16) {
        temp = data[tid - 16] + data[tid];
        data[tid] = temp;
    }
    __syncwarp();

    return data[tid];
}

__device__
int scan_block(int* data, const int tid) {
    const int lane = tid & 31;
    const int warpid = tid >> 5;

    // Step 1: Intra-warp scan in each warp
    int val = scan_warp(data, tid);
    __syncthreads();

    // Step 2: Collect per-warp partial results
    if(lane == 31)
        data[warpid] = data[tid];
    __syncthreads();

    // Step 3: Use 1st warp to scan per-warp results
    if(warpid == 0)
        scan_warp(data, tid);
    __syncthreads();

    // Step 4: Accumulate results from Steps 1 and 3
    if (warpid > 0)
        val += data[warpid - 1];
    __syncthreads();

    // Step 5: Write and return the final result
    data[tid] = val;
    __syncthreads();

    return val;
}

__global__
void scan_kernel(const int *input, int *output, cuda::std::atomic<char> *flags, int *counter) {
    __shared__ int bid;
    extern __shared__ int sdata[];

    const int tid = threadIdx.x; 
    if (tid == 0)
        bid = atomicAdd(counter, 1);
    __syncthreads();

    sdata[tid] = input[bid * blockDim.x + tid];
    __syncthreads();

    int val = scan_block(sdata, tid);
    __syncthreads();

    if (tid == blockDim.x - 1) {
        output[bid * blockDim.x + tid] = val;
        flags[bid].store('A');
    }
    __syncthreads();

    if (bid > 0) {
        int i = bid - 1;
        char flag;
        do {
            flag = flags[i].load();

            if (flag == 'X')
                continue;

            val += output[i * blockDim.x + blockDim.x - 1];

            if (flag == 'P')
                break;

            i -= 1;
        } while (i > 0);
    }

    if (tid == blockDim.x - 1)
        flags[bid].store('P');
    __syncthreads();

    output[bid * blockDim.x + tid] = val;
}

void scan(int* input, int* output, int n) {
    int block_size = 64;
    int grid_size = (n + block_size - 1) / block_size;

    cuda::std::atomic<char>* flags;
    int* counter;

    CUDA_CALL(cudaMallocManaged(&flags, grid_size * sizeof(cuda::std::atomic<char>)));
    CUDA_CALL(cudaMemset(flags, 'X', grid_size * sizeof(cuda::std::atomic<char>)));

    CUDA_CALL(cudaMallocManaged(&counter, sizeof(int)));
    CUDA_CALL(cudaMemset(counter, 0, sizeof(int)));

    scan_kernel<<<grid_size, block_size, block_size * sizeof(int)>>>(input, output, flags, counter);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(flags));
    CUDA_CALL(cudaFree(counter));
}
