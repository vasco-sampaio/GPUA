#include "scan.cuh"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "utils.h"

__device__
int scan_warp(int* data, const int tid) {
    const int lane = tid & 31; // index within the warp

    if (lane >= 1) data[tid] += data[tid - 1]; __syncwarp();
    if (lane >= 2) data[tid] += data[tid - 2]; __syncwarp();
    if (lane >= 4) data[tid] += data[tid - 4]; __syncwarp();
    if (lane >= 8) data[tid] += data[tid - 8]; __syncwarp();
    if (lane >= 16) data[tid] += data[tid - 16]; __syncwarp();

    return data[tid];
}


__device__
int scan_block(int* data, const int tid) {
    const int lane = tid & 31;
    const int warpid = tid >> 5;

    int val = scan_warp(data, tid);
    __syncthreads();

    if(lane == 31) data[warpid] = data[tid]; __syncthreads();
    if(warpid == 0) scan_warp(data, tid); __syncthreads();
    if (warpid > 0) val += data[warpid - 1]; __syncthreads();

    data[tid] = val;
    __syncthreads();

    return val;
}


__device__ int block_count = 0;

__global__
void inclusive_scan_kernel(const int *input, int *output, cuda::std::atomic<char> *flags, const int size) {
    __shared__ int bid;
    __shared__ int global_sum;
    extern __shared__ int sdata[];

    const int tid = threadIdx.x;

    if (tid == 0)
        bid = atomicAdd(&block_count, 1);
    __syncthreads();

    if (bid * blockDim.x + tid >= size)
        return;

    int thread_value = input[bid * blockDim.x + tid];
    sdata[tid] = thread_value;
    __syncthreads();

    int val = scan_block(sdata, tid);
    __syncthreads();

    if (bid == 0 && tid == size < blockDim.x ? size - 1 : blockDim.x - 1)
        flags[0].store(1);

    if (bid > 0 && tid == 0) {
        while (flags[bid - 1].load() == 0);
        global_sum = output[(bid - 1) * blockDim.x + blockDim.x - 1];
    }
    __syncthreads();

    output[bid * blockDim.x + tid] = val + global_sum;
    __syncthreads();
    if (tid == 0)
        flags[bid].store(1);
}


__global__
void exclusive_scan_kernel(const int *input, int* output, const int size) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size)
        output[tid] -= input[tid];
}


void scan(int* input, int* output, const int size, bool INCLUSIVE) {
    int block_size = BLOCK_SIZE(size);
    int grid_size = (size + block_size - 1) / block_size;

    cuda::std::atomic<char>* flags;

    cudaMalloc(&flags, grid_size * sizeof(cuda::std::atomic<char>));
    cudaMemset(flags, 0, grid_size * sizeof(cuda::std::atomic<char>));

    inclusive_scan_kernel<<<grid_size, block_size, block_size * sizeof(int)>>>(input, output, flags, size);

    if (!INCLUSIVE)
        exclusive_scan_kernel<<<grid_size, block_size, 0>>>(input, output, size);

    cudaFree(flags);
}
