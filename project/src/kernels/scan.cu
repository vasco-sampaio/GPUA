#include "scan.cuh"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "../utils.cuh"


__device__
inline int scan_warp(int* data, const int tid) {
    const int lane = tid & 31; // index within the warp
    int tmp;

    #pragma unroll
    for (int i = 1; i <= 32; i *= 2) {
        tmp = __shfl_up_sync(0xffffffff, data[tid], i);
        if (lane >= i) data[tid] += tmp;
    }

    return data[tid];
}


__device__
int scan_block(int* data, const int tid) {
    const int lane = tid & 31;
    const int warpid = tid >> 5;

    int val = scan_warp(data, tid);
    __syncthreads();

    if(lane == 31) data[warpid] = data[tid]; __syncthreads();

    if(warpid == 0)
        scan_warp(data, tid); __syncthreads();

    if (warpid > 0) val += data[warpid - 1]; __syncthreads();

    data[tid] = val;
    __syncthreads();

    return val;
}

// using volatile instead of cuda::std::atomic<int> fixed synccheck error
__global__
void inclusive_scan_kernel(const int *input, int *output, const int size, int* block_count, /*cuda::std::atomic<int>**/ volatile int* blocks_executed) {
    __shared__ int bid;
    __shared__ int global_sum;
    extern __shared__ int sdata[];

    const int tid = threadIdx.x;

    if (tid == 0) {
        bid = atomicAdd(block_count, 1);
        global_sum = 0;
    }
    __syncthreads();

    if (bid * blockDim.x + tid >= size)
        return;

    sdata[tid] = input[bid * blockDim.x + tid];
    __syncthreads();

    int val = scan_block(sdata, tid);

    if (tid == blockDim.x - 1)
        output[bid * blockDim.x + tid] = val;
    __syncthreads();

    if (bid > 0 && tid == 0) {
        while (*blocks_executed/*->load()*/ < bid);
        global_sum = output[(bid - 1) * blockDim.x + blockDim.x - 1];
    }
    __syncthreads();

    output[bid * blockDim.x + tid] = val + global_sum;
    if (tid == blockDim.x - 1)
        (*blocks_executed)++/*->fetch_add(1)*/;
}


__global__
void single_block_scan_kernel(const int *input, int *output, const int size) {
    extern __shared__ int sdata[];

    const int tid = threadIdx.x;

    if (tid >= size)
        return;

    sdata[tid] = input[tid];
    __syncthreads();

    int val = scan_block(sdata, tid);
    __syncthreads();

    output[tid] = val;
}


void scan(int* input, int* output, const int size, cudaStream_t& stream) {
    int block_size = 256; // 256 because of the shared memory size
    int grid_size = (size + block_size - 1) / block_size;

    if (grid_size == 0) {
        block_size = NEXT_POW_2(size);
        grid_size = 1;
    }

    if (grid_size == 1)
        single_block_scan_kernel<<<1, block_size, block_size * sizeof(int), stream>>>(input, output, size);
    else {
        int* block_count;
        int* blocks_executed;
        // cuda::std::atomic<int>* blocks_executed;
        CUDA_CALL(cudaMalloc(&block_count, sizeof(int)));
        // cudaMalloc(&blocks_executed, sizeof(cuda::std::atomic<int>));
        CUDA_CALL(cudaMalloc(&blocks_executed, sizeof(int)));
        CUDA_CALL(cudaMemsetAsync(block_count, 0, sizeof(int), stream));
        // cudaMemsetAsync(blocks_executed, 0, sizeof(cuda::std::atomic<int>), stream);
        CUDA_CALL(cudaMemsetAsync(blocks_executed, 0, sizeof(int), stream));
        inclusive_scan_kernel<<<grid_size, block_size, block_size * sizeof(int), stream>>>(input, output, size, block_count, blocks_executed);
        CUDA_CALL(cudaFree(block_count));
        CUDA_CALL(cudaFree(blocks_executed));
    }
}