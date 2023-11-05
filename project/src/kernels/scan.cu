#include "scan.cuh"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "utils.cuh"


template <ScanType type>
__device__
int scan_warp(int* data, const int tid) {
    const int lane = tid & 31; // index within the warp

    if (lane >= 1)
        // avoid race conditions
        atomicAdd(&data[tid], data[tid - 1]);
    __syncwarp();

    if (lane >= 2)
        atomicAdd(&data[tid], data[tid - 2]);
    __syncwarp();

    if (lane >= 4)
        atomicAdd(&data[tid], data[tid - 4]);
    __syncwarp();

    if (lane >= 8)
        atomicAdd(&data[tid], data[tid - 8]);
    __syncwarp();

    if (lane >= 16)
        atomicAdd(&data[tid], data[tid - 16]);
    __syncwarp();

    if (type == ScanType::EXCLUSIVE)
        return lane > 0 ? data[tid - 1] : 0;
    return data[tid];
}


template <ScanType type>
__device__
int scan_block(int* data, const int tid) {
    const int lane = tid & 31;
    const int warpid = tid >> 5;

    // Step 1: Intra-warp scan in each warp
    int val = scan_warp<type>(data, tid);
    __syncthreads();

    // Step 2: Collect per-warp partial results
    if(lane == 31)
        data[warpid] = data[tid];
    __syncthreads();

    // Step 3: Use 1st warp to scan per-warp results
    if(warpid == 0)
        scan_warp<ScanType::INCLUSIVE>(data, tid);
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


template <ScanType type>
__global__
void scan_kernel(const int *input, int *output, cuda::std::atomic<char> *flags, int *counter, const int size) {
    __shared__ int bid;
    extern __shared__ int sdata[];

    const int tid = threadIdx.x;

    if (tid == 0)
        bid = atomicAdd(counter, 1);

    __syncthreads();

    // thread divergence
    if (bid * blockDim.x + tid >= size)
        return;

    sdata[tid] = input[bid * blockDim.x + tid];
    __syncthreads();

    int val;
    if (bid == 0)
        val = scan_block<type>(sdata, tid);
    else
        val = scan_block<ScanType::INCLUSIVE>(sdata, tid);
    __syncthreads();

    if (tid == blockDim.x - 1) {
        output[bid * blockDim.x + tid] = val;
        if (bid == 0)
            flags[bid].store(1);
    }
    __syncthreads();

    if (bid > 0) {
        while (flags[bid - 1].load() != 1);
        val += output[(bid - 1) * blockDim.x + blockDim.x - 1];
    }
    __syncthreads();

    if (tid == blockDim.x - 1)
        flags[bid].store(1);

    output[bid * blockDim.x + tid] = val;
}


template <ScanType type>
__global__
void single_block_scan_kernel(const int *input, int *output, const int size) {
    extern __shared__ int sdata[];

    const int tid = threadIdx.x;

    sdata[tid] = input[tid];
    __syncthreads();

    int val = scan_block<type>(sdata, tid);
    __syncthreads();

    output[tid] = val;
}

#include <stdio.h>
template <ScanType type>
void scan(int* input, int* output, const int size, cudaStream_t* stream) {
    int block_size = BLOCK_SIZE(size);
    int grid_size = (size + block_size - 1) / block_size;

    // if size is a power of two and inferior to 1024, we can use a single block
    if (size <= 1024 && NEXT_POW_2(size) == size) {
        single_block_scan_kernel<type><<<1, block_size, block_size * sizeof(int), *stream>>>(input, output, size);
        return;
    }

    cuda::std::atomic<char>* flags;
    int* counter;

    CUDA_CALL(cudaMallocManaged(&flags, grid_size * sizeof(cuda::std::atomic<char>)));
    CUDA_CALL(cudaMemset(flags, 0, grid_size * sizeof(cuda::std::atomic<char>)));

    CUDA_CALL(cudaMallocManaged(&counter, sizeof(int)));
    CUDA_CALL(cudaMemset(counter, 0, sizeof(int)));

    scan_kernel<type><<<grid_size, block_size, block_size * sizeof(int), *stream>>>(input, output, flags, counter, size);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(flags));
    CUDA_CALL(cudaFree(counter));
}


template void scan<ScanType::EXCLUSIVE>(int* input, int* output, const int size, cudaStream_t* stream);
template void scan<ScanType::INCLUSIVE>(int* input, int* output, const int size, cudaStream_t* stream);
