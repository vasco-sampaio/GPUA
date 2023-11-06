#include "scan.cuh"

#include <cuda_runtime.h>
#include <cuda/atomic>

#include "utils.cuh"

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

    if(warpid == 0)
        scan_warp(data, tid); __syncthreads();

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

    sdata[tid] = input[bid * blockDim.x + tid];
    __syncthreads();

    int val = scan_block(sdata, tid);
    __syncthreads();

    if (tid == blockDim.x - 1) {
        output[bid * blockDim.x + tid] = val;
        if (bid == 0)
            flags[bid].store(1);
    }
    __syncthreads();

    if (bid > 0) {
        while (flags[bid - 1].load() != 1);
        global_sum = output[(bid - 1) * blockDim.x + blockDim.x - 1];
    }
    __syncthreads();

    output[bid * blockDim.x + tid] = val + global_sum;
    if (tid == blockDim.x - 1)
        flags[bid].store(1);
}


__global__
void shift_kernel(const int *buffer, int *output, const int size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size)
        output[tid] = (tid == 0) ? 0 : output[tid - 1]; // BUG?
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


template <ScanType type>
void scan(int* input, int* output, const int size, cudaStream_t& stream) {
    int block_size = 256; // 256 because of the shared memory size
    int grid_size = (size + block_size - 1) / block_size;

    if (grid_size == 0) {
        block_size = NEXT_POW_2(size);
        grid_size = 1;
    }

    if (grid_size == 1) {
        single_block_scan_kernel<<<1, block_size, block_size * sizeof(int), stream>>>(input, output, size);
    } else {
        cuda::std::atomic<char>* flags;

        CUDA_CALL(cudaMallocManaged(&flags, grid_size * sizeof(cuda::std::atomic<char>)));
        CUDA_CALL(cudaMemset(flags, 0, grid_size * sizeof(cuda::std::atomic<char>)));

        inclusive_scan_kernel<<<grid_size, block_size, block_size * sizeof(int), stream>>>(input, output, flags, size);

        CUDA_CALL(cudaFree(flags));
    }

    if (ScanType::EXCLUSIVE == type)
        shift_kernel<<<grid_size, block_size, 0, stream>>>(input, output, size);
}


template void scan<ScanType::EXCLUSIVE>(int* input, int* output, const int size, cudaStream_t& stream);
template void scan<ScanType::INCLUSIVE>(int* input, int* output, const int size, cudaStream_t& stream);
