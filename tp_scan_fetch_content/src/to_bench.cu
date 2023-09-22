#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>


template <typename T>
__global__
void kernel_scan_baseline(T* buffer, int size)
{
    for (int i = 1; i < size; ++i)
        buffer[i] += buffer[i - 1];
}

void baseline_scan(cuda_tools::host_shared_ptr<int> buffer)
{
    cudaProfilerStart();
    cudaFuncSetCacheConfig(kernel_scan_baseline<int>, cudaFuncCachePreferShared);

	kernel_scan_baseline<int><<<1, 1>>>(buffer.data_, buffer.size_);

    cudaDeviceSynchronize();
    kernel_check_error();
    
    cudaProfilerStop();
}

template <typename T>
__global__
void block_scan(T* buffer, int size)
{
    // TODO
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;

    sdata[tid] = buffer[tid];
    __syncthreads();

    for (auto s = 1; s < blockDim.x; s *= 2) {
        // avoid race conflicts of reading and writing in the same case
        int data;
        /*
         * if the condition eliminates some threads and there is a syncthreads
         * on it the behaviour is undefined
         */
        if (tid + s < blockDim.x)
            data = sdata[tid];
        __syncthreads();

        if (tid + s < blockDim.x)
            sdata[tid + s] += data;
        __syncthreads();
    }

    buffer[tid] = sdata[tid];
}

__inline__ __device__
int warp_reduce(int val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        val += __shfl_down_sync(~0, val, offset);
    return val;
}

/*
 * FLAGS:
 *  0 -> not done yet
 *  1 -> aggregate locally
 *  2 -> aggregate prefix
 */
template <typename T>
__global__
void kernel_your_scan(cuda::std::atomic<char>* flags, T* data, int* index, int size) {
    int id = atomicAdd_block(&index, 1);
    flags[id].load(0);

    int tid = threadIdx.x + id * blockDim.x;

    // reduce

    flags[id].load(1);

    // block scan

    int sum = 0;
    for (int i = 1; id - i >= 0; ++i) {
        while (flags[id - i] == 0);

        sum += data[(id - i) * blockDim.x + blockDim.x - 1];
        if (flags[id - i] == 2)
            break;
    }

    flags[id].load(2);
}

void your_scan(cuda_tools::host_shared_ptr<int> buffer)
{
    cudaProfilerStart();

    // TODO
    constexpr int blocksize = 256;
    constexpr int gridsize = (buffer.size_ + blocksize - 1) / blocksize;

    cuda::std::atomic<int>* flags;
    int* index;

    cudaMalloc(&flags, gridsize.size_ * sizeof(cuda::std::atomic<char>));
    cudaMalloc(&index, sizeof(int));

    *index = 0;

    kernel_your_scan<char><<<gridsize, blocksize>>>(cuda::std::atomic<bool>* flags, buffer.data_, int* index, buffer.size_);

    cudaFree(&flags);
    cudaDeviceSynchronize();
    kernel_check_error();
    
    cudaProfilerStop();
}