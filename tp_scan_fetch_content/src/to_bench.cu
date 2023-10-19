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
__global__ void kernel_your_scan(const T* __restrict__ g_idata, T* g_odata, int n) {
    extern __shared__ T sdata[];

    int tid = threadIdx.x;
    int pout = 0, pin = 1;

    // exclusive scan so shifted by 1
    sdata[pout * n + tid] = (tid > 0) ? g_idata[tid - 1] : 0;
    __syncthreads();

    for (int offset = 1; offset < n; offset *= 2) {
        pout = 1 - pout;
        pin = 1 - pout;

        if (tid >= offset)
            sdata[pout * n + tid] += sdata[pin * n + tid - offset];
        else
            sdata[pout * n + tid] += sdata[pin * n + tid];
    
        __syncthreads();
    }

    g_odata[tid] += sdata[pout * n + tid];
}

void your_scan(cuda_tools::host_shared_ptr<int> buffer)
{
    cudaProfilerStart();

    // TODO
    const int blocksize = 256;
    const int gridsize = (buffer.size_ + blocksize - 1) / blocksize;

    int* index;
    cudaMalloc(&index, buffer.size_ * sizeof(int));

    kernel_your_scan<int><<<gridsize, blocksize, buffer.size_ * sizeof(int)>>>(buffer.data_, index, buffer.size_);

    cudaDeviceSynchronize();
    kernel_check_error();
    
    cudaProfilerStop();
}