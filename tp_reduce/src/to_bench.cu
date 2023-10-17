#include "to_bench.cuh"

#include "cuda_tools/cuda_error_checking.cuh"
#include "cuda_tools/host_shared_ptr.cuh"

#include <cuda_profiler_api.h>

#include <iostream>

template <typename T>
__global__
void kernel_reduce_baseline(const T* __restrict__ buffer, T* __restrict__ total, int size)
{
    const int id = threadIdx.x + blockIdx.x * blockDim.x;
    if (id < size)
        atomicAdd(&total[0], buffer[id]);
}

void baseline_reduce(cuda_tools::host_shared_ptr<int> buffer,
    cuda_tools::host_shared_ptr<int> total)
{
    cudaProfilerStart();
    cudaFuncSetCacheConfig(kernel_reduce_baseline<int>, cudaFuncCachePreferShared);

    constexpr int blocksize = 64;
    const int gridsize = (buffer.size_ + blocksize - 1) / blocksize;

    kernel_reduce_baseline<int><<<gridsize, blocksize>>>(buffer.data_, total.data_, buffer.size_);

    cudaDeviceSynchronize();
    kernel_check_error();
    
    cudaProfilerStop();
}

template<int BLOCK_SIZE>
__device__
void warpReduce(int* sdata, int tid) {
    /*
     * Volatile ensures that there is not reordering between each
     * reduce step, but it's legacy.
     * __syncwarp is more modern
     */
    if constexpr (BLOCK_SIZE >= 64) sdata[tid] += sdata[tid + 32]; __syncwarp();
    if constexpr (BLOCK_SIZE >= 32) sdata[tid] += sdata[tid + 16]; __syncwarp();
    if constexpr (BLOCK_SIZE >= 16) sdata[tid] += sdata[tid + 8]; __syncwarp();
    if constexpr (BLOCK_SIZE >= 8) sdata[tid] += sdata[tid + 4]; __syncwarp();
    if constexpr (BLOCK_SIZE >= 4) sdata[tid] += sdata[tid + 2]; __syncwarp();
    if constexpr (BLOCK_SIZE >= 2) sdata[tid] += sdata[tid + 1]; __syncwarp();
}


template <typename T, int BLOCK_SIZE>
__global__
void kernel_your_reduce(const T* __restrict__ buffer, T* __restrict__ total)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = buffer[i]+ buffer[i + blockDim.x];
    __syncthreads();

    /*
     * The maximum threads per block and the first iteration is
     * blockDim.x / 2.
     */
    if constexpr (BLOCK_SIZE == 1024) {
        if (tid < 512)
            sdata[tid] += sdata[tid + 512];
        __syncthreads();
    }

    if constexpr (BLOCK_SIZE >= 512) {
        if (tid < 256)
            sdata[tid] += sdata[tid + 256];
        __syncthreads();
    }

    if constexpr (BLOCK_SIZE >= 256) {
        if (tid < 128)
            sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }

    if constexpr (BLOCK_SIZE >= 128) {
        if (tid < 64)
            sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }

    if (tid < 32) warpReduce<BLOCK_SIZE>(sdata, tid);

    if (tid == 0)
        total[blockIdx.x] = sdata[0];
}


void your_reduce(cuda_tools::host_shared_ptr<int> buffer,
    cuda_tools::host_shared_ptr<int> total)
{
    cudaProfilerStart();

    /*
    * Si le nombre de threads est trop faible on va generer un grand nombre de blocs et le
    * deuxieme appel au kernel ne va pas traiter toutes les valeurs car pas assez de threads dans un bloc
    */
    const int blockSize = 1024;

    /* 
     * On divise le nombre de blocs par 2 parce que chaque thread est desormais charge de load 2 valeurs, 
     * ceci a ete fait parce que au bout d'une iteration la moitie des threads etait inutilisee donc autant 
     * augmenter le work per thread
     */
    const int gridSize = (buffer.size_ + blockSize - 1) / (blockSize * 2);

    int *tmp;
    cudaMalloc(&tmp, gridSize * sizeof(int));

    kernel_your_reduce<int, 1024><<<gridSize, blockSize, sizeof(int) * blockSize>>>(buffer.data_, tmp);
    
    switch(gridSize / 2) {
        case 1024:
            kernel_your_reduce<int, 1024><<<1, 1024, sizeof(int) * 1024>>>(tmp, total.data_);
            break;
        case 512:
            kernel_your_reduce<int, 512><<<1, 512, sizeof(int) * 512>>>(tmp, total.data_);
            break;
        case 256:
            kernel_your_reduce<int, 256><<<1, 256, sizeof(int) * 256>>>(tmp, total.data_);
            break;
        case 128:
            kernel_your_reduce<int, 128><<<1, 128, sizeof(int) * 128>>>(tmp, total.data_);
            break;
        case 64:
            kernel_your_reduce<int, 64><<<1, 64, sizeof(int) * 64>>>(tmp, total.data_);
            break;
        case 32:
            kernel_your_reduce<int, 32><<<1, 32, sizeof(int) * 32>>>(tmp, total.data_);
            break;
        case 16:
            kernel_your_reduce<int, 16><<<1, 16, sizeof(int) * 16>>>(tmp, total.data_);
            break;
        case 8:
            kernel_your_reduce<int, 8><<<1, 8, sizeof(int) * 8>>>(tmp, total.data_);
            break;
        case 4:
            kernel_your_reduce<int, 4><<<1, 4, sizeof(int) * 4>>>(tmp, total.data_);
            break;
        case 2:
            kernel_your_reduce<int, 2><<<1, 2, sizeof(int) * 2>>>(tmp, total.data_);
            break;
        case 1:
            kernel_your_reduce<int, 1><<<1, 1, sizeof(int) * 1>>>(tmp, total.data_);
            break;
    }
   
    cudaDeviceSynchronize();
    kernel_check_error();
    
    cudaProfilerStop();
}