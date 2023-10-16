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

template <int BLOCK_SIZE>
__device__
void warpReduce(int* sdata, int tid) {
    /*
     * Volatile ensures that there is not reordering between each
     * reduce step, but it's legacy.
     * __syncwarp is more modern
     */

    if (BLOCK_SIZE >= 64) {
        sdata[tid] += sdata[tid + 32]; __syncwarp();
    }
    if (BLOCK_SIZE >= 32) {
        sdata[tid] += sdata[tid + 16]; __syncwarp();
    }
    if (BLOCK_SIZE >= 16) {
        sdata[tid] += sdata[tid + 8]; __syncwarp();
    }
    if (BLOCK_SIZE >= 8) {
        sdata[tid] += sdata[tid + 4]; __syncwarp();
    }
    if (BLOCK_SIZE >= 4) {
        sdata[tid] += sdata[tid + 2]; __syncwarp();
    }
    if (BLOCK_SIZE >= 2) {
        sdata[tid] += sdata[tid + 1]; __syncwarp();
    }
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
    if constexpr (BLOCK_SIZE >= 512) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 256]; __syncthreads();
        }
    }
    if constexpr (BLOCK_SIZE >= 256) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 128]; __syncthreads();
        }
    }
    if constexpr (BLOCK_SIZE >= 128) {
        if (tid < 256) {
            sdata[tid] += sdata[tid + 64]; __syncthreads();
        }
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

    printf("First Kernels\n");

    kernel_your_reduce<int, 512><<<gridSize, blockSize, sizeof(int) * blockSize>>>(buffer.data_, tmp);
    kernel_your_reduce<int, 256><<<gridSize, blockSize, sizeof(int) * blockSize>>>(buffer.data_, tmp);
    kernel_your_reduce<int, 128><<<gridSize, blockSize, sizeof(int) * blockSize>>>(buffer.data_, tmp);
   
    printf("Last Kernel\n");
    kernel_your_reduce<int, 64><<<1, gridSize / 2, sizeof(int) * gridSize>>>(tmp, total.data_);
   
    cudaDeviceSynchronize();
    kernel_check_error();
    
    cudaProfilerStop();
}