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

template <typename T>
__global__
void kernel_your_reduce(const T* __restrict__ buffer, T* __restrict__ total, int size)
{
    // TODO
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    sdata[tid] = buffer[i]+ buffer[i + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s)
            sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

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

    kernel_your_reduce<int><<<gridSize, blockSize, sizeof(int) * blockSize>>>(buffer.data_, tmp, blockSize);

    kernel_your_reduce<int><<<1, gridSize, sizeof(int) * gridSize>>>(tmp, total.data_, gridSize);
    
    cudaDeviceSynchronize();
    kernel_check_error();
    
    cudaProfilerStop();
}