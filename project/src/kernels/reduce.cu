#include "reduce.cuh"

#include "../utils.cuh"


__device__
inline int warpReduce(int val) {
    #pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}


__global__
void reduce_kernel(const int* __restrict__ buffer, int* __restrict__ total, const int size)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gridSize = blockDim.x * gridDim.x;

    sdata[tid] = 0;
    while (i < size) {
        sdata[tid] += buffer[i];
        i += gridSize;
    }
    __syncthreads();

    int val = warpReduce(sdata[tid]);

    if (tid % warpSize == 0)
        atomicAdd(total, val);
}

int reduce(int* input, const int size, cudaStream_t& stream)
{
    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / (blockSize * 2);

    int* output;
    CUDA_CALL(cudaMalloc(&output, sizeof(int)));
    CUDA_CALL(cudaMemsetAsync(output, 0, sizeof(int), stream));

    reduce_kernel<<<gridSize, blockSize, sizeof(int) * blockSize, stream>>>(input, output, size);

    int total;
    CUDA_CALL(cudaMemcpyAsync(&total, output, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaFree(output));

    return total;
}
