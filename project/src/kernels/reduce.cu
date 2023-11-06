#include "reduce.cuh"

#include "utils.cuh"


template<int BLOCK_SIZE>
__device__
void warpReduce(int* sdata, const int tid) {
    if constexpr (BLOCK_SIZE >= 64) sdata[tid] += sdata[tid + 32]; __syncwarp();
    if constexpr (BLOCK_SIZE >= 32) sdata[tid] += sdata[tid + 16]; __syncwarp();
    if constexpr (BLOCK_SIZE >= 16) sdata[tid] += sdata[tid + 8]; __syncwarp();
    if constexpr (BLOCK_SIZE >= 8) sdata[tid] += sdata[tid + 4]; __syncwarp();
    if constexpr (BLOCK_SIZE >= 4) sdata[tid] += sdata[tid + 2]; __syncwarp();
    if constexpr (BLOCK_SIZE >= 2) sdata[tid] += sdata[tid + 1]; __syncwarp();
}


template <int BLOCK_SIZE>
__global__
void reduce_kernel(const int* __restrict__ buffer, int* __restrict__ total)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    sdata[tid] = buffer[i]+ buffer[i + blockDim.x];
    __syncthreads();

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


void reduce(int* input, int* output, const int size)
{
    const int blockSize = 256;
    const int gridSize = (size + blockSize - 1) / (blockSize * 2);

    int *tmp;
    CUDA_CALL(cudaMalloc(&tmp, gridSize * sizeof(int)));

    reduce_kernel<1024><<<gridSize, blockSize, sizeof(int) * blockSize>>>(input, tmp);
    
    switch(gridSize / 2) {
        case 1024:
            reduce_kernel<1024><<<1, 1024, sizeof(int) * 1024>>>(tmp, output);
            break;
        case 512:
            reduce_kernel<512><<<1, 512, sizeof(int) * 512>>>(tmp, output);
            break;
        case 256:
            reduce_kernel<256><<<1, 256, sizeof(int) * 256>>>(tmp, output);
            break;
        case 128:
            reduce_kernel<128><<<1, 128, sizeof(int) * 128>>>(tmp, output);
            break;
        case 64:
            reduce_kernel<64><<<1, 64, sizeof(int) * 64>>>(tmp, output);
            break;
        case 32:
            reduce_kernel<32><<<1, 32, sizeof(int) * 32>>>(tmp, output);
            break;
        case 16:
            reduce_kernel<16><<<1, 16, sizeof(int) * 16>>>(tmp, output);
            break;
        case 8:
            reduce_kernel<8><<<1, 8, sizeof(int) * 8>>>(tmp, output);
            break;
        case 4:
            reduce_kernel<4><<<1, 4, sizeof(int) * 4>>>(tmp, output);
            break;
        case 2:
            reduce_kernel<2><<<1, 2, sizeof(int) * 2>>>(tmp, output);
            break;
        case 1:
            reduce_kernel<1><<<1, 1, sizeof(int) * 1>>>(tmp, output);
            break;
    }
}
