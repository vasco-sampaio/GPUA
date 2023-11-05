#include "kernels.cuh"

__global__
void predicate_kernel(int* predicate, const int* buffer, const int size) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size)
        predicate[i] += (buffer[i] != -27);
}


__global__
void scatter_kernel(const int* buffer, int* output, const int* predicate, const int size) {
    const int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < size)
        if (buffer[i] != -27)
            output[predicate[i]] = buffer[i];
}


__global__
void map_kernel(int* buffer, const int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
    {
        if (i % 4 == 0)
            buffer[i] += 1;
        else if (i % 4 == 1)
            buffer[i] -= 5;
        else if (i % 4 == 2)
            buffer[i] += 3;
        else if (i % 4 == 3)
            buffer[i] -= 8;
    }
}


__global__
void histogram_kernel(int* histogram, const int* buffer, const int size) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        atomicAdd(&histogram[buffer[i]], 1);
}


__global__
void histogram_equalization_kernel(int* buffer, const int* histogram, const int size, const int min, const int max) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < size)
        buffer[i] = (buffer[i] - min) * (max - min) / (size - 1);
}


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

    return data[tid];
}


__device__
int scan_block(int* data, const int tid) {
    const int lane = tid & 31;
    const int warpid = tid >> 5;

    int val = scan_warp(data, tid);
    __syncthreads();

    if(lane == 31)
        data[warpid] = data[tid];
    __syncthreads();

    if(warpid == 0)
        scan_warp(data, tid);
    __syncthreads();

    if (warpid > 0)
        val += data[warpid - 1];
    __syncthreads();

    data[tid] = val;
    __syncthreads();

    return val;
}


__device__ int block_count = 0;

__global__
void scan_kernel(const int *input, int *output, cuda::std::atomic<int> *sums, const int size, bool INCLUSIVE) {
    __shared__ int bid;
    __shared__ int global_sum;
    extern __shared__ int sdata[];

    const int tid = threadIdx.x;

    if (tid == 0)
        bid = atomicAdd(&block_count, 1);
    __syncthreads();

    if (bid * blockDim.x + tid >= size)
        return;

    int thread_value = input[bid * blockDim.x + tid];
    sdata[tid] = thread_value;
    __syncthreads();

    int val = scan_block(sdata, tid);
    __syncthreads();

    if (bid == 0 && tid == size < blockDim.x ? size - 1 : blockDim.x - 1)
        sums[0].store(val);

    if (bid > 0 && tid == 0) {
        int sum;
        while ((sum = sums[bid - 1].load()) == -1);
        global_sum = sum;
    }
    __syncthreads();

    int result = val + global_sum;
    if (INCLUSIVE)
        output[bid * blockDim.x + tid] = result;
    else
        output[bid * blockDim.x + tid] = result - thread_value;
    __syncthreads();
    if (tid == 0)
        sums[bid].store(result);
}
