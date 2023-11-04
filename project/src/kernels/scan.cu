#include "scan.cuh"

#include <cuda_runtime.h>
#include <cuda/atomic>
#include <iostream>
#include <stdio.h>

#define NEXT_POW_2(x) (1 << (32 - __builtin_clz(x - 1)))
#define PREV_POW_2(x) (1 << (31 - __builtin_clz(x)))

#define CUDA_CALL(x) cudaCheckError((x), __FILE__, __LINE__)

inline cudaError_t cudaCheckError(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
    return result;
}

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
void scan_kernel(const int *input, int *output, cuda::std::atomic<char> *flags, int *counter, int n) {
    __shared__ int bid;
    extern __shared__ int sdata[];

    const int tid = threadIdx.x;

    if (tid == 0)
        bid = atomicAdd(counter, 1);
    __syncthreads();

    if (bid * blockDim.x + tid >= n)
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
void scan/*_pow_2*/(int* input, int* output, int n) {
    int block_size = 64; // std::min(std::max(PREV_POW_2(n), 32), 1024);
    int grid_size = (n + block_size - 1) / block_size;

    cuda::std::atomic<char>* flags;
    int* counter;

    CUDA_CALL(cudaMallocManaged(&flags, grid_size * sizeof(cuda::std::atomic<char>)));
    CUDA_CALL(cudaMemset(flags, 0, grid_size * sizeof(cuda::std::atomic<char>)));

    CUDA_CALL(cudaMallocManaged(&counter, sizeof(int)));
    CUDA_CALL(cudaMemset(counter, 0, sizeof(int)));

    scan_kernel<type><<<grid_size, block_size, block_size * sizeof(int)>>>(input, output, flags, counter, n);

    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaFree(flags));
    CUDA_CALL(cudaFree(counter));
}

template void scan<ScanType::EXCLUSIVE>(int* input, int* output, int n);
template void scan<ScanType::INCLUSIVE>(int* input, int* output, int n);

// Padding the input array to the next power of 2 --> BAD IDEA
//void scan(int* input, int* output, int n) {
//    long pow_2 = NEXT_POW_2(n);
//    if (n == pow_2) {
//        scan_pow_2(input, output, n);
//        return;
//    }
//
//    int prev_pow_2 = PREV_POW_2(n);
//
//    if (n - prev_pow_2 < pow_2 - n) {
//        scan_pow_2(input, output, prev_pow_2);
//
//        int *last = output + prev_pow_2 - 1;
//
//        int rem = n - prev_pow_2;
//        int n_rem = NEXT_POW_2(rem + 1);
//
//        int *rem_ptr;
//        int *output_rem;
//        CUDA_CALL(cudaMalloc(&output_rem, n_rem * sizeof(int)));
//        CUDA_CALL(cudaMalloc(&rem_ptr, n_rem * sizeof(int)));
//
//        CUDA_CALL(cudaMemcpy(rem_ptr, last, sizeof(int), cudaMemcpyDeviceToDevice));
//        CUDA_CALL(cudaMemcpy(rem_ptr + 1, input + prev_pow_2, rem * sizeof(int), cudaMemcpyDeviceToDevice));
//        CUDA_CALL(cudaMemset(rem_ptr + 1 + rem , 0, (n_rem - rem - 1) * sizeof(int)));
//
//        scan_pow_2(rem_ptr, output_rem, n_rem);
//
//        CUDA_CALL(cudaMemcpy(output + prev_pow_2, output_rem + 1, rem * sizeof(int), cudaMemcpyDeviceToDevice));
//
//        CUDA_CALL(cudaFree(rem_ptr));
//        CUDA_CALL(cudaFree(output_rem));
//    } else {
//        int *input_pow_2, *output_pow_2;
//
//        CUDA_CALL(cudaMalloc(&input_pow_2, pow_2 * sizeof(int)));
//        CUDA_CALL(cudaMalloc(&output_pow_2, pow_2 * sizeof(int)));
//
//        CUDA_CALL(cudaMemcpy(input_pow_2, input, n * sizeof(int), cudaMemcpyDeviceToDevice));
//        CUDA_CALL(cudaMemset(input_pow_2 + n, 0, (pow_2 - n) * sizeof(int)));
//
//        scan_pow_2(input_pow_2, output_pow_2, pow_2);
//
//        CUDA_CALL(cudaMemcpy(output, output_pow_2, n * sizeof(int), cudaMemcpyDeviceToDevice));
//
//        CUDA_CALL(cudaFree(input_pow_2));
//        CUDA_CALL(cudaFree(output_pow_2));
//    }
//}