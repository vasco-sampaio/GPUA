#include "scan.cuh"

#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
        ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 


__device__
void block_scan_kernel(int* input, int* output, int tid, int n) {
    extern __shared__ int sdata[];
    int offset = 1;

    int ai = tid; int bi = tid + (n / 2);
    int bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    int bankOffsetB = CONFLICT_FREE_OFFSET(bi);
    sdata[ai + bankOffsetA] = input[ai];
    sdata[bi + bankOffsetB] = input[bi];

    int init1 = sdata[ai + bankOffsetA];
    int init2 = sdata[bi + bankOffsetB];

    for (int d = n>>1; d > 0; d >>= 1) {
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            sdata[bi] += sdata[ai];
        }
        offset *= 2;
    }

    if (tid==0) 
        sdata[n - 1 + CONFLICT_FREE_OFFSET(n - 1)] = 0;

    for (int d = 1; d < n; d *= 2) {
        offset >>= 1;
        __syncthreads();
        if (tid < d) {
            int ai = offset * (2 * tid + 1) - 1;
            int bi = offset * (2 * tid + 2) - 1;
            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);
            int t = sdata[ai];
            sdata[ai] = sdata[bi];
            sdata[bi] += t;
        }
    }
    __syncthreads();

    output[2 * tid] = sdata[2 * tid] + init1;
    output[2 * tid + 1] = sdata[2 * tid + 1] + init2;
}


// flags : P ready; A local sum ready; X not ready 
__global__
void scan_kernel(int* input, int* output, cuda::std::atomic<char> *flags, int* counter) {
    int bid = atomicAdd(counter, 1);
    int tid = threadIdx.x;

    block_scan_kernel(input + bid, output + bid, tid, blockDim.x / 2);

    flags[bid].store('A');

    if (*counter > 0) {
        int i = *counter - 1;
        char flag;
        do {
            flag = flags[i].load();

            while (flag == 'X');

            output[bid + tid] += output[i * blockDim.x + blockDim.x - 1];

            if (flag == 'P')
                break;

            i -= 1;
        } while (i > 0);

        flags[bid].store('P');
    }
}

void scan(int* input, int* output, int n) {
    int* d_input;
    int* d_output;
    cuda::std::atomic<char>* d_flags;
    int* d_counter;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMalloc(&d_flags, n * sizeof(cuda::std::atomic<char>));
    cudaMemset(&d_flags, 0, n * sizeof(cuda::std::atomic<char>));
    
    cudaMalloc(&d_counter, sizeof(int));
    cudaMemset(&d_counter, -1, sizeof(int));

    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;

    scan_kernel<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_output, d_flags, d_counter);

    cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_flags);
    cudaFree(d_counter);
}
