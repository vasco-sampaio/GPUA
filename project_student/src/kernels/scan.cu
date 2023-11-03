#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
#define CONFLICT_FREE_OFFSET(n) \
        ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS)) 


__global__ void scan_kernel(int* input, int* output, int n) {
    extern __shared__ int sdata[];
    int tid = threadIdx.x;
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

void scan(int* input, int* output, int n) {
    int* d_input;
    int* d_output;

    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));

    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);

    int block_size = 32; // n / 2
    int grid_size = (n + block_size - 1) / block_size;

    scan_kernel<<<grid_size, block_size, block_size * sizeof(int)>>>(d_input, d_output, n);

    cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
