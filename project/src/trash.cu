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

__device__
int scan_warp(int* data, const int tid) {
    const int lane = tid & 31; // index within the warp

    if (lane >= 1) data[tid] += data[tid - 1]; __syncwarp();
    if (lane >= 2) data[tid] += data[tid - 2]; __syncwarp();
    if (lane >= 4) data[tid] += data[tid - 4]; __syncwarp();
    if (lane >= 8) data[tid] += data[tid - 8]; __syncwarp();
    if (lane >= 16) data[tid] += data[tid - 16]; __syncwarp();

    return data[tid];
}


__device__
int scan_block(int* data, const int tid) {
    const int lane = tid & 31;
    const int warpid = tid >> 5;

    int val = scan_warp(data, tid);
    __syncthreads();

    if(lane == 31) data[warpid] = data[tid]; __syncthreads();

    if(warpid == 0) {
        scan_warp(data, tid); __syncthreads();
        printf("Two first values of the block %d: %d %d\n", blockIdx.x, data[0], data[1]);
    }

    if (warpid > 0) val += data[warpid - 1]; __syncthreads();

    data[tid] = val;
    __syncthreads();

    return val;
}

template <FindType F>
__global__
void find_first_value(const int *data, const int size, const int valueToFind, int *result) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    if (tid < size) {
        if constexpr (F == FindType::SMALLER) {
            if (data[tid] < valueToFind) {
                if (atomicCAS(result, -1, tid) != -1)
                    atomicMin(result, tid);
            }
        }
        else if constexpr (F == FindType::EQUAL) {
            if (data[tid] == valueToFind) {
                if (atomicCAS(result, -1, tid) != -1)
                    atomicMin(result, tid);
            }
        }
        else if constexpr (F == FindType::BIGGER) {
            if (data[tid] > valueToFind) {
                if (atomicCAS(result, -1, tid) != -1)
                    atomicMin(result, tid);
            }
        }
    }
}


template <FindType F>
int find_index(const int* buffer, const int size, const int value, cudaStream_t& stream) {
    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    int* result;
    CUDA_CALL(cudaMalloc(&result, sizeof(int)));
    CUDA_CALL(cudaMemsetAsync(result, -1, sizeof(int), stream));

    find_first_value<F><<<grid_size, block_size, 0, stream>>>(buffer, size, value, result);

    CUDA_CALL(cudaStreamSynchronize(stream));

    int *tmp;
    CUDA_CALL(cudaMallocHost(&tmp, sizeof(int)));
    CUDA_CALL(cudaMemcpyAsync(tmp, result, sizeof(int), cudaMemcpyDeviceToHost, stream));

    int res = *tmp;

    CUDA_CALL(cudaFreeHost(tmp));
    CUDA_CALL(cudaFree(result));

    return res;
}

template int find_index<FindType::SMALLER>(const int*, const int, const int, cudaStream_t&);
template int find_index<FindType::EQUAL>(const int*, const int, const int, cudaStream_t&);
template int find_index<FindType::BIGGER>(const int*, const int, const int, cudaStream_t&);

enum class FindType {
    SMALLER,
    EQUAL,
    BIGGER
};

