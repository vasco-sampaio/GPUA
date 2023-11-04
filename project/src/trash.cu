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