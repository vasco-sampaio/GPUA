#pragma once

#include <iostream>

#define CUDA_CALL(x) cudaCheckError((x), __FILE__, __LINE__)

inline cudaError_t cudaCheckError(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(result) << std::endl;
        exit(EXIT_FAILURE);
    }
    return result;
}


#define NEXT_POW_2(x) (1 << (32 - __builtin_clz(x - 1)))
#define PREV_POW_2(x) (1 << (31 - __builtin_clz(x)))
#define BLOCK_SIZE(x) std::min(std::max(PREV_POW_2(x), 32), 1024)
