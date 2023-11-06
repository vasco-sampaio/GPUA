#pragma once

#include <iostream>
#include <stdio.h>

#define CUDA_CALL(x) cudaCheckError((x), __FILE__, __LINE__)

inline cudaError_t cudaCheckError(cudaError_t result, const char *file, int line) {
    if (result != cudaSuccess) {
        std::cerr << "CUDA error at " << file << ":" << line << ": " << cudaGetErrorString(result) << std::endl;
        cudaDeviceReset();                                                              \
        exit(EXIT_FAILURE);
    }
    return result;
}


#define NEXT_POW_2(x) (1 << (32 - __builtin_clz(x - 1)))
#define PREV_POW_2(x) (1 << (31 - __builtin_clz(x)))
#define BLOCK_SIZE(x) std::min(std::max(PREV_POW_2(x), 32), 1024)


inline void print_array(const int* array, const int size, std::string title, bool copy = false) {
    printf("--------------------------------------- %s ---------------------------------------\n", title.c_str());
    if (copy) {
        int* tmp = new int[size];
        CUDA_CALL(cudaMemcpy(tmp, array, size * sizeof(int), cudaMemcpyDeviceToHost));
        array = tmp;
        for (int i = 0; i < size; i++) {
            printf("%d ", array[i]);
        }
        printf("\n");
        delete[] tmp;
        return;
    }

    for (int i = 0; i < size; i++) {
        printf("%d ", array[i]);
    }
    printf("\n");
}

inline void print_memory_info() {
    size_t free_byte;
    size_t total_byte;
    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
    if (cudaSuccess != cuda_status) {
        printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
        exit(1);
    }

    double free_db = (double) free_byte;
    double total_db = (double) total_byte;
    double used_db = total_db - free_db;
    printf("GPU memory usage: used = %f MB, free = %f MB, total = %f MB\n",
           used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
}
