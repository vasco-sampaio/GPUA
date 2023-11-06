#include "fix_gpu.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "kernels/scan.cuh"
#include "kernels/filter.cuh"
#include "kernels/histogram.cuh"

void print_buffer(int* buffer, const int size, bool copy)
{
    if (copy)
    {
        int* host_buffer = new int[size];
        cudaMemcpy(host_buffer, buffer, size * sizeof(int), cudaMemcpyDeviceToHost);
        buffer = host_buffer;
    }
    std::cout << "[";
    for (int i = 0; i < size; i++)
        std::cout << buffer[i] << ", ";
    std::cout << "]" << std::endl;
}


void fix_image_gpu(int* buffer, const int buffer_size, const int image_size)
{
    int* predicate_buffer;
    cudaMalloc(&predicate_buffer, buffer_size * sizeof(int));
    cudaMemset(predicate_buffer, 0, buffer_size * sizeof(int));

    predicate(predicate_buffer, buffer, buffer_size);

    scan<ScanType::EXCLUSIVE>(predicate_buffer, predicate_buffer, buffer_size);

    int* image_buffer;
    cudaMalloc(&image_buffer, image_size * sizeof(int));

    scatter(buffer, image_buffer, predicate_buffer, buffer_size);

    print_buffer(image_buffer, 50, true);

    cudaFree(predicate_buffer);
    cudaMemcpy(buffer, image_buffer, image_size * sizeof(int), cudaMemcpyDeviceToDevice);
    cudaFree(image_buffer);
}
