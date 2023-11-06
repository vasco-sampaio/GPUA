#include "fix_gpu.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "kernels/filter.cuh"
#include "kernels/histogram.cuh"
#include "kernels/scan.cuh"
#include "kernels/utils.cuh"

void print_array(int* array, int size, bool copy = false) {
    if (copy) {
        int* tmp = (int *)malloc(size * sizeof(int));
        cudaMemcpy(tmp, array, size * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < size; ++i)
            std::cout << tmp[i] << " ";
        std::cout << std::endl;

        return;
    }
    for (int i = 0; i < size; ++i)
        std::cout << array[i] << " ";
    std::cout << std::endl;
};

void fix_image_gpu(int* buffer, const int buffer_size, const int image_size, cudaStream_t* stream)
{
    // #1 Compact
    // Build predicate vector
    int* predicate_buffer;
    CUDA_CALL(cudaMallocManaged(&predicate_buffer, buffer_size * sizeof(int)));
    CUDA_CALL(cudaMemset(predicate_buffer, 0, buffer_size * sizeof(int)));
    // std::cout << "Buffer: " << std::endl;
    // print_array(buffer, 50, true);

    predicate(predicate_buffer, buffer, buffer_size, stream);
//    std::cout << "Predicate: " << std::endl;
//    print_array(predicate_buffer, 50);

    // Compute the exclusive sum of the predicate
    scan<ScanType::EXCLUSIVE>(predicate_buffer, predicate_buffer, buffer_size, stream);
//    std::cout << "Scan: " << std::endl;
//    print_array(predicate_buffer, 50);

    // Scatter to the corresponding addresses
    int* image_buffer;
    cudaMalloc(&image_buffer, image_size * sizeof(int));

    scatter(buffer, image_buffer, predicate_buffer, buffer_size, stream);
//    std::cout << "Scatter: " << std::endl;
//    print_array(image_buffer, 50, true);

    CUDA_CALL(cudaFree(predicate_buffer));

    // #2 Apply map to fix pixels

    map(image_buffer, image_size, stream);
//    std::cout << "Map: " << std::endl;
//    print_array(image_buffer, 50, true);

    // #3 Histogram equalization
    // Histogram

    int* histo;
    CUDA_CALL(cudaMalloc(&histo, 256 * sizeof(int)));
    CUDA_CALL(cudaMemset(histo, 0, 256 * sizeof(int)));

    histogram(histo, image_buffer, image_size, stream);
//    std::cout << "Histogram: " << std::endl;
//    print_array(histo, 256, true);

    // Compute the inclusive sum scan of the histogram

    scan<ScanType::INCLUSIVE>(histo, histo, 256, stream);
//    std::cout << "Scan: " << std::endl;
//    print_array(histo, 256, true);

    // Find the first non-zero value in the cumulative histogram
    int first_none_zero = find_index<FindType::BIGGER>(histo, 256, 0, stream);
//    std::cout << "First non zero: " << first_none_zero << std::endl;

    // Apply the map transformation of the histogram equalization
    histogram_equalization(image_buffer, histo, image_size, first_none_zero, stream);
    std::cout << "Histogram equalization: " << std::endl;
    print_array(image_buffer, 50, true);

    CUDA_CALL(cudaFree(histo));

    // #4 Copy back to host
    CUDA_CALL(cudaMemcpy(buffer, image_buffer, image_size * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaFree(image_buffer));
}