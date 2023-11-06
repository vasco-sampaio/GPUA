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


void fix_image_gpu(int* buffer, const int buffer_size, const int image_size, cudaStream_t* stream)
{
    // #1 Compact
    // Build predicate vector
    int* predicate_buffer;
    CUDA_CALL(cudaMallocManaged(&predicate_buffer, buffer_size * sizeof(int)));
    CUDA_CALL(cudaMemset(predicate_buffer, 0, buffer_size * sizeof(int)));
    predicate(predicate_buffer, buffer, buffer_size, stream);

    // Compute the exclusive sum of the predicate
    scan<ScanType::EXCLUSIVE>(predicate_buffer, predicate_buffer, buffer_size, stream);

    // Scatter to the corresponding addresses
    int* image_buffer;
    cudaMalloc(&image_buffer, image_size * sizeof(int));
    scatter(buffer, image_buffer, predicate_buffer, buffer_size, stream);

    CUDA_CALL(cudaFree(predicate_buffer));

    int minus_27 = find_index<FindType::EQUAL>(image_buffer, image_size, -27, stream);
    std::cout << "minus_27: " << minus_27  << "image_size: " << image_size << std::endl;

    // #2 Apply map to fix pixels
    map(image_buffer, image_size, stream);

    // #3 Histogram equalization
    int* histo;
    CUDA_CALL(cudaMalloc(&histo, 256 * sizeof(int)));
    CUDA_CALL(cudaMemset(histo, 0, 256 * sizeof(int)));

    histogram(histo, image_buffer, image_size, stream);
    scan<ScanType::INCLUSIVE>(histo, histo, 256, stream);
    int first_none_zero = find_index<FindType::BIGGER>(histo, 256, 0, stream);
    histogram_equalization(image_buffer, histo, image_size, first_none_zero, stream);


    CUDA_CALL(cudaFree(histo));

    CUDA_CALL(cudaMemcpy(buffer, image_buffer, image_size * sizeof(int), cudaMemcpyDeviceToDevice));
    CUDA_CALL(cudaFree(image_buffer));
}