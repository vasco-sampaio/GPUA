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

void fix_image_gpu(int* buffer, const int width, const int height, cudaStream_t* stream)
{
    int image_size = width * height;

    // #1 Compact
    // Build predicate vector
    int* predicate_scan;
    CUDA_CALL(cudaMallocManaged(&predicate_scan, image_size * sizeof(int)));
    CUDA_CALL(cudaMemset(predicate_scan, 0, image_size * sizeof(int)));
    std::cout << "Buffer: " << std::endl;
    // print_array(buffer, 50, true);

    predicate(predicate_scan, buffer, image_size, stream);
    std::cout << "Predicate: " << std::endl;
    print_array(predicate_scan, 50);

    // Compute the exclusive sum of the predicate
    scan<ScanType::EXCLUSIVE>(predicate_scan, predicate_scan, image_size, stream);
    std::cout << "Scan: " << std::endl;
    print_array(predicate_scan, 50);

    // Scatter to the corresponding addresses
    scatter(buffer, predicate_scan, image_size, stream);
    std::cout << "Scatter: " << std::endl;
    // print_array(buffer, 50, true);

    // #2 Apply map to fix pixels
    // std::cout << "Old image size: " << image_size << "\tNew image size: " << predicate_scan[image_size - 1] << std::endl;
    const int new_image_size = predicate_scan[image_size - 1];
    CUDA_CALL(cudaFree(predicate_scan));

    // Verify that there are no -27 left
    int first_minus_27 = find_if(buffer, [](auto v) { return v == -27; }, new_image_size, stream);
    if (first_minus_27 != -1)
        std::cout << "There are still -27 in the image: " << first_minus_27 << std::endl;


//    map(buffer, new_image_size, stream);
//
//    // #3 Histogram equalization
//    // Histogram
//    int* histo;
//    CUDA_CALL(cudaMallocManaged(&histo, 256 * sizeof(int)));
//    CUDA_CALL(cudaMemset(histo, 0, 256 * sizeof(int)));
//
//    histogram(histo, buffer, new_image_size, stream);
//
//    // Compute the inclusive sum scan of the histogram
//    scan<ScanType::INCLUSIVE>(histo, histo, 256, stream);
//
//    // Find the first non-zero value in the cumulative histogram
//    int first_none_zero = find_if(histo, [](auto v) { return v != 0; }, 256, stream);
//
//    // Apply the map transformation of the histogram equalization
//    histogram_equalization(buffer, histo, new_image_size, histo[first_none_zero], 255, stream);
//
//    CUDA_CALL(cudaFree(histo));
}