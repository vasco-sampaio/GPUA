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


void fix_image_gpu(int* buffer, const int buffer_size, const int image_size)
{
    // #1 Compact
    // Build predicate vector
    int* predicate_scan;
    CUDA_CALL(cudaMallocManaged(&predicate_scan, buffer_size * sizeof(int)));
    CUDA_CALL(cudaMemset(predicate_scan, 0, buffer_size * sizeof(int)));

    predicate(predicate_scan, buffer, buffer_size);

    // Compute the exclusive sum of the predicate
    scan<ScanType::EXCLUSIVE>(predicate_scan, predicate_scan, buffer_size);

    // Scatter to the corresponding addresses
    scatter(buffer, predicate_scan, buffer_size);

    CUDA_CALL(cudaFree(predicate_scan));

    // #2 Apply map to fix pixels

    // Verify that there are no -27 left
    int first_minus_27 = find_if(buffer, image_size, -27, true);
    if (first_minus_27 != image_size)
        std::cout << "There are still -27 in the image: " << first_minus_27 << std::endl;

    map(buffer, image_size);


    // #3 Histogram equalization
    // Histogram
    // int* histo;
    // CUDA_CALL(cudaMallocManaged(&histo, 256 * sizeof(int)));
    // CUDA_CALL(cudaMemset(histo, 0, 256 * sizeof(int)));

    // std::cout << "Histo: " << std::endl;
    // histogram(histo, buffer, new_image_size);
    // print_array(histo, 50);

    // Compute the inclusive sum scan of the histogram
//    scan<ScanType::INCLUSIVE>(histo, histo, 256);
//
//    // Find the first non-zero value in the cumulative histogram
//    int first_none_zero = find_if(histo, [](int v) { return v != 0; }, 256);
//
//    // Apply the map transformation of the histogram equalization
//    histogram_equalization(buffer, histo, new_image_size, histo[first_none_zero], 255);
//
//    CUDA_CALL(cudaFree(histo));
}