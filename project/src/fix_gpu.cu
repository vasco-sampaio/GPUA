#include "fix_gpu.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "kernels/filter.cuh"
#include "kernels/histogram.cuh"
#include "kernels/scan.cuh"
#include "kernels/utils.cuh"

void fix_image_gpu(int* buffer, const int width, const int height, cudaStream_t* stream)
{
    int image_size = width * height;

    // #1 Compact

    // Build predicate vector
    int* predicate_map;
    CUDA_CALL(cudaMallocManaged(&predicate_map, image_size * sizeof(int)));
    CUDA_CALL(cudaMemset(predicate_map, 0, image_size * sizeof(int)));

    predicate(predicate_map, buffer, image_size, stream);

    // Compute the exclusive sum of the predicate

    int* predicate_scan;
    CUDA_CALL(cudaMallocManaged(&predicate_scan, image_size * sizeof(int)));

    scan<ScanType::EXCLUSIVE>(predicate_map, predicate_scan, image_size, stream);

    // Scatter to the corresponding addresses

    scatter(buffer, predicate_scan, image_size, stream);

    CUDA_CALL(cudaFree(predicate_map));
    CUDA_CALL(cudaFree(predicate_scan));

    /*
    ** Intermediate Result: .buffer will store the good values at the beginning
    ** by overwriting the existing values. We can get the new size of the buffer by
    ** reading the last predicate case
    */

    // #2 Apply map to fix pixels

    image_size = predicate_scan[image_size - 1];
    map(buffer, image_size, stream);

    // #3 Histogram equalization

    // Histogram

    int* histo;
    CUDA_CALL(cudaMallocManaged(&histo, 256 * sizeof(int)));
    CUDA_CALL(cudaMemset(histo, 0, 256 * sizeof(int)));

    histogram(histo, buffer, image_size, stream);

    // Compute the inclusive sum scan of the histogram

    scan<ScanType::INCLUSIVE>(histo, histo, 256, stream);

    // Find the first non-zero value in the cumulative histogram

    int* first_none_zero = std::find_if(histo, histo + 256, [](auto v) { return v != 0; });
    const int cdf_min = *first_none_zero;

    // Apply the map transformation of the histogram equalization
    histogram_equalization(buffer, histo, image_size, cdf_min, 255, stream);

    CUDA_CALL(cudaFree(histo));
}