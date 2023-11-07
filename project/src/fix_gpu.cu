#include "fix_gpu.cuh"

#include "kernels/filter.cuh"
#include "kernels/scan.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>


void fix_image_gpu(Image& image) {
    const int image_size = image.width * image.height;
    const int buffer_size = image.size();

    int* d_buffer;
    int* d_predicate_buffer;

    cudaMalloc(&d_buffer, buffer_size * sizeof(int));
    cudaMalloc(&d_predicate_buffer, buffer_size * sizeof(int));

    cudaMemcpy(d_buffer, image.buffer, buffer_size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_predicate_buffer, 0, buffer_size * sizeof(int));

    predicate(d_predicate_buffer, d_buffer, buffer_size);
    scan(d_predicate_buffer, d_predicate_buffer, buffer_size);

    cudaFree(d_buffer);

    int host_predicate[buffer_size];
    cudaMemcpy(host_predicate, d_predicate_buffer, buffer_size * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_predicate_buffer);

    for (std::size_t i = 0; i < buffer_size; ++i)
        if (image.buffer[i] != -27)
            image.buffer[host_predicate[i]] = image.buffer[i];


    for (int i = 0; i < image_size; ++i)
    {
        if (i % 4 == 0)
            image.buffer[i] += 1;
        else if (i % 4 == 1)
            image.buffer[i] -= 5;
        else if (i % 4 == 2)
            image.buffer[i] += 3;
        else if (i % 4 == 3)
            image.buffer[i] -= 8;
    }

    // #3 Histogram equalization

    // Histogram

    std::array<int, 256> histo;
    histo.fill(0);
    for (int i = 0; i < image_size; ++i)
        ++histo[image.buffer[i]];

    // Compute the inclusive sum scan of the histogram

    std::inclusive_scan(histo.begin(), histo.end(), histo.begin());

    // Find the first non-zero value in the cumulative histogram

    auto first_none_zero = std::find_if(histo.begin(), histo.end(), [](auto v) { return v != 0; });
    
    const int cdf_min = *first_none_zero;

    // Apply the map transformation of the histogram equalization

    std::transform(image.buffer, image.buffer + image_size, image.buffer,
                   [image_size, cdf_min, &histo](int pixel)
                   {
                       return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
                   }
    );
}