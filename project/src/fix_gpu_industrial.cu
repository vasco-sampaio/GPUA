#include "fix_gpu_industrial.cuh"

#include "CUB_Thrust/compact.cuh"
#include "CUB_Thrust/map.cuh"
#include "CUB_Thrust/histogram.cuh"


void fix_gpu_industrial(Image& image) {
    const int buffer_size = image.size();
    const int image_size = image.width * image.height;

    // #1 Compact
    int* d_input;
    int *d_output;
    cudaMalloc(&d_input, buffer_size * sizeof(int));
    cudaMalloc(&d_output, buffer_size * sizeof(int));

    cudaMemcpy(d_input, image.buffer, buffer_size * sizeof(int), cudaMemcpyHostToDevice);

    compact_scan(d_input, d_output, buffer_size);

    // #2 Apply map to fix pixels
    modify_buffer(d_output, d_output, image_size);

    // #3 Histogram
    int* histogram;
    cudaMalloc(&histogram, 257 * sizeof(int)); // CUB requires 257 bins, otherwise the last bin is not computed
    cudaMemset(histogram, 0, 257 * sizeof(int));

    histogram_equalization(d_output, histogram, image_size);

    cudaMemcpy(image.buffer, d_output, image_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}
