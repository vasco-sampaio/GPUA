#include "fix_gpu.cuh"

#include "kernels/filter.cuh"
#include "kernels/scan.cuh"
#include "kernels/histogram.cuh"
#include "kernels/reduce.cuh"

void fix_image_gpu(Image& image, cudaStream_t& stream) {
    const int image_size = image.width * image.height;
    const int buffer_size = image.size();

    int* d_buffer;
    int* d_predicate_buffer;

    CUDA_CALL(cudaMalloc(&d_buffer, buffer_size * sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_predicate_buffer, buffer_size * sizeof(int)));

    CUDA_CALL(cudaMemcpyAsync(d_buffer, image.buffer, buffer_size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMemsetAsync(d_predicate_buffer, 0, buffer_size * sizeof(int), stream));

    // #1 Compact

    predicate(d_predicate_buffer, d_buffer, buffer_size, stream);
    scan(d_predicate_buffer, d_predicate_buffer, buffer_size, stream);

    scatter(d_buffer, d_buffer, d_predicate_buffer, buffer_size, stream);

    CUDA_CALL(cudaFree(d_predicate_buffer));


    // #2 Map to fix pixels

    map(d_buffer, image_size, stream);


    // #3 Histogram equalization

    // Histogram

    int *d_histo;
    CUDA_CALL(cudaMalloc(&d_histo, 256 * sizeof(int)));
    CUDA_CALL(cudaMemsetAsync(d_histo, 0, 256 * sizeof(int), stream));

    const int cdf_min_idx = histogram(d_histo, d_buffer, image_size, stream);

    // Compute the inclusive sum scan of the histogram

    scan(d_histo, d_histo, 256, stream);

    // Apply the map transformation of the histogram equalization

    histogram_equalization(d_buffer, d_histo, image_size, cdf_min_idx, stream);

    CUDA_CALL(cudaMemcpyAsync(image.buffer, d_buffer, image_size * sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CALL(cudaFree(d_histo));

    // #4 Compute total (buffer is already on the device)
    image.to_sort.total = reduce(d_buffer, image_size, stream);

    CUDA_CALL(cudaFree(d_buffer));
}