#include "fix_gpu.cuh"

#include "cuda_streams.cuh"
#include "kernels/filter.cuh"
#include "kernels/histogram.cuh"
#include "kernels/scan.cuh"
#include "kernels/utils.cuh"


void fix_image_gpu(Image& image, cudaStream_t& stream) {
    const int image_size = image.width * image.height;
    const int buffer_size = image.size();

    int* d_buffer;
    int* d_predicate_buffer;
    int* d_histo;

    CUDA_CALL(cudaMalloc(&d_buffer, buffer_size * sizeof(int)));
    CUDA_CALL(cudaMalloc(&d_predicate_buffer, buffer_size * sizeof(int)));

    CUDA_CALL(cudaMemcpyAsync(d_buffer, image.buffer, buffer_size * sizeof(int), cudaMemcpyHostToDevice, stream));
    CUDA_CALL(cudaMemsetAsync(d_predicate_buffer, 0, buffer_size * sizeof(int), stream));

    predicate(d_predicate_buffer, d_buffer, buffer_size, stream);
    scan<ScanType::EXCLUSIVE>(d_predicate_buffer, d_predicate_buffer, buffer_size, stream);
    scatter(d_buffer, d_buffer, d_predicate_buffer, buffer_size, stream);
    
    CUDA_CALL(cudaFree(d_predicate_buffer)); // stuck here


    map(d_buffer, image_size, stream);

    CUDA_CALL(cudaMalloc(&d_histo, 256 * sizeof(int)));
    CUDA_CALL(cudaMemsetAsync(d_histo, 0, 256 * sizeof(int), stream));

    histogram(d_histo, d_buffer, image_size, stream);
    scan<ScanType::INCLUSIVE>(d_histo, d_histo, 256, stream);

    int host_histo[256];
    CUDA_CALL(cudaMemcpyAsync(host_histo, d_histo, 256 * sizeof(int), cudaMemcpyDeviceToHost, stream));

    int* first_non_zero = std::find_if(host_histo, host_histo + 256, [](int val) { return val != 0; });
    histogram_equalization(d_buffer, d_histo, image_size, *first_non_zero, stream);

    CUDA_CALL(cudaMemcpyAsync(image.buffer, d_buffer, image_size * sizeof(int), cudaMemcpyDeviceToHost, stream));

    CUDA_CALL(cudaFree(d_buffer));
    CUDA_CALL(cudaFree(d_histo));
}