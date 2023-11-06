#include "fix_gpu.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "kernels/scan.cuh"
#include "kernels/filter.cuh"
#include "kernels/histogram.cuh"


#include <iostream>

void fix_image_gpu(int* buffer, const int buffer_size, const int image_size, cudaStream_t* stream)
{
    int* predicate_buffer;
    cudaMalloc(&predicate_buffer, buffer_size * sizeof(int));
    cudaMemset(predicate_buffer, 0, buffer_size * sizeof(int));

    predicate(predicate_buffer, buffer, buffer_size, stream);

    scan(predicate_buffer, predicate_buffer, buffer_size, stream, false);

    int* image_buffer;
    cudaMalloc(&image_buffer, image_size * sizeof(int));

    scatter(buffer, image_buffer, predicate_buffer, buffer_size, stream);

    cudaFree(predicate_buffer);

    cudaMemcpyAsync(buffer, image_buffer, image_size * sizeof(int), cudaMemcpyDeviceToDevice, *stream);

    cudaFree(image_buffer);
}
