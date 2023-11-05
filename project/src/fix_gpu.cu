#include "fix_gpu.cuh"

#include <array>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <iostream>

#include "kernels.cuh"


void fix_image_gpu(Image& image)
{
    const int buffer_size = image.size();
    const int image_size = image.width * image.height;

    const int block_size = 256;
    int grid_size = (buffer_size + block_size - 1) / block_size;

    int *d_buffer;
    cudaMalloc(&d_buffer, buffer_size * sizeof(int));
    cudaMemcpy(d_buffer, image.buffer, buffer_size * sizeof(int), cudaMemcpyHostToDevice);
   
    int* predicate;
    cudaMalloc(&predicate, buffer_size * sizeof(int));
    cudaMemset(predicate, 0, buffer_size * sizeof(int));
    
    cuda::std::atomic<int>* sums;
    cudaMalloc(&sums, grid_size * sizeof(cuda::std::atomic<int>));
    cudaMemset(sums, -1, grid_size * sizeof(cuda::std::atomic<int>));

    int* image_buffer;
    cudaMalloc(&image_buffer, image_size * sizeof(int));
    
    std::cout << "Predicate" << std::endl;
    predicate_kernel<<<grid_size, block_size>>>(predicate, d_buffer, buffer_size);

    std::cout << "Scan" << std::endl;
    scan_kernel<<<grid_size, block_size, block_size * sizeof(int)>>>(predicate, predicate, sums, buffer_size, false);

    std::cout << "Scatter" << std::endl;
    scatter_kernel<<<grid_size, block_size>>>(d_buffer, image_buffer, predicate, buffer_size);

    // grid_size = (image_size + block_size - 1) / block_size;
    // map_kernel<<<block_size, grid_size>>>(buffer, image_size);


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

    cudaMemcpy(image.buffer, d_buffer, image_size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_buffer);
    cudaFree(image_buffer);
    cudaFree(predicate);
    cudaFree(sums);
}