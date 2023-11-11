#include "histogram.cuh"

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/find.h>


struct IsNonZero {
    __host__ __device__
    bool operator()(const int& x) const
    {   
        return x != 0;
    }
};


struct EqualizePixel
{
    int* histo;
    int image_size;
    int cdf_min;

    EqualizePixel(int* histo, int image_size, int cdf_min)
        : histo(histo), image_size(image_size), cdf_min(cdf_min) {}

    __host__ __device__
    int operator()(const int& pixel) const {
        return std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f);
    }
};    


void histogram_equalization(int* d_image, int* d_histogram, const int size) {
    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                        d_image, d_histogram, 257, 0, 256, size);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                        d_image, d_histogram, 257, 0, 256, size);

    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_histogram, d_histogram, 256);

    cudaFree(d_temp_storage);

    thrust::device_vector<int> d_histo_vec(d_histogram, d_histogram + 256);

    thrust::device_vector<int>::iterator iter;
    iter = thrust::find_if(d_histo_vec.begin(), d_histo_vec.end(), IsNonZero());

    int cdf_min = *iter;

    thrust::device_vector<int> d_image_vec(d_image, d_image + size);

    thrust::transform(d_image_vec.begin(), d_image_vec.end(), d_image_vec.begin(), EqualizePixel(thrust::raw_pointer_cast(d_histo_vec.data()), size, cdf_min));

    thrust::copy(d_image_vec.begin(), d_image_vec.end(), d_image);
}