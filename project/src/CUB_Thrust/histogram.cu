#include "histogram.cuh"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/find.h>
#include <cub/cub.cuh>


void histogram_equalization(thrust::device_vector<int>& image, thrust::device_vector<int>& histogram)
{
    // Compute histogram
    int* d_image = thrust::raw_pointer_cast(image.data());
    int* d_histo = thrust::raw_pointer_cast(histogram.data());

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    thrust::device_vector<int>::size_type num_items = image.size();

    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                        d_image, d_histo, 256, 0, 255, num_items);
    cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);


    cub::CachingDeviceAllocator g_allocator(true);
    cub::DoubleBuffer<int> d_keys(d_image, d_histo);
    cub::DoubleBuffer<int> d_values(d_histo, d_image);
    cub::DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys, d_values, num_items);
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                        d_image, d_histo, 256, 0, 255, num_items);

    // Cumulative histogram
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_image, d_histo, 256);


    // Normalize
    thrust::device_vector<int>::const_iterator iter;
    iter = thrust::find_if(histogram.begin(), histogram.end(), IsNotZero());

    NormalizeHistogram normalize(num_items, *iter, d_histo);
    thrust::transform(thrust::counting_iterator<int>(0), thrust::counting_iterator<int>(256), histogram.begin(), normalize);

    cudaFree(d_temp_storage);
}
