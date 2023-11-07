#include "histogram.cuh"

#include <cub/device/device_histogram.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_scan.cuh>
#include <thrust/transform.h>

void histogram_equalization(thrust::device_vector<int>& image, thrust::device_vector<int>& histogram)
{
    // Compute histogram
    int* d_image = thrust::raw_pointer_cast(image.data());
    int* d_histo = thrust::raw_pointer_cast(histogram.data());

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    int num_items = image.size();

    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                        d_image, d_histo, 256, 0, 255, num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceHistogram::HistogramEven(d_temp_storage, temp_storage_bytes,
                                        d_image, d_histo, 256, 0, 255, num_items);

    // Compute Inclusive Scan
    cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes,
                                  d_image, d_histo, 256);


    // Normalize
    thrust::device_vector<int>::iterator iter;
    iter = thrust::find_if(histogram.begin(), histogram.end(), IsNotZero());

    NormalizeHistogram normalize(num_items, *iter, d_histo);
    thrust::transform(thrust::make_counting_iterator(0), thrust::make_counting_iterator(256), histogram.begin(), normalize);

    cudaFree(d_temp_storage);
}
