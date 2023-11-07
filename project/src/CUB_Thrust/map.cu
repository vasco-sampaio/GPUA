#include "map.cuh"

#include <cub/device/device_transform.cuh>


void modify_buffer(thrust::device_vector<int>& input, thrust::device_vector<int>& output)
{
    int* d_input = thrust::raw_pointer_cast(input.data());
    int* d_output = thrust::raw_pointer_cast(output.data());

    int num_items = input.size();

    void* d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;

    cub::DeviceTransform::Transform(d_temp_storage, temp_storage_bytes, d_input, d_output, PixelModificator(), num_items);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceTransform::Transform(d_temp_storage, temp_storage_bytes, d_input, d_output, PixelModificator(), num_items);

    cudaFree(d_temp_storage);
}
