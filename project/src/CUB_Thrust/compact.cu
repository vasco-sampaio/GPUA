#include "compact.cuh"

#include <cub/cub.cuh>


struct is_positive
{
    __host__ __device__
    bool operator()(const int& x) const
    {
        return x >= 0;
    }
};


void compact_scan(thrust::device_vector<int>& input, thrust::device_vector<int>& output)
{
    int* d_input = thrust::raw_pointer_cast(input.data());
    int* d_output = thrust::raw_pointer_cast(output.data());

    void* d_temp_storage = nullptr;
    size_t temp_storage_bytes = 0;

    int num_items = input.size();
    int num_selected = 0;

    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_input, d_output, &num_selected, num_items, is_positive());
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_input, d_output, &num_selected, num_items, is_positive());

    cudaFree(d_temp_storage);
}