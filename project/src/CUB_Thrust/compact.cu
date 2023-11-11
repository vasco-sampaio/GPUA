#include "compact.cuh"

#include <cub/cub.cuh>


struct DifferentThan
{
    int compare;

    CUB_RUNTIME_FUNCTION __forceinline__
    DifferentThan(int compare) : compare(compare) {}

    CUB_RUNTIME_FUNCTION __forceinline__
    bool operator()(const int &a) const {
        return (a != compare);
    }
};

void compact_scan(int* d_input, int* d_output, const int size)
{
    void     *d_temp_storage = NULL;
    size_t   temp_storage_bytes = 0;
    int      *d_num_selected_out = nullptr;

    cudaMalloc(&d_num_selected_out, sizeof(int));
    cudaMemset(d_num_selected_out, 0, sizeof(int));

    DifferentThan select_op(-27);

    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_input, d_output, d_num_selected_out, size, select_op);

    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  
    cub::DeviceSelect::If(d_temp_storage, temp_storage_bytes, d_input, d_output, d_num_selected_out, size, select_op);

    cudaFree(d_temp_storage);
    cudaFree(d_num_selected_out);
}