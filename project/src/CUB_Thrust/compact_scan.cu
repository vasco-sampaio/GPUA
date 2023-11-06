#include "compact_scan.cuh"

struct PredicateEqual {
    int predicate_value;

    PredicateEqual(int predicate_value) : predicate_value(predicate_value) {}

    __host__ __device__
    bool operator()(const int& value) const {
        return value == predicate_value;
    }
};

void compactScan(int* input, int* predicate, int num_elements) {
    thrust::device_ptr<int> input_ptr(input);
    thrust::device_ptr<int> predicate_ptr(predicate);

    thrust::transform(predicate_ptr, predicate_ptr + num_elements, predicate_ptr,
                      input_ptr, thrust::identity<int>());

    thrust::device_vector<int> scan_result(num_elements);
    thrust::exclusive_scan(predicate_ptr, predicate_ptr + num_elements, scan_result.begin());

    int predicate_value = -27;
    PredicateEqual pred_equal(predicate_value);
    thrust::copy_if(input_ptr, input_ptr + num_elements, scan_result.begin(),
                    input_ptr, pred_equal);
}
