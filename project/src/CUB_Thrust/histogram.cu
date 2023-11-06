#include "histogram.cuh"

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/iterator/zip_iterator.h>
#include <iostream>

void computeHistogramEqualization(int* compacted_input, int* equalized_output, int num_elements) {
    thrust::device_ptr<int> compacted_ptr(compacted_input);
    thrust::device_ptr<int> equalized_ptr(equalized_output);


    thrust::device_vector<int> histogram(256, 0);
    thrust::transform(compacted_ptr, compacted_ptr + num_elements, histogram.begin(),
                      thrust::identity<int>());


    thrust::inclusive_scan(histogram.begin(), histogram.end(), histogram.begin());


    int max_cdf = histogram.back();
    thrust::transform(histogram.begin(), histogram.end(), histogram.begin(),
                      NormalizeCDF(max_cdf));


    thrust::transform(compacted_ptr, compacted_ptr + num_elements, equalized_ptr,
                      EqualizeValues(histogram));
}


struct NormalizeCDF : public thrust::unary_function<int, int> {
    int max_cdf;
    NormalizeCDF(int max_cdf) : max_cdf(max_cdf) {}

    __host__ __device__
    int operator()(int cdf) const {
        return (int)((cdf * 255) / max_cdf);
    }
};


struct EqualizeValues {
    thrust::device_vector<int> cdf;

    EqualizeValues(thrust::device_vector<int> cdf) : cdf(cdf) {}

    __host__ __device__
    int operator()(int value) const {
        return cdf[value];
    }
};
