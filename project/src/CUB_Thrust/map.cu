#include "map.cuh"

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/device_ptr.h>


struct PixelModificator {
    __host__ __device__
    int operator()(const int& x, const int& i) const {
        if (i % 4 == 0) return x + 1;
        else if (i % 4 == 1) return x - 5;
        else if (i % 4 == 2) return x + 3;
        else if (i % 4 == 3) return x - 8;
        else return x;
    }
};


void modify_buffer(int* d_input, int* d_output, const int size) {
    thrust::device_ptr<int> input(d_input);
    thrust::device_ptr<int> output(d_output);

    thrust::counting_iterator<int> indices(0);

    PixelModificator pixel_mod;

    thrust::transform(input, input + size, indices, output, pixel_mod);
}
