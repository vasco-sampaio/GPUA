#pragma once

#include <thrust/device_vector.h>

struct PixelModificator {
    __host__ __device__
    int operator()(const int& x, const int& i) const
    {
        if (i % 4 == 0)
            return x + 1;
        else if (i % 4 == 1)
            return x - 5;
        else if (i % 4 == 2)
            return x + 3;
        else if (i % 4 == 3)
            return x - 8;
        else
            return x;
    }
};


void modify_buffer(thrust::device_vector<int>& input, thrust::device_vector<int>& output);
