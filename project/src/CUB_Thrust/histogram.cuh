#pragma once

#include <thrust/device_vector.h>


struct IsNotZero
{
    __host__ __device__
    bool operator()(int x)
    {
        return x != 0;
    }
};


struct NormalizeHistogram
{
    int image_size;
    float cdf_min;
    int* histo;

    __host__ __device__
    NormalizeHistogram(int image_size, float cdf_min, int* histo) : image_size(image_size), cdf_min(cdf_min), histo(histo) {}

    __host__ __device__
    int operator()(int pixel) const
    {
        return static_cast<int>(std::roundf(((histo[pixel] - cdf_min) / static_cast<float>(image_size - cdf_min)) * 255.0f));
    }
};


void histogram_equalization(thrust::device_vector<int>& image, thrust::device_vector<int>& histogram);
