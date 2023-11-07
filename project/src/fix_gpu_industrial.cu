#include "fix_gpu_industrial.cuh"

#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/transform.h>

#include "CUB_Thrust/compact.cuh"
#include "CUB_Thrust/map.cuh"
#include "CUB_Thrust/histogram.cuh"


void fix_gpu_industrial(Image& image) {

    const int buffer_size = image.size();
    const int image_size = image.width * image.height;

    // #1 Compact
    thrust::device_vector<int> input;
    thrust::copy(image.buffer, image.buffer + buffer_size, input.begin());

    thrust::device_vector<int> output(image_size, 0);

    compact_scan(input, output);

    // #2 Apply map to fix pixels
    modify_buffer(output, output);

    // #3 Histogram equalization
    thrust::device_vector<int> histogram(256, 0);
    histogram_equalization(output, histogram);
  
    thrust::copy(output.begin(), output.end(), image.buffer);
}
