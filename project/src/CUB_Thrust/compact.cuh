#pragma once

#include <thrust/device_vector.h>

void compact_scan(thrust::device_vector<int>& input, thrust::device_vector<int>& output);
