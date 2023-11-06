#pragma once

#include "image.hh"

void fix_image_gpu(int* buffer, const int buffer_size, const int image_size, cudaStream_t* stream);
