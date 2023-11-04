#pragma once

#include "image.hh"

void fix_image_gpu(int* buffer, const int width, const int height, cudaStream_t* stream);
