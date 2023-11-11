#pragma once

#include "image.hh"


void fix_image_gpu(Image& image, cudaStream_t& stream);
