#pragma once

void histogram(int* histogram, const int* buffer, const int size, cudaStream_t& stream);
void histogram_equalization(int* buffer, const int* histogram, const int size, const int cdf_min_idx, cudaStream_t& stream);
