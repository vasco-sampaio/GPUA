#pragma once

enum class ScanType
{
    EXCLUSIVE,
    INCLUSIVE
};

template <ScanType type>
void scan(int* input, int* output, const int size, cudaStream_t* stream);
