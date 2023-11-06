#pragma once

enum ScanType {
    EXCLUSIVE,
    INCLUSIVE
};

template <ScanType T>
void scan(int* input, int* output, const int size);
