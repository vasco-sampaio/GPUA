#pragma once

enum class FindType {
    SMALLER,
    EQUAL,
    BIGGER
};


void predicate(int* predicate, const int* buffer, const int size, cudaStream_t& stream);
void scatter(int* buffer, int* output, const int* predicate, const int size, cudaStream_t& stream);
void map(int* buffer, const int size, cudaStream_t& stream);

template <FindType F>
int find_index(const int* buffer, const int size, const int value, cudaStream_t& stream);
