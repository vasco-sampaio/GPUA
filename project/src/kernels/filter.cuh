#pragma once


void predicate(int* predicate, const int* buffer, const int size, cudaStream_t* stream);
void scatter(int* buffer, const int* predicate, const int size, cudaStream_t* stream);
void map(int* buffer, const int size, cudaStream_t* stream);

int find_if(int* buffer, const int size, const int value, const bool is_equal, cudaStream_t* stream);

