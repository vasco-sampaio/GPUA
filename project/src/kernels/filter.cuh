#pragma once


void predicate(int* predicate, const int* buffer, const int size, cudaStream_t& stream);
void scatter(int* buffer, int* output, const int* predicate, const int size, cudaStream_t& stream);
void map(int* buffer, const int size, cudaStream_t& stream);
