#pragma once

typedef bool (*predicate_t)(int);

void predicate(int* predicate, const int* buffer, const int size, cudaStream_t* stream);
void scatter(int* buffer, const int* predicate, const int size, cudaStream_t* stream);
void map(int* buffer, const int size, cudaStream_t* stream);
int find_if(int* buffer, predicate_t fct, const int size, cudaStream_t* stream);
