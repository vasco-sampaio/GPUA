#pragma once


void predicate(int* predicate_buffer, const int* buffer, const int size);

void scatter(int* buffer, int* output, const int* predicate, const int size);

void map(int* buffer, const int size);
