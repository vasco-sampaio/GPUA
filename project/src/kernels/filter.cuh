#pragma once


void predicate(int* predicate, const int* buffer, const int size);
void scatter(int* buffer, const int* predicate, const int size);
void map(int* buffer, const int size);

int find_if(int* buffer, const int size, const int value, const bool is_equal);

