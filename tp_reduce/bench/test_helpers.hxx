#include "test_helpers.hh"

#include "cuda_tools/host_shared_ptr.cuh"

#include <iostream>
#include <algorithm>
#include <benchmark/benchmark.h>

template <typename T>
void check_buffer(cuda_tools::host_shared_ptr<int> buffer,
                  T expected,
                  benchmark::State& st)
{
    int* host_buffer = buffer.download();

    if (host_buffer[0] != expected)
    {
        std::cout << "Expected " << expected << ", got " << host_buffer[0] << std::endl;
        st.SkipWithError("Failed test");
    }
}

void check_buffer(cuda_tools::host_shared_ptr<int> buffer,
                  cuda_tools::host_shared_ptr<int> expected,
                  benchmark::State& st)
{
    int* host_buffer = buffer.download();

    if (!std::equal(host_buffer,
                    host_buffer + buffer.size_,
                    expected.host_data_))
    {
        auto [first, second] = std::mismatch(host_buffer,
                                             host_buffer + buffer.size_,
                                             expected.host_data_);
        std::cout << "Error at " << first - host_buffer << ": " << *first << " "
                  << *second << std::endl;
        st.SkipWithError("Failed test");
    }
}