#pragma once

#include <benchmark/benchmark.h>


class Fixture
{
  public:
    static bool no_check;

    template <typename FUNC, typename... Args>
    void bench_reduce(benchmark::State& st,
                      FUNC callback,
                      int size,
                      Args&&... args)
    {
        constexpr int val = 1;
        cuda_tools::host_shared_ptr<int> buffer(size);
        cuda_tools::host_shared_ptr<int> total(1);

        buffer.device_fill(val);

        for (auto _ : st)
        {
            st.PauseTiming();
            total.device_fill(0);
            st.ResumeTiming();
            callback(buffer, total);
        }

        st.SetBytesProcessed(int64_t(st.iterations()) *
                             int64_t(size * sizeof(int)));

        if (!no_check)
            check_buffer(total, size, st);
    }

    template <typename FUNC>
    void register_reduce(benchmark::State &st, FUNC func)
    {
        int size = st.range(0);
        this->bench_reduce(st, func, size);
    }
};

bool Fixture::no_check = false;