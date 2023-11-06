#pragma once


#define NEXT_POW_2(x) (1 << (32 - __builtin_clz(x - 1)))
#define PREV_POW_2(x) (1 << (31 - __builtin_clz(x)))
#define BLOCK_SIZE(x) std::min(std::max(PREV_POW_2(x), 32), 1024)
