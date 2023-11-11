#pragma once

#include <cuda_runtime.h>

const int NUM_STREAMS = 8;
extern cudaStream_t streams[NUM_STREAMS];

void initializeStreams();
cudaStream_t getStream(int index);
void cleanupStreams();
