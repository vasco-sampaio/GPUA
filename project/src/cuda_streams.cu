#include "cuda_streams.cuh"

cudaStream_t streams[NUM_STREAMS];

void initializeStreams() {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamCreate(&streams[i]);
    }
}

cudaStream_t getStream(int index) {
    return streams[index];
}

void cleanupStreams() {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
}