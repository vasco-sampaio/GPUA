#include "cuda_streams.cuh"

#include "utils.cuh"

cudaStream_t streams[NUM_STREAMS];

void initializeStreams() {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CALL(cudaStreamCreate(&streams[i]));
    }
}

cudaStream_t getStream(int index) {
    return streams[index];
}

void cleanupStreams() {
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CUDA_CALL(cudaStreamSynchronize(streams[i]));
        CUDA_CALL(cudaStreamDestroy(streams[i]));
    }
}