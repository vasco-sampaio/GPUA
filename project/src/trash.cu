// Padding the input array to the next power of 2 --> BAD IDEA
//void scan(int* input, int* output, int n) {
//    long pow_2 = NEXT_POW_2(n);
//    if (n == pow_2) {
//        scan_pow_2(input, output, n);
//        return;
//    }
//
//    int prev_pow_2 = PREV_POW_2(n);
//
//    if (n - prev_pow_2 < pow_2 - n) {
//        scan_pow_2(input, output, prev_pow_2);
//
//        int *last = output + prev_pow_2 - 1;
//
//        int rem = n - prev_pow_2;
//        int n_rem = NEXT_POW_2(rem + 1);
//
//        int *rem_ptr;
//        int *output_rem;
//        CUDA_CALL(cudaMalloc(&output_rem, n_rem * sizeof(int)));
//        CUDA_CALL(cudaMalloc(&rem_ptr, n_rem * sizeof(int)));
//
//        CUDA_CALL(cudaMemcpy(rem_ptr, last, sizeof(int), cudaMemcpyDeviceToDevice));
//        CUDA_CALL(cudaMemcpy(rem_ptr + 1, input + prev_pow_2, rem * sizeof(int), cudaMemcpyDeviceToDevice));
//        CUDA_CALL(cudaMemset(rem_ptr + 1 + rem , 0, (n_rem - rem - 1) * sizeof(int)));
//
//        scan_pow_2(rem_ptr, output_rem, n_rem);
//
//        CUDA_CALL(cudaMemcpy(output + prev_pow_2, output_rem + 1, rem * sizeof(int), cudaMemcpyDeviceToDevice));
//
//        CUDA_CALL(cudaFree(rem_ptr));
//        CUDA_CALL(cudaFree(output_rem));
//    } else {
//        int *input_pow_2, *output_pow_2;
//
//        CUDA_CALL(cudaMalloc(&input_pow_2, pow_2 * sizeof(int)));
//        CUDA_CALL(cudaMalloc(&output_pow_2, pow_2 * sizeof(int)));
//
//        CUDA_CALL(cudaMemcpy(input_pow_2, input, n * sizeof(int), cudaMemcpyDeviceToDevice));
//        CUDA_CALL(cudaMemset(input_pow_2 + n, 0, (pow_2 - n) * sizeof(int)));
//
//        scan_pow_2(input_pow_2, output_pow_2, pow_2);
//
//        CUDA_CALL(cudaMemcpy(output, output_pow_2, n * sizeof(int), cudaMemcpyDeviceToDevice));
//
//        CUDA_CALL(cudaFree(input_pow_2));
//        CUDA_CALL(cudaFree(output_pow_2));
//    }
//}




    int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
    {
        // -- Pipeline initialization

        std::cout << "File loading..." << std::endl;

        // - Get file paths

        using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
        std::vector<std::string> filepaths;
        for (const auto& dir_entry : recursive_directory_iterator("/afs/cri.epita.fr/resources/teach/IRGPUA/images"))
            filepaths.emplace_back(dir_entry.path());

        Pipeline pipeline(filepaths);

        // -- Main loop containing image retrying from pipeline and fixing

        const int nb_images = pipeline.images.size();
        std::cout << "Nb images: " << nb_images << std::endl;
        std::vector<Image> images(nb_images);

        // - Init streams
        // const int nb_streams = 4;
        // cudaStream_t* streams = new cudaStream_t[nb_streams];
        // for (int i = 0; i < nb_streams; ++i) {
        //     cudaStreamCreate(&streams[i]);
        // }

        std::cout << "Done, starting compute" << std::endl;

        std::vector<int*> d_buffers(nb_images);

        #pragma omp parallel for
        for (int i = 0; i < nb_images; ++i)
        {
            // int stream_id = i % nb_streams;

            images[for (int i = 0; i < nb_streams; ++i)
            cudaStreamDestroy(streams[i]);
ALL(cudaMalloc(&d_buffers[i], images[i].width * images[i].height * sizeof(int)));
            // CUDA_CALL(cudaMemcpyAsync(d_buffers[i], images[i].buffer, images[i].width * images[i].height * sizeof(int), cudaMemcpyHostToDevice, streams[stream_id]));
            CUDA_CALL(cudaMemcpy(d_buffers[i], images[i].buffer, images[i].width * images[i].height * sizeof(int), cudaMemcpyHostToDevice);

            fix_image_gpu(d_buffers[i], images[i].size(), images[i].height * images[i].width, 0);

            // CUDA_CALL(cudaMemcpyAsync(images[i].buffer, d_buffers[i], images[i].width * images[i].height * sizeof(int), cudaMemcpyDeviceToHost, streams[stream_id]));
            CUDA_CALL(cudaMemcpy(images[i].buffer, d_buffers[i], images[i].width * images[i].height * sizeof(int), cudaMemcpyDeviceToHost));
        }

        // for (int i = 0; i < nb_images; ++i) {
        //     int stream_id = i % nb_streams;
        //     CUDA_CALL(cudaStreamSynchronize(streams[stream_id]));
        // }

        std::cout << "Done with compute, starting stats" << std::endl;

        /* #pragma for (int i = 0; i < nb_streams; ++i)
            cudaStreamDestroy(streams[i]);

            return images[n++].to_sort;
        });

        std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
            return a.total < b.total;
        });

        for (int i = 0; i < nb_images; ++i)
        {
            std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
            std::ostringstream oss;
            oss << "Image#" << images[i].to_sort.id << ".pgm";
            std::string str = oss.str();
            images[i].write(str);
        }*/

        // std::cout << "Done, the internet is safe now :)" << std::endl;

        // Destroy streams
        for (int i = 0; i < nb_streams; ++i)
            cudaStreamDestroy(streams[i]);


        for (int i = 0; i < nb_images; ++i)
            CUDA_CALL(cudaFreeHost(images[i].buffer));

        return 0;
    }



    __global__
    void find_if_kernel(int* buffer, int* result, const int value, const bool is_equal, const int size) {
        const int tid = blockIdx.x * blockDim.x + threadIdx.x;

        if (tid < size && is_equal ? buffer[tid] == value : buffer[tid] != value) {
            int old = atomicCAS(result, -1, tid);
            if (old != -1)
                atomicMin(result, tid);
        }
    }


int find_if(int* buffer, const int size, const int value, const bool is_equal) {

    const int block_size = BLOCK_SIZE(size);
    const int grid_size = (size + block_size - 1) / block_size;

    int* result;
    CUDA_CALL(cudaMallocManaged(&result, sizeof(int)));
    CUDA_CALL(cudaMemset(result, -1, sizeof(int)));

    find_if_kernel<<<grid_size, block_size, 0>>>(buffer, result, value, is_equal, size);

    CUDA_CALL(cudaDeviceSynchronize());

    int res = *result;
    CUDA_CALL(cudaFree(result));

    return res;
}


int main() {
    int input[] = {
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
        // 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
        // 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1
    };
    const int n = 40;
    int* output = new int[n];

    int *d_input, *d_output;
    cudaMalloc(&d_input, n * sizeof(int));
    cudaMalloc(&d_output, n * sizeof(int));
    cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, output, n * sizeof(int), cudaMemcpyHostToDevice);

    std::cout << "Kernel launch" << std::endl;
    // Call the scan function
    scan<ScanType::EXCLUSIVE>(d_input, d_output, n);

    std::cout << "Kernel launch done" << std::endl;

    cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);
    // Expected
    std::cout << "Expected output : ";
    int acc = 0;
    for (int i = 0; i < n; ++i) {
        if (i > 0)
            acc += input[i - 1];
        std::cout << "| " << i << ": " << acc << " |";
    }
    std::cout << '\n' << std::endl;
    std::cout << std::endl;

    // Print the output array
    std::cout << "Output :          ";
    for (int i = 0; i < n; ++i)
        std::cout << "| " << i << ": " << output[i] << " |";
    std::cout << std::endl;

    delete[] output;
 
    cudaFree(d_input);
    cudaFree(d_output);
 
    return 0;
}

void print_array(int* array, int size, bool copy = false) {
    if (copy) {
        int* tmp = (int *)malloc(size * sizeof(int));
        cudaMemcpy(tmp, array, size * sizeof(int), cudaMemcpyDeviceToHost);

        for (int i = 0; i < size; ++i)
            std::cout << tmp[i] << " ";
        std::cout << std::endl;

        return;
    }
    for (int i = 0; i < size; ++i)
        std::cout << array[i] << " ";
    std::cout << std::endl;
};