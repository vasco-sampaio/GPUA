#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include "fix_gpu_industrial.cuh"

#include <cub/device/device_reduce.cuh>

#include <thrust/device_vector.h>
#include <thrust/sort.h>

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>


using ToSort = Image::ToSort;

#if MODE == 2
    struct ToSortCmp {
        __host__ __device__
        bool operator()(const ToSort& o1, const ToSort& o2) {
            return o1.total < o2.total;
        }
    };
#endif



int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    std::cout << "File loading..." << std::endl;

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("/afs/cri.epita.fr/resources/teach/IRGPUA/images"))
        filepaths.emplace_back(dir_entry.path());

    Pipeline pipeline(filepaths);

    const int nb_images = pipeline.images.size();
    std::cout << "Nb images: " << nb_images << std::endl;
    std::vector<Image> images(nb_images);

    std::cout << "Done, starting compute" << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        images[i] = pipeline.get_image(i);
        #if MODE == 0
            fix_image_cpu(images[i]);
        #elif MODE == 1
            cudaStream_t stream = getStream(i % NUM_STREAMS);
            fix_image_gpu(images[i], stream);
        #elif MODE == 2
            fix_gpu_industrial(images[i]);
        #endif
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    #if MODE != 1
        #pragma omp parallel for
        for (int i = 0; i < nb_images; ++i)
        {
            #if MODE == 0
                auto& image = images[i];
                const int image_size = image.width * image.height;
                image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);
            #elif MODE == 2
                const int num_items = images[i].width * images[i].height;

                int      *d_in;
                void     *d_temp_storage = NULL;
                size_t   temp_storage_bytes = 0;
                int      *d_out = nullptr;

                cudaMalloc(&d_in, sizeof(int) * num_items);
                cudaMalloc(&d_out, sizeof(int));
                cudaMemcpy(d_in, images[i].buffer, sizeof(int) * num_items, cudaMemcpyHostToDevice);

                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
                cudaMalloc(&d_temp_storage, temp_storage_bytes);
                cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);

                cudaMemcpy(&images[i].to_sort.total, d_out, sizeof(int), cudaMemcpyDeviceToHost);

                cudaFree(d_temp_storage);
                cudaFree(d_out);
                cudaFree(d_in);
            #endif
        }
    #endif

    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    #if MODE == 2
        thrust::device_vector<ToSort> d_to_sort = to_sort;
        thrust::sort(d_to_sort.begin(), d_to_sort.end(), ToSortCmp());
        thrust::copy(d_to_sort.begin(), d_to_sort.end(), to_sort.begin());
    #else
        std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
            return a.total < b.total;
        });
    #endif

    for (int i = 0; i < nb_images; ++i)
    {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    for (int i = 0; i < nb_images; ++i) {
        #if MODE == 0
            free(images[i].buffer);
        #else
            CUDA_CALL(cudaFreeHost(images[i].buffer));
        #endif
    }

    return 0;
}
