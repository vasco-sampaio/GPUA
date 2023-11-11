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

struct ToSortCmp {
    __host__ __device__
    bool operator()(const ToSort& o1, const ToSort& o2) {
        return o1.total < o2.total;
    }
};


int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("/afs/cri.epita.fr/resources/teach/IRGPUA/images"))
        filepaths.emplace_back(dir_entry.path());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retrying from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::cout << "Nb images: " << nb_images << std::endl;
    std::vector<Image> images(nb_images);

    std::cout << "Done, starting compute" << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        images[i] = pipeline.get_image(i);
        fix_gpu_industrial(images[i]);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)
    // - First compute the total of each image

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
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
    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead
    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    thrust::device_vector<ToSort> d_to_sort = to_sort;
    thrust::sort(d_to_sort.begin(), d_to_sort.end(), ToSortCmp());
    thrust::copy(d_to_sort.begin(), d_to_sort.end(), to_sort.begin());

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
        cudaFreeHost(images[i].buffer);
    }

    return 0;
}
