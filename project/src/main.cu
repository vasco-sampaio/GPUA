#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include "fix_gpu_industrial.cuh"


#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

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

//    #pragma omp parallel for
//    for (int i = 0; i < nb_images; ++i)
//    {
//        int* d_out;
//        cudaMalloc(&d_out, sizeof(int));
//
//        void* d_temp_storage = nullptr;
//        size_t temp_storage_bytes = 0;
//
//        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_buffer, d_out, image_size);
//        cudaMalloc(&d_temp_storage, temp_storage_bytes);
//        cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_buffer, d_out, image_size);
//
//        cudaMemcpy(&images[i].to_sort.total, d_out, sizeof(int), cudaMemcpyDeviceToHost);
//
//        cudaFree(d_out);
//        cudaFree(d_temp_storage);
//    }
    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead
    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
//    using ToSort = Image::ToSort;
//    std::vector<ToSort> to_sort(nb_images);
//    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
//    {
//        return images[n++].to_sort;
//    });
//    // TODO OPTIONAL : make it GPU compatible (aka faster)
//    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
//        return a.total < b.total;
//    });
//
//
//    for (int i = 0; i < nb_images; ++i)
//    {
//        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
//        std::ostringstream oss;
//        oss << "Image#" << images[i].to_sort.id << ".pgm";
//        std::string str = oss.str();
//        images[i].write(str);
//    }
    std::cout << "Done, the internet is safe now :)" << std::endl;

    for (int i = 0; i < nb_images; ++i) {
        cudaFreeHost(images[i].buffer);
    }

    return 0;
}
