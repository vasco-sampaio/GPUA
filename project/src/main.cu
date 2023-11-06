#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include "fix_gpu.cuh"
#include "kernels/reduce.cuh"

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
    for (const auto& dir_entry : recursive_directory_iterator("/afs/cri.epita.fr/resources/teach/IRGPUA/images")) {
        filepaths.emplace_back(dir_entry.path());
        break;
    }


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

        int *d_buffer;
        cudaMalloc(&d_buffer, images[i].size() * sizeof(int));
        cudaMemcpy(d_buffer, images[i].buffer, images[i].size() * sizeof(int), cudaMemcpyHostToDevice);

        fix_image_gpu(d_buffer, images[i].size(), images[i].width * images[i].height);
        cudaMemcpy(images[i].buffer, d_buffer, images[i].size() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_buffer);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

//    // -- All images are now fixed : compute stats (total then sort)
//    // - First compute the total of each image
//    // TODO : make it GPU compatible (aka faster)
//    // You can use multiple CPU threads for your GPU version using openmp or not
//    // Up to you :)
//    #pragma omp parallel for
//    for (int i = 0; i < nb_images; ++i)
//    {
//        const int image_size = images[i].width * images[i].height;
//        int *total;
//        cudaMalloc(&total, sizeof(int));
//        reduce(d_buffers[i], total, image_size);
//
//        cudaMemcpy(&images[i].to_sort.total, total, sizeof(int), cudaMemcpyDeviceToHost);
//        cudaMemcpy(images[i].buffer, d_buffers[i], images[i].size() * sizeof(int), cudaMemcpyDeviceToHost);
//        cudaFree(total);
//        cudaFree(d_buffers[i]);
//    }
//    // - All totals are known, sort images accordingly (OPTIONAL)
//    // Moving the actual images is too expensive, sort image indices instead
//    // Copying to an id array and sort it instead
//    // TODO OPTIONAL : for you GPU version you can store it the way you want
//    // But just like the CPU version, moving the actual images while sorting will be too slow
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
//    // TODO : Test here that you have the same results
//    // You can compare visually and should compare image vectors values and "total" values
//    // If you did the sorting, check that the ids are in the same order
//    for (int i = 0; i < nb_images; ++i)
//    {
//        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
//        std::ostringstream oss;
//        oss << "Image#" << images[i].to_sort.id << ".pgm";
//        std::string str = oss.str();
//        images[i].write(str);
//    }
//    std::cout << "Done, the internet is safe now :)" << std::endl;

    for (int i = 0; i < nb_images; ++i) {
        cudaFreeHost(images[i].buffer);
    }

    return 0;
}
