#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include "fix_gpu.cuh"
#include "kernels/scan.cuh"
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
    for (const auto& dir_entry : recursive_directory_iterator("/afs/cri.epita.fr/resources/teach/IRGPUA/images"))
        filepaths.emplace_back(dir_entry.path());


    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retrying from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::cout << "Nb images: " << nb_images << std::endl;
    std::vector<Image> images(nb_images);

    std::cout << "Done, starting compute" << std::endl;

    std::vector<int*> d_buffers(nb_images);

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        images[i] = pipeline.get_image(i);

        d_buffers[i] = new int[images[i].size()];
        cudaMemcpy(d_buffers[i], images[i].buffer, images[i].size() * sizeof(int), cudaMemcpyHostToDevice);

        fix_image_gpu(d_buffers[i], images[i].size(), images[i].width * images[i].height);

        cudaMemcpy(images[i].buffer, d_buffers[i], images[i].size() * sizeof(int), cudaMemcpyDeviceToHost);

        delete[] d_buffers[i];
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    for (int i = 0; i < nb_images; ++i) {
        cudaFreeHost(images[i].buffer);
    }

    return 0;
}
