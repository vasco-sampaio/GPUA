#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include "kernels/scan.cuh"
#include "kernels/reduce.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

#define DEBUG true

#if !DEBUG
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

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);

    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course (wait for last class)
        images[i] = pipeline.get_image(i);
        fix_image_cpu(images[i]);
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);
    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i)
    {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    // Cleaning
    // TODO : Don't forget to update this if you change allocation style
    for (int i = 0; i < nb_images; ++i)
        free(images[i].buffer);

    return 0;
}

#else
    #include <cassert>
    int main() {
        const int n = 256;
        int* input = new int[n];
        int* output = new int[n];

        // Initialize the input array
        for (int i = 0; i < n; ++i) {
            input[i] = i;
        }

        int *d_input, *d_output;
        cudaMalloc(&d_input, n * sizeof(int));
        cudaMalloc(&d_output, n * sizeof(int));

        cudaMemcpy(d_input, input, n * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_output, output, n * sizeof(int), cudaMemcpyHostToDevice);

        std::cout << "Kernel launch" << std::endl;
        // Call the scan function
        scan(d_input, d_output, n);

        std::cout << "Kernel launch done" << std::endl;

        cudaMemcpy(input, d_input, n * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(output, d_output, n * sizeof(int), cudaMemcpyDeviceToHost);

        assert(output[n- 1] == acc);

        // Expected
        std::cout << "Expected output : ";
        int acc = 0;
        for (int i = 0; i < n; ++i) {
            acc += i;
            std::cout << "| " << acc << " |";
        }
        std::cout << std::endl;

        // Print the output array
        std::cout << "Output :          ";
        for (int i = 0; i < n; ++i) {
            std::cout << "| " << output[i] << " |";
        }
        std::cout << std::endl;

        delete[] input;
        delete[] output;

        cudaFree(d_input);
        cudaFree(d_output);

        return 0;
    }
#endif