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

#define CPU 0

#if CPU
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

        // - Init streams
        const int nb_streams = 4;
        cudaStream_t* streams = new cudaStream_t[nb_streams];
        for (int i = 0; i < nb_streams; ++i) {
            cudaStreamCreate(&streams[i]);
        }

        std::cout << "Done, starting compute" << std::endl;

        std::vector<int*> d_buffers(nb_images);

        #pragma omp parallel for
        for (int i = 0; i < nb_images; ++i)
        {
            int stream_id = i % nb_streams;

            images[i] = pipeline.get_image(i);

            CUDA_CALL(cudaMalloc(&d_buffers[i], images[i].width * images[i].height * sizeof(int)));
            CUDA_CALL(cudaMemcpyAsync(d_buffers[i], images[i].buffer, images[i].width * images[i].height * sizeof(int), cudaMemcpyHostToDevice, streams[stream_id]));

            fix_image_gpu(d_buffers[i], images[i].width, images[i].height, &streams[stream_id]);

            CUDA_CALL(cudaMemcpyAsync(images[i].buffer, d_buffers[i], images[i].width * images[i].height * sizeof(int), cudaMemcpyDeviceToHost, streams[stream_id]));
        }

        for (int i = 0; i < nb_images; ++i) {
            int stream_id = i % nb_streams;
            CUDA_CALL(cudaStreamSynchronize(streams[stream_id]));
        }

        std::cout << "Done with compute, starting stats" << std::endl;

//        // -- All images are now fixed : compute stats (total then sort)
//
//        // - First compute the total of each image
//        #pragma omp parallel for
//        for (int i = 0; i < nb_images; ++i)
//        {
//            int stream_id = i % nb_streams;
//
//            auto& image = images[i];
//            const int image_size = image.width * image.height;
//
//            int *d_total;
//            CUDA_CALL(cudaMallocManaged(&d_total, sizeof(int)));
//
//            // reduce(nullptr, d_total, image_size, &streams[stream_id]);
//            image.to_sort.total = *d_total;
//
//            CUDA_CALL(cudaFree(d_total));
//        }
//
//        // TODO OPTIONAL : for you GPU version you can store it the way you want
//        using ToSort = Image::ToSort;
//        std::vector<ToSort> to_sort(nb_images);
//        std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
//        {
//            return images[n++].to_sort;
//        });
//
//        // TODO OPTIONAL : make it GPU compatible (aka faster)
//        std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
//            return a.total < b.total;
//        });
//
//        // TODO : Test here that you have the same results
//        for (int i = 0; i < nb_images; ++i)
//        {
//            std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
//            std::ostringstream oss;
//            oss << "Image#" << images[i].to_sort.id << ".pgm";
//            std::string str = oss.str();
//            images[i].write(str);
//        }
//
//        std::cout << "Done, the internet is safe now :)" << std::endl;
//
//        // Cleaning
//        // TODO : Don't forget to update this if you change allocation style
////        for (int i = 0; i < nb_images; ++i)
////            CUDA_CALL(cudaFree(d_buffers[i]));

        // - free streams
        for (int i = 0; i < nb_streams; ++i)
            cudaStreamDestroy(streams[i]);

        for (int i = 0; i < nb_images; ++i)
            CUDA_CALL(cudaFreeHost(images[i].buffer));

        return 0;
    }
#endif
