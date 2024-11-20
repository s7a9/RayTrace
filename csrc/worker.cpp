#include "worker.h"

#include "raytrace.cuh"

#include <iostream>
#include <thread>
#include <chrono>
#include <cmath>

#include <opencv2/opencv.hpp>

namespace vrt {

void Worker::post_process_loop_() {
    while (!should_exit.load()) {
        if (!has_new_output.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        cv::Mat img(loader.config().height, loader.config().width, CV_32FC3);
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            memcpy(img.data, h_buffer_, n_pixels_ * sizeof(float3));
        }
        cv::flip(img, img, 0); // flip image upside down
        img.convertTo(img, CV_8UC3, 255.0f);
        cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
        // construct filename using time
        // auto now = std::chrono::system_clock::now();
        // auto now_c = std::chrono::system_clock::to_time_t(now);
        {
            std::lock_guard<std::mutex> lock(filename_mutex);
            char time_string[MaxFilenameLength] = "output";
            int n;
            // n = strftime(time_string, MaxFilenameLength, "%Y-%m-%d-%H-%M-%S", std::localtime(&now_c));
            n = snprintf(output_filename, MaxFilenameLength, "%s/%s-%s.png", output_dir.c_str(), scene_name.c_str(), time_string);
            if (n >= MaxFilenameLength) {
                std::cerr << "Filename too long" << std::endl;
                return;
            }
            std::cout << "Image saved to " << output_filename << std::endl;
            cv::imwrite(output_filename, img);
        }
        has_new_output.store(false);
    }
    std::cout << "encoder: cleaning up" << std::endl;
}

void print_progress_bar(int current, int total, std::chrono::duration<float> elapsed) {
    int bar_width = 32;
    float progress = (float)current / total;
    int pos = bar_width * progress;
    std::cout << "[";
    for (int i = 0; i < bar_width; i++) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] " << int(progress * 100.0) << "%\t";
    std::cout << "Elapsed(s): " << elapsed.count() << ", ";
    std::cout << "ETA(s): " << (elapsed / progress - elapsed).count() << "" << std::endl;
}

void Worker::render_(float3* d_buffer, Ray* d_rays) {
    auto config = loader.config();
    // record start time
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "---> Raytracing Start <---" << std::endl;
    int n_sample_remain = config.n_samples;
    cudaMemset(d_buffer, 0, n_pixels_ * sizeof(float3));
    while (n_sample_remain > 0) {
        int n_sample = std::min(n_sample_remain, config.batch_size);
        setup_raytrace(
            d_rand_state_, config.width, config.height, n_sample,
            config.camera_pos, config.camera_dir, config.camera_up, config.fov,
            d_rays
        );
        cudaDeviceSynchronize();
        raytrace(
            d_rand_state_, config.width * config.height, n_sample,
            config.max_depth, config.background, config.russian_roulette,
            n_rays_, d_rays,
            loader.num_object(), loader.device_objects(),
            loader.num_material(), loader.device_materials(),
            d_buffer
        );
        cudaDeviceSynchronize();
        n_sample_remain -= n_sample;
        // print a moving progress bar
        print_progress_bar(config.n_samples - n_sample_remain, config.n_samples, std::chrono::high_resolution_clock::now() - start);
    }
    post_process(config.width * config.height, config.n_samples, config.gamma, d_buffer);
    std::cout << "---> done in ";
    // record end time
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms <---" << std::endl;
    // post_process_(d_buffer);
}

void Worker::run_loop_() {
    float3* h_buffer = new float3[n_pixels_];
    while (!should_exit.load()) {
        if (!has_new_input.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        render_(d_buffer_, d_rays_); // leave for double buffering
        cudaMemcpy(h_buffer, d_buffer_, n_pixels_ * sizeof(float3), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        {
            std::lock_guard<std::mutex> lock(buffer_mutex);
            std::swap(h_buffer_, h_buffer);
        }
        has_new_output.store(true);
        has_new_input.store(false);
    }
    std::cout << "worker: cleaning up" << std::endl;
    delete[] h_buffer;
}

Worker::Worker(const std::string& scene_file, const std::string& output_dir, int batch_size) : 
    should_exit(false), has_new_input(false), has_new_output(false), output_dir(output_dir) {
    output_filename[0] = '\0';
    if (loader.load_scene(scene_file) != 0) {
        std::cerr << "Failed to load scene file " << scene_file << std::endl;
        exit(0);
    }
    auto& config = loader.config();
    if (batch_size <= 0 || batch_size > config.n_samples) {
        batch_size = config.n_samples;
    } else if (config.n_samples % batch_size != 0) {
        std::cerr << "Warning: batch size " << batch_size 
            << " is not a divisor of sample per pixel " << config.n_samples << std::endl;
        std::cerr << "Setting batch size to spp, which may cause CUDA OOM." << std::endl;
        batch_size = config.n_samples;
    }
    config.batch_size = batch_size;
    init_randstate(&d_rand_state_, config.width, config.height);
    n_pixels_ = config.width * config.height;
    n_rays_ = n_pixels_ * config.batch_size;
    // malloc device memory and detect OOM error
    auto err = cudaMalloc(&d_buffer_, n_pixels_ * sizeof(float3));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for buffer, " 
            << "error: " << cudaGetErrorString(err) << std::endl;
        exit(0);
    }
    err = cudaMalloc(&d_rays_, n_rays_ * sizeof(Ray));
    if (err != cudaSuccess) {
        std::cerr << "Failed to allocate device memory for rays, " 
            << "error: " << cudaGetErrorString(err) << std::endl;
        exit(0);
    }
    cudaDeviceSynchronize();
    // extract scene name as output filename prefix
    auto pos = scene_file.find_last_of('/');
    pos = (pos == std::string::npos) ? 0 : pos + 1;
    scene_name = scene_file.substr(pos, scene_file.find_last_of('.') - pos);
    std::cout << "worker: loaded scene: " << scene_name << std::endl;
    // allocate host memory for buffer
    h_buffer_ = new float3[n_pixels_];
    // copy camera position and direction
    initial_cam_pos = config.camera_pos;
    initial_cam_dir = config.camera_dir;
    initial_cam_up = config.camera_up;
}

Worker::~Worker() {
    should_exit.store(true);
    render_thread.join();
    encode_thread.join();
    cudaFree(d_buffer_);
    cudaFree(d_rays_);
    cudaFree(d_rand_state_);
    delete[] h_buffer_;
}

void Worker::run() {
    render_thread = std::thread(&Worker::run_loop_, this);
    encode_thread = std::thread(&Worker::post_process_loop_, this);
}

void Worker::reset_camera() {
    auto& config = loader.config();
    config.camera_pos = initial_cam_pos;
    config.camera_dir = initial_cam_dir;
    config.camera_up = initial_cam_up;
    has_new_input.store(true);
}

void Worker::update_camera(float3 pos, float3 rot) {
    auto& config = loader.config();
    config.camera_pos = pos;
    // calculate camera direction by phone rotation z-alpha, x-beta, y-gamma
    // discard original camera direction
    rot *= 0.0174532925f; // convert to radians
    float cos_alpha = cos(rot.x), sin_alpha = sin(rot.x);
    float cos_beta = cos(rot.y);
    float3 dir = make_float3(cos_alpha, sin_alpha, -cos_beta);
    float3 up = make_float3(0.0f, 0.0f, 1.0f);
    config.camera_up = up;
    config.camera_dir = dir;
    has_new_input.store(true);
}

} // namespace vrt