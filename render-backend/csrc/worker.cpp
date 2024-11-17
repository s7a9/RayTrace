#include "worker.h"

#include "raytrace.cuh"

#include <iostream>
#include <thread>
#include <chrono>

#include <opencv2/opencv.hpp>

namespace vrt {

float3 rotate(float3 v, float3 axis, float angle) {
    float s = sin(angle);
    float c = cos(angle);
    float3 u = normalize(axis);
    float3 ret;
    ret.x = v.x * (c + (1 - c) * u.x * u.x) + v.y * ((1 - c) * u.x * u.y - s * u.z) + v.z * ((1 - c) * u.x * u.z + s * u.y);
    ret.y = v.x * ((1 - c) * u.y * u.x + s * u.z) + v.y * (c + (1 - c) * u.y * u.y) + v.z * ((1 - c) * u.y * u.z - s * u.x);
    ret.z = v.x * ((1 - c) * u.z * u.x - s * u.y) + v.y * ((1 - c) * u.z * u.y + s * u.x) + v.z * (c + (1 - c) * u.z * u.z);
    return ret;
}

void update_config_state(RenderConfig& config, float3 delta_pos, float3 delta_rot) {
    config.camera_pos += delta_pos;
    // calculate new camera direction and up vector
    float3 front = normalize(config.camera_dir);
    float3 right = normalize(cross(front, config.camera_up));
    float3 up = cross(right, front);
    float3 new_front = front;
    float3 new_up = up;
    // rotate around right axis
    float3 new_front_rot = rotate(front, right, delta_rot.y);
    float3 new_up_rot = rotate(up, right, delta_rot.y);
    // rotate around up axis
    new_front = rotate(new_front_rot, up, delta_rot.x);
    new_up = rotate(new_up_rot, up, delta_rot.x);
    config.camera_dir = new_front;
    config.camera_up = new_up;
}

void Worker::post_process_(float3* d_buffer) {
    cv::Mat img(loader.config().height, loader.config().width, CV_32FC3);
    cudaMemcpy(img.data, d_buffer, n_pixels_ * sizeof(float3), cudaMemcpyDeviceToHost);
    cv::flip(img, img, 0); // flip image upside down
    img.convertTo(img, CV_8UC3, 255.0f);
    cv::cvtColor(img, img, cv::COLOR_RGB2BGR);
    // construct filename using time
    // auto now = std::chrono::system_clock::now();
    // auto now_c = std::chrono::system_clock::to_time_t(now);
    std::lock_guard<std::mutex> lock(output_mutex);
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
    post_process(config.width * config.height, config.n_samples, config.alpha, d_buffer);
    std::cout << "---> done in ";
    // record end time
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms <---" << std::endl;
    post_process_(d_buffer);
}

void Worker::run_loop_() {
    while (!should_exit.load()) {
        if (has_new_input.load()) {
            render_(d_buffer_, d_rays_); // leave for double buffering
            has_new_input.store(false);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

Worker::Worker(const std::string& scene_file, const std::string& output_dir, int batch_size) : 
    should_exit(false), has_new_input(false), output_dir(output_dir) {
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
}

Worker::~Worker() {
    should_exit.store(true);
    render_thread.join();
    cudaFree(d_buffer_);
    cudaFree(d_rays_);
    cudaFree(d_rand_state_);
}

void Worker::run() {
    render_thread = std::thread(&Worker::run_loop_, this);
}

void Worker::update_camera(float3 delta_pos, float3 delta_rot) {
    update_config_state(loader.config(), delta_pos, delta_rot);
    has_new_input.store(true);
}

} // namespace vrt