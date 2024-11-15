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
    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);
    std::lock_guard<std::mutex> lock(output_mutex);
    char time_string[MaxFilenameLength];
    int n = strftime(time_string, MaxFilenameLength, "%Y-%m-%d-%H-%M-%S.png", std::localtime(&now_c));
    n = snprintf(output_filename, MaxFilenameLength, "%s/%s", output_dir.c_str(), time_string);
    if (n >= MaxFilenameLength) {
        std::cerr << "Filename too long" << std::endl;
        return;
    }
    std::cout << "Image saved to " << output_filename << std::endl;
    cv::imwrite(output_filename, img);
}

void Worker::render_(float3* d_buffer, Ray* d_rays) {
    auto config = loader.config();
    // record start time
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "---> Raytracing <---" << std::endl;
    cudaDeviceSynchronize();
    setup_raytrace(d_rand_state_, 0, config.width, config.height, config.n_samples, 
        config.camera_pos, config.camera_dir, config.camera_up, config.fov, d_buffer, d_rays);
    cudaDeviceSynchronize();
    // print config
    std::cout << "Camera position: " << config.camera_pos.x << " " << config.camera_pos.y << " " << config.camera_pos.z << std::endl;
    std::cout << "Camera direction: " << config.camera_dir.x << " " << config.camera_dir.y << " " << config.camera_dir.z << std::endl;
    std::cout << "Camera up: " << config.camera_up.x << " " << config.camera_up.y << " " << config.camera_up.z << std::endl;
    std::cout << "FOV: " << config.fov << std::endl;
    std::cout << "Max ray depth: " << config.max_depth << std::endl;
    std::cout << "Alpha: " << config.alpha << std::endl;
    std::cout << "Background: " << config.background.x << " " << config.background.y << " " << config.background.z << std::endl;
    std::cout << "Russian roulette: " << config.russian_roulette << std::endl;
    raytrace(
        d_rand_state_, n_pixels_, 
        config.n_samples, config.max_depth, config.alpha, config.background, 
        config.russian_roulette, 
        n_rays_, d_rays, 
        loader.num_object(), loader.device_objects(),
        loader.num_material(), loader.device_materials(),
        d_buffer
    );
    std::cout << "Raytracing done" << std::endl;
    // record end time
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "Time taken: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
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

Worker::Worker(const std::string& scene_file, const std::string& output_dir) : 
    should_exit(false), has_new_input(false), output_dir(output_dir) {
    output_filename[0] = '\0';
    if (loader.load_scene(scene_file) != 0) {
        std::cerr << "Failed to load scene file " << scene_file << std::endl;
        exit(0);
    }
    auto& config = loader.config();
    init_randstate(&d_rand_state_, config.width, config.height);
    n_pixels_ = config.width * config.height;
    n_rays_ = n_pixels_ * config.n_samples;
    cudaMalloc(&d_buffer_, n_pixels_ * sizeof(float3));
    cudaMalloc(&d_rays_, n_rays_ * sizeof(Ray));
    cudaDeviceSynchronize();
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