#ifndef WORKER_H
#define WORKER_H
/* This file contains the worker which controls the cuda raytrace kernels.
 * The worker would first load scene data to construct the world on the device.
 * Then the worker would listen to http server for rendering requests.
 * The worker would then render the scene and save the result to a file.
 * while path would be updated to the http server.
*/

#include <thread>
#include <memory>
#include <atomic>

#include "sceneloader.h"
#include "camera.cuh"

namespace vrt {

constexpr int MaxFilenameLength = 256;

class Worker {
    SceneLoader loader;
    std::atomic_bool should_exit;
    std::atomic_bool has_new_input;
    char output_filename[MaxFilenameLength];
    std::thread render_thread;
    float3* d_buffer_;
    std::unique_ptr<float3*> h_buffer_;
    Ray* d_rays_;
    curandState* d_rand_state_;
    size_t n_pixels_;
    size_t n_rays_;
    std::mutex output_mutex;
    std::string output_dir;

    void post_process_(float3* d_buffer);

    void render_(float3* d_buffer, Ray* d_rays);

    void run_loop_();

public:
    Worker(const std::string& scene_file, const std::string& output_dir);

    ~Worker();

    void run();

    void stop() { should_exit.store(true); }

    void update_camera(float3 delta_pos, float3 delta_rot);

    // causion: need to free the memory after use
    void get_output_filename(char filename[MaxFilenameLength]) {
        std::lock_guard<std::mutex> lock(output_mutex);
        strcpy(filename, output_filename);
    }
};

}

#endif // !WORKER_H