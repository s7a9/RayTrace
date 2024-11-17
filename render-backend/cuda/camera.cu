#include "camera.cuh"

#include <curand.h>

namespace vrt {

__global__ void init_randstate_kernel(curandState *state, int seed, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    if (x >= width || y >= height) return;
    curand_init(seed, idx, 0, &state[idx]);
}

__host__ void init_randstate(curandState** state, int width, int height) {
    size_t total_pixels = width * height;
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    cudaMalloc(state, total_pixels * sizeof(curandState));
    init_randstate_kernel<<<grid, block>>>(*state, time(0), width, height);
}

__global__ void setup_raytrace_kernel(
    curandState *state,
    int width, int height, int spp,
    float3 camera_pos, float3 lower_left, float3 dx, float3 dy,
    float fov, Ray* rays
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;
    if (x >= width || y >= height) return;
    // make camera vectors
    // split the screen into width x height pixels
    // sample spp rays per at pixel (x, y)
    for (int i = 0; i < spp; i++) {
        float u = (x + curand_uniform(&state[idx])) / (float)width;
        float v = (y + curand_uniform(&state[idx])) / (float)height;
        float3 ray_dir = normalize(lower_left + u * dx + v * dy);
        rays[idx * spp + i] = Ray(camera_pos, ray_dir);
    }
}

__host__ void setup_raytrace(
    curandState *state,
    int width, int height, int spp, 
    float3 camera_pos, float3 camera_dir, float3 camera_up, float fov,
    Ray* rays
) {
    // normalize camera vectors
    camera_dir = normalize(camera_dir);
    camera_up = normalize(camera_up);
    float3 right = cross(camera_dir, camera_up);
    camera_up = cross(right, camera_dir);
    float aspect_ratio = (float)width / (float)height;
    float half_height = tan(fov / 2.0f);
    float half_width = aspect_ratio * half_height;
    float3 lower_left = camera_dir - half_width * right - half_height * camera_up;
    float3 dx = 2.0f * half_width * right;
    float3 dy = 2.0f * half_height * camera_up;
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    setup_raytrace_kernel<<<grid, block>>>(state, width, height, spp, camera_pos, 
        lower_left, dx, dy, fov, rays);
    // debug print rays
    #ifdef VRT_DEBUG
    Ray* h_rays = new Ray[width * height * spp];
    cudaMemcpy(h_rays, rays, width * height * spp * sizeof(Ray), cudaMemcpyDeviceToHost);
    for (int i = 0; i < width * height * spp; i++) {
        // printf("%d: (%f, %f, %f) -> (%f, %f, %f)\n", i, h_rays[i].origin.x, h_rays[i].origin.y, h_rays[i].origin.z, h_rays[i].direction.x, h_rays[i].direction.y, h_rays[i].direction.z);
    }
    delete[] h_rays;
    #endif
}

} // namespace vrt