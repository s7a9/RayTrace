#include "transform.cuh"

#include <cmath>

namespace vrt {

#define CALC_ROT_MATRIX(rot_rad) \
    float cosx = cosf(rot_rad.x), sinx = sinf(rot_rad.x); \
    float cosy = cosf(rot_rad.y), siny = sinf(rot_rad.y); \
    float cosz = cosf(rot_rad.z), sinz = sinf(rot_rad.z); \
    float rot00 = cosy * cosz; \
    float rot01 = -cosy * sinz; \
    float rot02 = siny; \
    float rot10 = sinx * siny * cosz + cosx * sinz; \
    float rot11 = -sinx * siny * sinz + cosx * cosz; \
    float rot12 = -sinx * cosy; \
    float rot20 = -cosx * siny * cosz + sinx * sinz; \
    float rot21 = cosx * siny * sinz + sinx * cosz; \
    float rot22 = cosx * cosy;

#define ROTATE_PARAMS float rot00, float rot01, float rot02, float rot10, float rot11, float rot12, float rot20, float rot21, float rot22
#define ROTATE_ARGS rot00, rot01, rot02, rot10, rot11, rot12, rot20, rot21, rot22

__forceinline__ __device__ float3 mmv3x3(float3& v, ROTATE_PARAMS) {
    return make_float3(
        rot00 * v.x + rot01 * v.y + rot02 * v.z,
        rot10 * v.x + rot11 * v.y + rot12 * v.z,
        rot20 * v.x + rot21 * v.y + rot22 * v.z
    );
}

__global__ void apply_transform_kernel(float3 scale, float3 translate, Triangle* triangles, int n_triangles, ROTATE_PARAMS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_triangles) return;
    Triangle& tri = triangles[idx];
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        tri.v[i].position *= scale;
        tri.v[i].position = mmv3x3(tri.v[i].position, ROTATE_ARGS);
        tri.v[i].position += translate;
        tri.v[i].normal = mmv3x3(tri.v[i].normal, ROTATE_ARGS);
    }
}

__host__ void apply_transform(const ModelTransform& transform, Triangle* triangles, int n_triangles, cudaStream_t stream) {
    // Calculate the rotation matrix
    float3 rot_rad = make_float3(transform.rotation.x * M_PI / 180.0f, transform.rotation.y * M_PI / 180.0f, transform.rotation.z * M_PI / 180.0f);
    CALC_ROT_MATRIX(rot_rad);
    // Launch the kernel
    int block_size = 256;
    int grid_size = (n_triangles + block_size - 1) / block_size;
    apply_transform_kernel<<<grid_size, block_size, 0, stream>>>(transform.scale, transform.translation, triangles, n_triangles, ROTATE_ARGS);
}

__global__ void apply_transform_vec_kernel(float3 scale, float3 translate, float3* vec, int n_vec, ROTATE_PARAMS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec) return;
    vec[idx] *= scale;
    vec[idx] = mmv3x3(vec[idx], ROTATE_ARGS) + translate;
}

__host__ void apply_transform(const ModelTransform& transform, float3* vec, int n_vec, cudaStream_t stream) {
    // Calculate the rotation matrix
    float3 rot_rad = make_float3(transform.rotation.x * M_PI / 180.0f, transform.rotation.y * M_PI / 180.0f, transform.rotation.z * M_PI / 180.0f);
    CALC_ROT_MATRIX(rot_rad);
    // Launch the kernel
    int block_size = 256;
    int grid_size = (n_vec + block_size - 1) / block_size;
    apply_transform_vec_kernel<<<grid_size, block_size, 0, stream>>>(transform.scale, transform.translation, vec, n_vec, ROTATE_ARGS);
}

}