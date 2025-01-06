#include "rasterize.cuh"

namespace vrt {

struct mat4x4 {
    float v[4][4];

    __host__ __device__ mat4x4() {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                v[i][j] = 0.0f;
            }
        }
    }

    __host__ __device__ mat4x4(
        float a00, float a01, float a02, float a03,
        float a10, float a11, float a12, float a13,
        float a20, float a21, float a22, float a23,
        float a30, float a31, float a32, float a33
    ) {
        v[0][0] = a00; v[0][1] = a01; v[0][2] = a02; v[0][3] = a03;
        v[1][0] = a10; v[1][1] = a11; v[1][2] = a12; v[1][3] = a13;
        v[2][0] = a20; v[2][1] = a21; v[2][2] = a22; v[2][3] = a23;
        v[3][0] = a30; v[3][1] = a31; v[3][2] = a32; v[3][3] = a33;
    }
};

__host__ inline mat4x4 operator*(const mat4x4& a, const mat4x4& b) {
    mat4x4 res;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                res.v[i][j] += a.v[i][k] * b.v[k][j];
            }
        }
    }
    return res;
}

__device__ float3 operator*(const mat4x4& a, const float3& b) {
    return make_float3(
        a.v[0][0] * b.x + a.v[0][1] * b.y + a.v[0][2] * b.z + a.v[0][3],
        a.v[1][0] * b.x + a.v[1][1] * b.y + a.v[1][2] * b.z + a.v[1][3],
        a.v[2][0] * b.x + a.v[2][1] * b.y + a.v[2][2] * b.z + a.v[2][3]
    );
}

__device__ static float atomicMin(float* address, float val)
{
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = ::atomicCAS(address_as_i, assumed,
            __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__device__ float3 barycentric(float3 a, float3 b, float3 c, float2 p) {
    float3 s[2];
    s[0] = make_float3(c.x - a.x, b.x - a.x, a.x - p.x);
    s[1] = make_float3(c.y - a.y, b.y - a.y, a.y - p.y);
    float3 u = cross(s[0], s[1]);
    if (fabs(u.z) > 1e-2) {
        return make_float3(1.0f - (u.x + u.y) / u.z, u.y / u.z, u.x / u.z);
    }
    return make_float3(-1.0f, 1.0f, 1.0f);
}

__device__ float3 get_color(const Material& material, const Triangle& tri, const float3& bary) {
    if (material.cu_array == nullptr) {
        return material.albedo;
    }
    float tex_x = 0.0f, tex_y = 0.0f;
    #pragma unroll
    for (int i = 0; i < 3; i++) {
        tex_x += get(bary, i) * tri.v[i].texcoord.x;
        tex_y += get(bary, i) * tri.v[i].texcoord.y;
    }
    tex_x = fminf(fmaxf(tex_x, 0.0f), 1.0f);
    tex_y = fminf(fmaxf(tex_y, 0.0f), 1.0f);
    float4 color = tex2D<float4>(material.tex_obj, tex_x, tex_y);
    return make_float3(color.x, color.y, color.z);
}

__global__ void rasterize_kernel(
    int bin_size, int width, int height,
    int n_triangles, const Triangle* triangles,
    int n_materials, const Material* materials,
    mat4x4 proj, float3* framebuffer, float* depthbuffer
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bin_start = tid * bin_size;
    for (int i = 0; i < bin_size; ++i) {
        int tri_id = bin_start + i;
        if (tri_id > n_triangles) {
            break;
        }
        const Triangle& tri = triangles[tri_id];
        float3 v[3];
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            v[j] = tri.v[j].position;
            v[j] = proj * v[j];
            v[j] /= v[j].z;
        }
        // calculate the bounding box of the triangle
        float2 min_pos = make_float2(FLT_MAX, FLT_MAX);
        float2 max_pos = make_float2(-FLT_MAX, -FLT_MAX);
        #pragma unroll
        for (int j = 0; j < 3; j++) {
            min_pos.x = fminf(min_pos.x, v[j].x);
            min_pos.y = fminf(min_pos.y, v[j].y);
            max_pos.x = fmaxf(max_pos.x, v[j].x);
            max_pos.y = fmaxf(max_pos.y, v[j].y);
        }
        // calculate the bounding box of the screen
        int min_x = fmaxf(0, ceilf(min_pos.x)) * width;
        int min_y = fmaxf(0, ceilf(min_pos.y)) * height;
        int max_x = fminf(1, floorf(max_pos.x)) * width;
        int max_y = fminf(1, floorf(max_pos.y)) * height;
        // rasterize the triangle
        for (int y = min_y; y <= max_y; y++) {
            for (int x = min_x; x <= max_x; x++) {
                float2 p = make_float2(x + 0.5f, y + 0.5f);
                float3 bary = barycentric(v[0], v[1], v[2], p);
                if (get(bary, 0) < 0.0f || get(bary, 1) < 0.0f || get(bary, 2) < 0.0f) {
                    continue;
                }
                float z = 0.0f;
                #pragma unroll
                for (int j = 0; j < 3; j++) {
                    z += get(bary, j) * v[j].z;
                }
                int idx = y * width + x;
                float old_z = atomicMin(&depthbuffer[idx], z);
                if (z < old_z) {
                    float3 color = get_color(materials[tri.material_id], tri, bary);
                    framebuffer[idx] = color;
                }
            }
        }
    }
}

__host__ inline mat4x4 build_projection(const RenderConfig& config) {
    const float3 &cam_pos = config.camera_pos;
    const float3 &cam_dir = config.camera_dir;
    const float3 &cam_up = config.camera_up;
    auto fov = config.fov;
    float aspect = config.width / config.height;
    // calculate the projection matrix and view matrix
    float3 x = normalize(cross(cam_dir, cam_up));
    float3 y = normalize(cross(x, cam_dir));
    float3 z = cam_dir;
    float3 eye = cam_pos;
    float f = 1.0f / tan(fov * 0.5f);
    float nf = 1.0f / (config.max_depth - 1.0f);
    float fn = 1.0f / (config.max_surface - config.max_depth);
    mat4x4 proj = mat4x4(
        f / aspect, 0.0f, 0.0f, 0.0f,
        0.0f, f, 0.0f, 0.0f,
        0.0f, 0.0f, (config.max_surface + config.max_depth) * fn, 1.0f,
        0.0f, 0.0f, -config.max_depth * config.max_surface * nf, 0.0f
    );
    mat4x4 view = mat4x4(
        x.x, x.y, x.z, -dot3(x, eye),
        y.x, y.y, y.z, -dot3(y, eye),
        z.x, z.y, z.z, -dot3(z, eye),
        0.0f, 0.0f, 0.0f, 1.0f
    );
    return proj * view;
}

__global__ void setup_buffer_kernel(int width, int height, float3* framebuffer, float* depthbuffer) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= width * height) return;
    framebuffer[idx] = make_float3(0.0f, 0.0f, 0.0f);
    depthbuffer[idx] = 0.f;
}

__host__ void rasterize(
    const RenderConfig& config,
    int n_objects, const RenderObject* objects,
    int n_materials, const Material* materials,
    float3* framebuffer, float* depthbuffer
) {
    constexpr int tri_bin_size = 128;
    constexpr int block_size = 128;
    mat4x4 proj = build_projection(config);
    for (int i = 0; i < n_objects; i++) {
        const RenderObject& obj = objects[i];
        int n_tri = obj.n_triangles; // total number of triangles
        Triangle* triangles = obj.triangles;
        int block_tri_num = tri_bin_size * block_size;
        int n_blocks = (n_tri + block_tri_num - 1) / block_tri_num;
        rasterize_kernel<<<n_blocks, block_size>>>(
            tri_bin_size, config.width, config.height,
            n_tri, triangles, n_materials, materials, proj, framebuffer, depthbuffer
        );
    }
    constexpr int setup_block_size = 1024;
    int setup_grid_size = (config.width * config.height + setup_block_size - 1) / setup_block_size;
    setup_buffer_kernel<<<setup_grid_size, setup_block_size>>>(config.width, config.height, framebuffer, depthbuffer);
}

} // namespace vrt