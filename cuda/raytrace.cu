#include "raytrace.cuh"

namespace vrt {

#define SUBROUTINE __device__ __forceinline__

SUBROUTINE float3 random_in_unit_sphere3(curandState *state) {
    float3 p;
    do {
        p = 2.0f * make_float3(curand_uniform(state), curand_uniform(state), curand_uniform(state)) - make_float3(1.0f, 1.0f, 1.0f);
    } while (dot3(p, p) >= 1.0f);
    return p;
}

struct HitRecord {
    float3 point;
    float3 normal;
    float3 color;
    float t, u, v;
    Triangle* triangle;
    int material_id;
};

SUBROUTINE bool triangle_intersect(const Triangle& tri, const Ray& r, HitRecord& rec) {
    float3 e1 = tri.v[1].position - tri.v[0].position;
    float3 e2 = tri.v[2].position - tri.v[0].position;
    float3 p = cross(r.direction, e2);
    float det = dot3(e1, p);
    if (det > -1e-6f && det < 1e-6f) return false;
    float inv_det = 1.0f / det;
    float3 tvec = r.origin - tri.v[0].position;
    float u = dot3(tvec, p) * inv_det;
    if (u < 0.0f || u > 1.0f) return false;
    float3 q = cross(tvec, e1);
    float v = dot3(r.direction, q) * inv_det;
    if (v < 0.0f || u + v > 1.0f) return false;
    float t = dot3(e2, q) * inv_det;
    if (t < 0.0f || t > rec.t) return false;
    rec.t = t; rec.u = u; rec.v = v;
    rec.point = r.at(t);
    return true;
}

SUBROUTINE bool object_intersect(const RenderObject& object, const Ray& r, HitRecord& rec) {
    // use a stack to traverse the BVH
    int stack[MAX_HIT_STACK_SIZE];
    int top = 1;
    int hit_triangle = -1;
    stack[0] = object.n_bvh_nodes - 1;
    while (top > 0) {
        int node_idx = stack[--top];
        auto& node = object.bvh[node_idx];
        float t_near, t_far;
        if (!node.intersect(r, t_near, t_far)) continue;
        if (t_near > rec.t) continue; // early termination
        if (node.is_leaf) {
            if (node.left >= 0 && triangle_intersect(object.triangles[node.left], r, rec)) {
                hit_triangle = node.left;
            }
            if (node.right >= 0 && triangle_intersect(object.triangles[node.right], r, rec)) {
                hit_triangle = node.right;
            }
        } else {
            if (node.left >= 0 && top < MAX_HIT_STACK_SIZE) {
                stack[top++] = node.left;
            }
            if (node.right >= 0 && top < MAX_HIT_STACK_SIZE) {
                stack[top++] = node.right;
            }
        }
    }
    if (hit_triangle >= 0) {
        rec.triangle = &object.triangles[hit_triangle];
        rec.material_id = rec.triangle->material_id;
        return true;
    }
    return false;
}

SUBROUTINE void triangle_interpolate(HitRecord& rec, const Material* materials) { // get the normal and color
    float u = rec.u, v = rec.v;
    float w = 1.0f - u - v;
    float3 norm = w * rec.triangle->v[0].normal + u * rec.triangle->v[1].normal + v * rec.triangle->v[2].normal;
    rec.normal = normalize(norm);
    auto& material = materials[rec.material_id];
    if (material.cu_array == nullptr) {
        rec.color = material.albedo;
    } else {
        float texcoord_x = w * rec.triangle->v[0].texcoord.x + u * rec.triangle->v[1].texcoord.x 
            + v * rec.triangle->v[2].texcoord.x;
        float texcoord_y = w * rec.triangle->v[0].texcoord.y + u * rec.triangle->v[1].texcoord.y 
            + v * rec.triangle->v[2].texcoord.y;
        float4 texel = tex2D<float4>(material.tex_obj, texcoord_x, texcoord_y);
        rec.color = make_float3(texel.x, texel.y, texel.z);
    }
}

SUBROUTINE float3 reflect(const float3& v, const float3& n) {
    return v - 2.0f * dot3(v, n) * n;
}

SUBROUTINE bool refract(const float3& v, const float3& n, float ni_over_nt, float3& refracted) {
    float dt = dot3(v, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0.0f) {
        refracted = ni_over_nt * (v - n * dt) - n * sqrt(discriminant);
        return true;
    }
    return false;
}

SUBROUTINE float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * pow(1.0f - cosine, 5);
}

#define RAY(dir) Ray(rec.point + 1e-4f * dir, dir)

SUBROUTINE bool material_scatter(
    curandState* rand, const Material& material, const Ray& r, const HitRecord& rec, 
    Ray& wo, float3& color
) {
    float3 direction;
    switch (material.type) {
    case Material::LAMBERTIAN: {
        // turn the normal into same hemisphere as the ray direction
        direction = rec.normal;
        if (dot3(r.direction, rec.normal) > 0.0f) {
            direction = -rec.normal;
        }
        direction = normalize(direction + random_in_unit_sphere3(rand));
        float cos_theta = abs(dot3(direction, rec.normal));
        color *= rec.color * cos_theta;
        wo = RAY(direction);
        return true;
    }
    case Material::METAL: {
        float3 normal = rec.normal;
        // turn the normal into same hemisphere as the ray direction
        if (dot3(r.direction, rec.normal) > 0.0f) {
            normal = -rec.normal;
        }
        // float3 reflection = normalize(reflect(r.direction, normal)); 
        // direction = normalize(reflection + material.metal_fuzz * random_in_unit_sphere3(rand));
        // float cos_theta = abs(dot3(direction, reflection));
        // color *= rec.color * cos_theta;
        float3 random = random_in_unit_sphere3(rand);
        if (4 * curand_uniform(rand) < 3 + material.metal_fuzz) { // same as lambertian
            direction = normal + random;
            float cos_theta = abs(dot3(direction, normal));
            color *= rec.color * cos_theta;
        } else { // pure reflection -> specular
            direction = reflect(r.direction, normal);
            direction = normalize(direction) + material.metal_fuzz * random;
        }
        wo = RAY(direction);
        return true;
    }
    case Material::REFLECTIVE: {
        float3 direction = reflect(r.direction, rec.normal);
        color *= rec.color;
        wo = RAY(direction);
        return true;
    }
    case Material::REFRACTIVE: {
        float3 outward_normal;
        float ni_over_nt;
        float cosine;
        float3 direction;
        if (dot3(r.direction, rec.normal) > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = material.optical_density;
            cosine = material.optical_density * dot3(r.direction, rec.normal);
        } else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / material.optical_density;
            cosine = -dot3(r.direction, rec.normal);
        }
        float reflect_prob;
        if (refract(r.direction, outward_normal, ni_over_nt, direction)) {
            reflect_prob = schlick(cosine, material.optical_density);
        } else {
            reflect_prob = 1.0f;
        }
        if (curand_uniform(rand) < reflect_prob) {
            direction = reflect(r.direction, rec.normal);
        }
        color *= rec.color;
        wo = RAY(direction);
        return true;
    }
    }
    return false;
}

__global__ void raytrace_kernel(
    curandState* randstate, int n_pixel, int spp, float3 ambient,
    int max_depth, float russian_roulette,
    int n_rays, Ray* rays,
    int n_objects, const RenderObject* objects,
    int n_materials, const Material* materials,
    float3* output_buffer
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_rays || rays[i].disabled) return;
    curandState* rand = &randstate[i % n_pixel];
    while (spp--) {
        Ray ray = rays[i];
        float3 color = make_float3(1.0f, 1.0f, 1.0f);
        int depth = max_depth;
        while (depth--) {
            HitRecord rec;
            rec.t = INFINITY;
            bool hit = false;
            for (int j = 0; j < n_objects; j++) {
                hit |= object_intersect(objects[j], ray, rec);
            }
            if (!hit) {
                color *= ambient;
                break;
            }
            triangle_interpolate(rec, materials);
            auto& material = materials[rec.material_id];
            if (material.type == Material::LIGHT) {
                color *= rec.color;
                break;
            }
            else if (curand_uniform(rand) < russian_roulette) {
                Ray new_ray;
                if (!material_scatter(rand, material, ray, rec, new_ray, color)) {
                    color *= ambient;
                    break;
                }
                ray = new_ray;
            }
            else {
                color *= ambient;
                break;
            }
        }
        if (depth < 0) continue;
        // atmoically add the color to the output buffer;
        atomicAdd(&output_buffer[i].x, color.x);
        atomicAdd(&output_buffer[i].y, color.y);
        atomicAdd(&output_buffer[i].z, color.z);
    }
}

__global__ void post_process_kernel(int n_pixels, int spp, float gamma, float3* output_buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_pixels) return;
    output_buffer[i] = output_buffer[i] / (float)spp;
    // clamp the color
    output_buffer[i].x = min(1.0f, max(0.0f, output_buffer[i].x));
    output_buffer[i].y = min(1.0f, max(0.0f, output_buffer[i].y));
    output_buffer[i].z = min(1.0f, max(0.0f, output_buffer[i].z));
    // gamma correction
    output_buffer[i].x = pow(output_buffer[i].x, 1.0f / gamma);
    output_buffer[i].y = pow(output_buffer[i].y, 1.0f / gamma);
    output_buffer[i].z = pow(output_buffer[i].z, 1.0f / gamma);
}

__host__ void raytrace(
    curandState* randstate, int n_pixel, int spp,
    int max_depth, float3 ambient, float russian_roulette,
    int n_rays, Ray* rays,
    int n_objects, const RenderObject* objects,
    int n_materials, const Material* materials,
    float3* output_buffer
) {
    int block_size = 64;
    int num_blocks = (n_rays + block_size - 1) / block_size;
    raytrace_kernel<<<num_blocks, block_size>>>(
        randstate, n_pixel, spp, ambient, max_depth, russian_roulette,
        n_rays, rays, n_objects, objects, n_materials, materials,
        output_buffer
    );
    // print the output buffer to file
    #ifdef VRT_DEBUG
    float3* output = new float3[n_pixels];
    cudaMemcpy(output, output_buffer, n_pixels * sizeof(float3), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_pixels; i++) {
        printf("%d: (%f, %f, %f)\n", i, output[i].x, output[i].y, output[i].z);
    }
    delete[] output;
    #endif
}


__host__ void post_process(int n_pixels, int spp, float gamma, float3* output_buffer) {
    int block_size = 256;
    int num_blocks = (n_pixels + block_size - 1) / block_size;
    post_process_kernel<<<num_blocks, block_size>>>(
        n_pixels, spp, gamma, output_buffer
    );
    cudaDeviceSynchronize();
}

}