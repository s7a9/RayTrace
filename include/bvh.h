#ifndef BVH_H
#define BVH_H

#include <vector>
#include <memory>
#include <algorithm>
#include <iostream>

#include "primitives.cuh"

namespace vrt {

struct AABB {
    float3 min;
    float3 max;

    __host__ AABB() : min(make_float3(INFINITY, INFINITY, INFINITY)), max(make_float3(-INFINITY, -INFINITY, -INFINITY)) {}

    __host__ AABB(const float3& a, const float3& b) : min(a), max(b) {}

    __host__ AABB(const Triangle& tri) {
        min = max = tri.v[0].position;
        for (int i = 1; i < 3; i++) {
            expand(tri.v[i].position);
        }
    }

    __host__ void expand(const float3& p) {
        min = make_float3(fminf(min.x, p.x), fminf(min.y, p.y), fminf(min.z, p.z));
        max = make_float3(fmaxf(max.x, p.x), fmaxf(max.y, p.y), fmaxf(max.z, p.z));
    }

    __host__ void expand(const AABB& box) {
        min = make_float3(fminf(min.x, box.min.x), fminf(min.y, box.min.y), fminf(min.z, box.min.z));
        max = make_float3(fmaxf(max.x, box.max.x), fmaxf(max.y, box.max.y), fmaxf(max.z, box.max.z));
    }

    #ifdef __CUDACC__
    __forceinline__ __device__ bool intersect(const Ray& r, float& t_near, float& t_far) const {
        float t0 = -INFINITY, t1 = INFINITY;
        #pragma unroll
        for (int i = 0; i < 3; i++) {
            float inv_d = 1.0f / get(r.direction, i);
            float t0_ = (get(min, i) - get(r.origin, i)) * inv_d;
            float t1_ = (get(max, i) - get(r.origin, i)) * inv_d;
            if (inv_d < 0.0f) {
                float tmp = t0_;
                t0_ = t1_;
                t1_ = tmp;
            }
            t0 = fmaxf(t0, t0_);
            t1 = fminf(t1, t1_);
            if (t0 > t1) return false;
        }
        t_near = t0;
        t_far = t1;
        return t_far > 0.0f;
    }
    #endif
};

struct BVHNode {
    bool is_leaf;
    AABB box;
    int left, right;

    __host__ BVHNode() : is_leaf(false), left(-1), right(-1) {}

    #ifdef __CUDACC__
    __forceinline__ __device__ bool intersect(const Ray& r, float& t_near, float& t_far) const {
        return box.intersect(r, t_near, t_far);
    }
    #endif
};

__host__ int build_bvh(Triangle* triangles, int n_triangles, std::vector<BVHNode>& bvh);

}

#endif // !BVH_H