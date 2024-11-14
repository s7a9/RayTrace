#ifndef TRANSFORM_CUH
#define TRANSFORM_CUH

#include "common.cuh"
#include "primitives.cuh"
#include "bvh.h"

namespace vrt {

struct ModelTransform {
    float scale;
    float3 translation;
    float3 rotation;
};

__host__ void apply_transform(const ModelTransform& transform, Triangle* triangles, int n_triangles, cudaStream_t stream = 0);

__host__ void apply_transform(const ModelTransform& transform, float3* vec, int n_vec, cudaStream_t stream = 0);

}

#endif // !TRANSFORM_CUH