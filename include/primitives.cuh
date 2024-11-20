#ifndef PRIMITIVES_CUH
#define PRIMITIVES_CUH

#include "common.cuh"

namespace vrt {

struct Ray {
    float3 origin;
    float3 direction;

    __device__ __host__ Ray() {}

    __device__ __host__ Ray(float3 origin, float3 direction) : 
        origin(origin), direction(direction) {}
    
    __device__ __host__ float3 at(float t) const {
        return origin + t * direction;
    }
};

struct Vertex {
    float3 position;
    float3 normal;
    float2 texcoord;
};

struct Triangle {
    int material_id;
    Vertex v[3];
};

}

#endif // !PRIMITIVES_CUH