#ifndef RAYTRACE_CUH
#define RAYTRACE_CUH

#include "renderobject.cuh"

namespace vrt {

__host__ void raytrace(
    curandState* randstate, int n_randstate, int spp,
    int max_depth, float alpha, float3 ambient, float russian_roulette,
    int n_rays, Ray* rays,
    int n_objects, const RenderObject* objects,
    int n_materials, const Material* materials,
    float3* output_buffer
);

}

#endif // !RAYTRACE_CUH