#ifndef RAYTRACE_CUH
#define RAYTRACE_CUH

#include "renderobject.cuh"

namespace vrt {

__host__ void raytrace(
    curandState* randstate, int n_pixel, int spp,
    int max_depth, float3 ambient, float russian_roulette,
    int n_rays, Ray* rays,
    int n_objects, const RenderObject* objects,
    int n_materials, const Material* materials,
    float3* output_buffer
);

__host__ void post_process(int n_pixels, int spp, float gamma, float3* output_buffer);

}

#endif // !RAYTRACE_CUH