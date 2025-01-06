/* This project is mainly based on ray tracing, the part of rasterization
** works as a patch attached to the ray tracing part. */
# ifndef RASTERIZE_CUH
# define RASTERIZE_CUH

#include "renderobject.cuh"

namespace vrt {

__host__ void rasterize(
    const RenderConfig& config,
    int n_objects, const RenderObject* objects,
    int n_materials, const Material* materials,
    float3* framebuffer, float* depthbuffer
);

}

# endif // !RASTERIZE_CUH