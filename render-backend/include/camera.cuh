#ifndef CAMERA_CUH
#define CAMERA_CUH

#include "common.cuh"
#include "primitives.cuh"

namespace vrt {

__host__ void init_randstate(curandState** state, int width, int height);

__host__ void setup_raytrace(
    curandState *state,
    int width, int height, int spp, 
    float3 camera_pos, float3 camera_dir, float3 camera_up, float fov,
    Ray* rays
);

}


#endif // !CAMERA_CUH