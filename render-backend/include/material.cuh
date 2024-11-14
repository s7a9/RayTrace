#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "common.cuh"

#include <cuda_texture_types.h>

namespace vrt {

struct Material {
    enum MaterialType: uint8_t {
        NONE, LIGHT, LAMBERTIAN, METAL, REFRACTIVE, REFLECTIVE
    };

    enum TextureType: uint8_t {
        SIMPLE, IMAGE
    };

    MaterialType type;
    TextureType tex_type;
    float3 albedo;
    float optical_density;
    float metal_fuzz;
    // cudaArray* cu_array;
    // texture<float, cudaTextureType2D, cudaReadModeElementType> texref;
};

}

#endif // !MATERIAL_CUH