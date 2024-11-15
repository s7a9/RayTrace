#ifndef MATERIAL_CUH
#define MATERIAL_CUH

#include "common.cuh"

#include <cuda_texture_types.h>

namespace vrt {

struct Material {
    enum MaterialType: uint8_t {
        NONE, LIGHT, LAMBERTIAN, METAL, REFRACTIVE, REFLECTIVE
    };

    // enum TextureType: uint8_t {
    //     SIMPLE, IMAGE
    // };

    MaterialType type;
    // TextureType tex_type; // now use cu_array == nullptr to indicate no texture
    float3 albedo;
    float optical_density;
    float metal_fuzz;
    cudaArray* cu_array;
    cudaTextureObject_t tex_obj;
};

__host__ inline Material make_material(
    Material::MaterialType type, float3 albedo, float optical_density,  float metal_fuzz, 
    cudaArray* cu_array = nullptr, cudaTextureObject_t tex_obj = 0
) {
    Material mat;
    mat.type = type;
    mat.albedo = albedo;
    mat.optical_density = optical_density;
    mat.metal_fuzz = metal_fuzz;
    mat.cu_array = cu_array;
    mat.tex_obj = tex_obj;
    return mat;
}

}

#endif // !MATERIAL_CUH