#ifndef RENDEROBJECT_CUH
#define RENDEROBJECT_CUH

#include "bvh.h"
#include "primitives.cuh"
#include "material.cuh"

namespace vrt {

struct RenderObject {
    Triangle* triangles;
    BVHNode* bvh;
    int n_triangles;
    int n_bvh_nodes;
};

}

#endif // !RENDEROBJECT_CUH