#include "bvh.h"

namespace vrt {

struct Centroid {
    float3 position;
    int index;
};

void print_bvh_recursive(const std::vector<BVHNode>& bvh, int node, int depth) {
    for (int i = 0; i < depth; i++) {
        std::cout << "  ";
    }
    std::cout << "Node " << node << " ";
    if (bvh[node].is_leaf) { // print aabb
        auto& aabb = bvh[node].box;
        std::cout << "Leaf: " << aabb.min.x << " " << aabb.min.y << " " << aabb.min.z << " " 
            << aabb.max.x << " " << aabb.max.y << " " << aabb.max.z << std::endl;
    } else {
        std::cout << "Internal: " << bvh[node].left << " " << bvh[node].right;
        auto& aabb = bvh[node].box;
        std::cout << " " << aabb.min.x << " " << aabb.min.y << " " << aabb.min.z << " " 
            << aabb.max.x << " " << aabb.max.y << " " << aabb.max.z << std::endl;
        print_bvh_recursive(bvh, bvh[node].left, depth + 1);
        print_bvh_recursive(bvh, bvh[node].right, depth + 1);
    }
}

void build_recursive(std::vector<Centroid>& centroids, int start, int end, std::vector<BVHNode>& bvh, Triangle* triangles) {
    // std::cout << "Building BVH from " << start << " to " << end << std::endl;
    if (start + 1 >= end) {
        BVHNode node;
        node.is_leaf = true;
        node.left = centroids[start].index;
        node.right = -1;
        AABB box = AABB(triangles[node.left]);
        if (end > start) {
            node.right = centroids[start + 1].index;
            box.expand(triangles[node.right]);
        } 
        node.box = box;
        bvh.push_back(node);
        return;
    }
    AABB box;
    for (int i = start; i <= end; i++) {
        box.expand(centroids[i].position);
    }
    int axis = 0;
    float max_extent = box.max.x - box.min.x;
    float extent = box.max.y - box.min.y;
    if (extent > max_extent) {
        axis = 1;
        max_extent = extent;
    }
    extent = box.max.z - box.min.z;
    if (extent > max_extent) {
        axis = 2;
    }
    int mid = (start + end) / 2;
    std::nth_element(centroids.begin() + start, centroids.begin() + mid, centroids.begin() + end + 1, 
        [axis](const Centroid& a, const Centroid& b) {
            return get(a.position, axis) < get(b.position, axis);
        });
    BVHNode node;
    node.is_leaf = false;
    build_recursive(centroids, start, mid, bvh, triangles);
    node.left = bvh.size() - 1;
    build_recursive(centroids, mid + 1, end, bvh, triangles);
    node.right = bvh.size() - 1;
    box.expand(bvh[node.left].box);
    box.expand(bvh[node.right].box);
    node.box = box;
    bvh.push_back(node);
}

int build_bvh(Triangle* triangles, int n_triangles, std::vector<BVHNode>& bvh) {
    bvh.clear();
    std::vector<Centroid> centroids(n_triangles);
    for (int i = 0; i < n_triangles; i++) {
        centroids[i].position = (triangles[i].v[0].position 
            + triangles[i].v[1].position 
            + triangles[i].v[2].position) / 3.0f;
        centroids[i].index = i;
    }
    build_recursive(centroids, 0, n_triangles - 1, bvh, triangles);
    // expane all aabb slightly to avoid floating point error
    // constexpr float3 vec_eps = {1e-6f, 1e-6f, 1e-6f};
    // for (auto& node : bvh) {
    //     node.box.min -= vec_eps;
    //     node.box.max += vec_eps;
    // }
    #ifdef VRT_DEBUG
    int root = bvh.size() - 1;
    print_bvh_recursive(bvh, root, 0);
    #endif
    return bvh.size();
}

}