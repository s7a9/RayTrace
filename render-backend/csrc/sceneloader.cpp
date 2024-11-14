#include "sceneloader.h"

#include <sstream>
#include <iostream>
#include <fstream>

#include "tiny_obj_loader.h"

namespace vrt {

static const char* material_type_str[] = {
    "NONE", "LIGHT", "LAMBERTIAN", "METAL", "REFRACTIVE", "REFLECTIVE"
};

SceneLoader::SceneLoader() : d_objects_(nullptr), n_objects_(0) {}

SceneLoader::~SceneLoader() {
    if (d_objects_) {
        for (int i = 0; i < n_objects_; i++) {
            cudaFree(objects_[i].triangles);
            cudaFree(objects_[i].bvh);
        }
        cudaFree(d_objects_);
        d_objects_ = nullptr;
        n_objects_ = 0;
    }
}

int SceneLoader::load_object(std::ifstream& file) {
    ModelTransform transform;
    float3& position = transform.translation;
    float3& rotation = transform.rotation;
    float& scale = transform.scale;
    position = rotation = make_float3(0.0f, 0.0f, 0.0f);
    scale = 1.0f;
    Material::MaterialType type = Material::NONE;
    Material::TextureType tex_type = Material::IMAGE;
    float3 albedo = make_float3(0.f, 0.f, 0.f);
    float optical_density = 1.0f;
    float metal_fuzz = 0.0f;
    bool disable_normal;
    std::string objfilename;
    std::string mtldir;
    std::string line;
    std::string key;
    // Read configuration
    while (!file.eof()) {
        file >> key;
        if (key == "position") { file >> position.x >> position.y >> position.z; }
        else if (key == "rotation") { file >> rotation.x >> rotation.y >> rotation.z; }
        else if (key == "scale") { file >> scale; }
        else if (key == "path") {
            std::getline(file, objfilename);
            objfilename = objfilename.substr(1);
        }
        else if (key == "mtldir") { file >> mtldir; }
        else if (key == "albedo") { file >> albedo.x >> albedo.y >> albedo.z; }
        else if (key == "optical_density") { file >> optical_density; }
        else if (key == "metal_fuzz") { file >> metal_fuzz; }
        else if (key == "end") { break; }
        else if (key == "disable_normal") { disable_normal = true; }
        else if (key == "#") { std::getline(file, line); }
        else if (key == "type") {
            std::string type_str;
            file >> type_str;
            if (type_str == "light") { type = Material::LIGHT; }
            else if (type_str == "lambertian") { type = Material::LAMBERTIAN; }
            else if (type_str == "metal") { type = Material::METAL; }
            else if (type_str == "refractive") { type = Material::REFRACTIVE; }
            else if (type_str == "reflective") { type = Material::REFLECTIVE; }
            else {
                std::getline(file, line);
                std::cerr << "Warning: Unknown material type " << type_str << std::endl;
                return -1;
            }
            tex_type = Material::SIMPLE;
        }
        else {
            std::getline(file, line);
            std::cerr << "Warning: Unknown key in object " << key << std::endl;
            return -1;
        }
    }
    if (objfilename.empty()) {
        std::cerr << "Error: No object file specified" << std::endl;
        return -1;
    }
    if (tex_type == Material::SIMPLE) {
        std::cout << "! Simple Material type: " << material_type_str[type] << std::endl;
    }
    // Load object
    std::cout << "Loading object " << objfilename << std::endl;
    tinyobj::ObjReaderConfig reader_config;
    reader_config.triangulate = true;
    reader_config.mtl_search_path = mtldir;
    tinyobj::ObjReader reader;
    if (!reader.ParseFromFile(objfilename, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "Error: " << reader.Error() << std::endl;
        }
        return -1;
    }
    if (!reader.Warning().empty()) {
        std::cerr << "Warning: " << reader.Warning() << std::endl;
    }
    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    if (shapes.size() == 0) {
        std::cerr << "Error: No shapes in object file" << std::endl;
        return -1;
    }
    for (const auto& shape : shapes) {
        std::cout << "    Loading shape " << shape.name << "...";
        // check if the shape has no triangles
        int n_triangles = shape.mesh.indices.size() / 3;
        if (n_triangles == 0) {
            std::cerr << "\n! Warning: Shape " << shape.name << " has no triangles" << std::endl;
            continue;
        }
        objects_.push_back(RenderObject());
        transforms_.push_back(transform);
        auto& obj = objects_.back();
        obj.n_triangles = n_triangles;
        obj.triangles = (Triangle*)malloc(obj.n_triangles * sizeof(Triangle));
        obj.material.albedo = albedo;
        obj.material.type = type;
        obj.material.tex_type = tex_type;
        obj.material.optical_density = optical_density;
        obj.material.metal_fuzz = metal_fuzz;
        // Load triangles
        for (size_t i = 0, itri = 0; i < shape.mesh.indices.size(); i += 3) {
            Triangle& tri = obj.triangles[itri++];
            // vertex position
            for (int j = 0; j < 3; j++) {
                int idx = shape.mesh.indices[i + j].vertex_index;
                tri.v[j].position.x = attrib.vertices[3 * idx + 0];
                tri.v[j].position.y = attrib.vertices[3 * idx + 1];
                tri.v[j].position.z = attrib.vertices[3 * idx + 2];
            }
            // vertex normal
            float3 normal;
            if (disable_normal || shape.mesh.indices[i].normal_index < 0) {
                float3 e1 = tri.v[1].position - tri.v[0].position;
                float3 e2 = tri.v[2].position - tri.v[0].position;
                normal = normalize(cross(e1, e2));
                for (int j = 0; j < 3; j++) {
                    tri.v[j].normal = normal;
                }
            } else {
                for (int j = 0; j < 3; j++) {
                    int idx = shape.mesh.indices[i + j].normal_index;
                    tri.v[j].normal.x = attrib.normals[3 * idx + 0];
                    tri.v[j].normal.y = attrib.normals[3 * idx + 1];
                    tri.v[j].normal.z = attrib.normals[3 * idx + 2];
                }
            }
            // vertex texture
            if (shape.mesh.indices[i].texcoord_index >= 0) {
                for (int j = 0; j < 3; j++) {
                    int idx = shape.mesh.indices[i + j].texcoord_index;
                    tri.v[j].texcoord.x = attrib.texcoords[2 * idx + 0];
                    tri.v[j].texcoord.y = attrib.texcoords[2 * idx + 1];
                }
            }
        }
        std::cout << obj.n_triangles << " triangles" << std::endl;
    }
    return 0;
}

int SceneLoader::load_scene(const std::string& filename) {
    memset(&config_, 0, sizeof(RenderConfig));
    int cur_obj = 0, ret;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return -1;
    }
    std::string config_key;
    std::string line;
    file >> config_key;
    if (config_key != "scene_config") {
        std::cerr << "Error: Invalid configuration file" << std::endl;
        return -1;
    }
    while (!file.eof()) {
        file >> config_key;
        if (config_key == "max_ray_depth") { file >> config_.max_depth;
        } 
        else if (config_key == "max_surface") {
            file >> config_.max_surface;
        }
        else if (config_key == "#") {
            std::getline(file, line);
        }
        else if (config_key == "size") {
            file >> config_.width >> config_.height;
        }
        else if (config_key == "sample_per_pixel") {
            file >> config_.n_samples;
        }
        else if (config_key == "fov") {
            file >> config_.fov;
            config_.fov = config_.fov * M_PI / 180.0f;
        }
        else if (config_key == "alpha") {
            file >> config_.alpha;
        }
        else if (config_key == "background") {
            file >> config_.background.x >> config_.background.y >> config_.background.z;
        }
        else if (config_key == "camera_pos") {
            file >> config_.camera_pos.x >> config_.camera_pos.y >> config_.camera_pos.z;
        }
        else if (config_key == "camera_dir") {
            file >> config_.camera_dir.x >> config_.camera_dir.y >> config_.camera_dir.z;
        }
        else if (config_key == "camera_up") {
            file >> config_.camera_up.x >> config_.camera_up.y >> config_.camera_up.z;
        }
        else if (config_key == "russian_roulette") {
            file >> config_.russian_roulette;
        }
        else if (config_key == "object") {
            ret = load_object(file);
            if (ret < 0) {
                return ret;
            }
            cur_obj++;
        }
        else if (config_key == "end") {
            break;
        }
        else {
            std::cerr << "Warning: Unknown key " << config_key << std::endl;
            return -1;
        }
    }
    // check if there is any object loaded
    if (cur_obj == 0) {
        std::cerr << "Error: No object loaded" << std::endl;
        return -1;
    }
    // Copy triangles to device
    std::cout << "Copying data to device" << std::endl;
    n_objects_ = objects_.size();
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
    }
    std::vector<BVHNode> bvh;
    std::vector<Triangle*> h_tris(n_objects_, nullptr);
    for (int i = 0; i < n_objects_; i++) {
        std::cout << "Copying object\t" << i << " /\t" <<  n_objects_ << '\r' << std::flush;
        Triangle* d_tri;
        h_tris[i] = objects_[i].triangles;
        auto& stream = streams[i % 4];
        int n_tri = objects_[i].n_triangles;
        cudaStreamSynchronize(stream);
        cudaMallocAsync(&d_tri, n_tri * sizeof(Triangle), stream);
        cudaMemcpyAsync(d_tri, objects_[i].triangles, n_tri * sizeof(Triangle), cudaMemcpyHostToDevice, stream);
        objects_[i].triangles = d_tri;
        apply_transform(transforms_[i], d_tri, objects_[i].n_triangles, stream);
        cudaMemcpyAsync(h_tris[i], d_tri, objects_[i].n_triangles * sizeof(Triangle), cudaMemcpyDeviceToHost, stream);
        #ifdef VRT_DEBUG // print transformed triangles
        for (int j = 0; j < n_tri; j++) {
            std::cout << "\nTriangle " << j << std::endl;
            // position and normal
            for (int k = 0; k < 3; k++) {
                std::cout << "    Position: " << h_tris[i][j].v[k].position.x << " " << h_tris[i][j].v[k].position.y << " " << h_tris[i][j].v[k].position.z;
                std::cout << "\tNormal: " << h_tris[i][j].v[k].normal.x << " " << h_tris[i][j].v[k].normal.y << " " << h_tris[i][j].v[k].normal.z << std::endl;
            }
        }
        #endif
    }
    // build bvh tree
    std::cout << "\nBuilding BVH tree" << std::endl;
    for (int i = 0; i < n_objects_; i++) {
        auto& stream = streams[i % 4];
        int n_tri = objects_[i].n_triangles;
        int n_bvh = build_bvh(h_tris[i], n_tri, bvh);
        objects_[i].n_bvh_nodes = n_bvh;
        // std::cout << "Built BVH with " << n_bvh << " nodes" << std::endl;
        cudaStreamSynchronize(stream);
        cudaMallocAsync(&objects_[i].bvh, n_bvh * sizeof(BVHNode), stream);
        cudaMemcpyAsync(objects_[i].bvh, bvh.data(), n_bvh * sizeof(BVHNode), cudaMemcpyHostToDevice, stream);        
        free(h_tris[i]);
    }
    for (int i = 0; i < 4; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaMalloc(&d_objects_, n_objects_ * sizeof(RenderObject));
    cudaMemcpy(d_objects_, objects_.data(), n_objects_ * sizeof(RenderObject), cudaMemcpyHostToDevice);
    return 0;
}

}