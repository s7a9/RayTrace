#include "sceneloader.h"

#include <sstream>
#include <iostream>
#include <fstream>

#include <opencv2/opencv.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
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
        for (auto& mat : materials_) {
            if (mat.tex_obj) {
                cudaDestroyTextureObject(mat.tex_obj);
                cudaFreeArray(mat.cu_array);
            }
        }
        cudaFree(d_materials_);
        cudaFree(d_objects_);
        d_objects_ = nullptr;
        n_objects_ = 0;
    }
}

int SceneLoader::load_object(std::ifstream& file) {
    ModelTransform transform;
    float3& position = transform.translation;
    float3& rotation = transform.rotation;
    float3& scale = transform.scale;
    position = rotation = make_float3(0.0f, 0.0f, 0.0f);
    scale = make_float3(1.0f, 1.0f, 1.0f);
    Material::MaterialType type = Material::NONE;
    float3 albedo = make_float3(0.f, 0.f, 0.f);
    float optical_density = 1.0f;
    float metal_fuzz = 0.0f;
    bool disable_normal = false;
    bool simple_material = false;
    std::string objfilename;
    std::string mtldir;
    std::string line;
    std::string key;
    // Read configuration
    while (!file.eof()) {
        file >> key;
        if (key == "position") { file >> position.x >> position.y >> position.z; }
        else if (key == "rotation") { file >> rotation.x >> rotation.y >> rotation.z; }
        else if (key == "scale") { file >> scale.x >> scale.y >> scale.z; }
        else if (key == "path") {
            std::getline(file, objfilename);
            objfilename = objfilename.substr(1);
        }
        else if (key == "mtldir") { 
            std::getline(file, mtldir);
            mtldir = mtldir.substr(1);
        }
        else if (key == "albedo") { file >> albedo.x >> albedo.y >> albedo.z; }
        else if (key == "optical_density") { file >> optical_density; }
        else if (key == "metal_fuzz") { file >> metal_fuzz; }
        else if (key == "end") { break; }
        else if (key == "disable_normal") { disable_normal = true; }
        else if (key == "simple_material") { simple_material = true; }
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
    if (type == Material::NONE) {
        std::cerr << "Error: No material type specified" << std::endl;
        return -1;
    }
    // Load object
    std::cout << "Loading object " << objfilename << std::endl;
    if (simple_material) {
        std::cout << "!   Simple Material type: " << material_type_str[type] << std::endl;
    }
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
    auto& materials = reader.GetMaterials();
    int material_offset = materials_.size();
    if (shapes.size() == 0) {
        std::cerr << "Error: No shapes in object file" << std::endl;
        return -1;
    }
    // load shapes
    for (const auto& shape : shapes) {
        std::cout << "    Loading shape " << shape.name << "...";
        // check if the shape has no triangles
        int n_triangles = shape.mesh.indices.size() / 3;
        if (n_triangles == 0) {
            std::cerr << "\n! Warning: Shape " << shape.name << " has no triangles, ignored." << std::endl;
            continue;
        }
        objects_.push_back(RenderObject());
        transforms_.push_back(transform);
        auto& obj = objects_.back();
        obj.n_triangles = n_triangles;
        obj.triangles = (Triangle*)malloc(obj.n_triangles * sizeof(Triangle));
        // Load triangles
        for (size_t i = 0, itri = 0; i < shape.mesh.indices.size(); i += 3, itri++) {
            Triangle& tri = obj.triangles[itri];
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
            tri.material_id = material_offset;
            if (!simple_material && shape.mesh.material_ids[itri] >= 0) {
                tri.material_id += shape.mesh.material_ids[itri] + 1;
            }
        }
        std::cout << obj.n_triangles << " triangles" << std::endl;
    }
    // load materials
    materials_.push_back(make_material(type, albedo, optical_density, metal_fuzz)); // make base material
    if (!simple_material && materials.size() > 0) {
        for (size_t i = 0; i < materials.size(); i++) {
            const auto& mat = materials[i];
            switch (mat.illum) {
            case 1: type = Material::LAMBERTIAN; break;
            case 2: type = Material::METAL; break;
            case 3: type = Material::LIGHT; break;
            case 4: type = Material::REFRACTIVE; break; // custom type glass
            case 5: type = Material::REFLECTIVE; break; // custom type mirror
            default: type = Material::LAMBERTIAN; break;
            }
            float3 albedo;
            if (type == Material::LIGHT) {
                albedo = make_float3(mat.emission[0], mat.emission[1], mat.emission[2]);
            } else {
                albedo = make_float3(mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]);
                if (type == Material::REFRACTIVE) {
                    optical_density = mat.ior;
                    albedo = (1.f - mat.dissolve) * make_float3(1.f, 1.f, 1.f) + mat.dissolve * albedo;
                }
            }
            materials_.push_back(make_material(type, albedo, optical_density, metal_fuzz));
            if (!mat.diffuse_texname.empty()) {
                std::string texname = mtldir + "/" + mat.diffuse_texname;
                std::cout << "    Loading texture " << texname << std::endl;
                if (load_texture(texname, materials_.back()) < 0) {
                    return -1;
                }
            }
        }
    }
    return 0;
}

int SceneLoader::load_texture(const std::string& filename, Material& material) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Error: Cannot load texture " << filename << std::endl;
        return -1;
    }
    cv::flip(img, img, 0); // flip image
    cv::cvtColor(img, img, cv::COLOR_BGR2RGBA);
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
    cudaArray* cu_array;
    cudaMallocArray(&cu_array, &channelDesc, img.cols, img.rows);
    cudaMemcpy2DToArray(cu_array, 0, 0, img.data, img.step, 
        img.cols * sizeof(uchar4), img.rows, cudaMemcpyHostToDevice);
    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeArray;
    res_desc.res.array.array = cu_array;
    cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0] = cudaAddressModeBorder;
    tex_desc.addressMode[1] = cudaAddressModeBorder;
    tex_desc.filterMode = cudaFilterModeLinear;
    tex_desc.readMode = cudaReadModeNormalizedFloat;
    tex_desc.normalizedCoords = 1;
    cudaTextureObject_t tex_obj;
    cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr);
    material.cu_array = cu_array;
    material.tex_obj = tex_obj;
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
        else if (config_key == "gamma") {
            file >> config_.gamma;
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
        else if (config_key == "enable_blend") {
            config_.blend = true;
        }
        else if (config_key == "msaa") {
            file >> config_.msaa;
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
    cudaDeviceSynchronize();
    cudaStream_t streams[4];
    for (int i = 0; i < 4; i++) {
        cudaStreamCreate(&streams[i]);
    }
    n_objects_ = objects_.size();
    std::vector<BVHNode> bvh;
    std::vector<Triangle*> h_tris(n_objects_, nullptr);
    for (int i = 0; i < n_objects_; i++) {
        std::cout << "Copying object\t" << (i + 1) << " /\t" <<  n_objects_ << '\r' << std::flush;
        Triangle* d_tri;
        h_tris[i] = objects_[i].triangles;
        auto& stream = streams[i % 4];
        int n_tri = objects_[i].n_triangles;
        cudaStreamSynchronize(stream);
        cudaMallocAsync(&d_tri, n_tri * sizeof(Triangle), stream);
        cudaMemcpyAsync(d_tri, objects_[i].triangles, n_tri * sizeof(Triangle), 
            cudaMemcpyHostToDevice, stream);
        objects_[i].triangles = d_tri;
        apply_transform(transforms_[i], d_tri, objects_[i].n_triangles, stream);
        cudaMemcpyAsync(h_tris[i], d_tri, objects_[i].n_triangles * sizeof(Triangle), 
            cudaMemcpyDeviceToHost, stream);
        #ifdef VRT_DEBUG // print transformed triangles
        for (int j = 0; j < n_tri; j++) {
            std::cout << "\nTriangle " << j << "\tTexture: " << h_tris[i][j].material_id << std::endl;
            // position and normal
            for (int k = 0; k < 3; k++) {
                std::cout << "    Position: " << h_tris[i][j].v[k].position.x << " " << h_tris[i][j].v[k].position.y << " " << h_tris[i][j].v[k].position.z;
                std::cout << "\tNormal: " << h_tris[i][j].v[k].normal.x << " " << h_tris[i][j].v[k].normal.y << " " << h_tris[i][j].v[k].normal.z << std::endl;
            }
        }
        #endif
    }
    // build bvh tree
    std::cout << "\nBuilding BVH tree... ";
    int n_bvh_node_total = 0;
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
        n_bvh_node_total += n_bvh;
    }
    std::cout << "Done. total " << n_bvh_node_total << " nodes" << std::endl;
    for (int i = 0; i < 4; i++) {
        cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaMalloc(&d_objects_, n_objects_ * sizeof(RenderObject));
    cudaMemcpy(d_objects_, objects_.data(), n_objects_ * sizeof(RenderObject), cudaMemcpyHostToDevice);
    // print all the materials
    #ifdef VRT_DEBUG
    std::cout << "Materials:" << std::endl;
    for (size_t i = 0; i < materials_.size(); i++) {
        std::cout << "    Material " << i << ": " << material_type_str[materials_[i].type] << std::endl;
        std::cout << "        Albedo: " << materials_[i].albedo.x << " " << materials_[i].albedo.y << " " << materials_[i].albedo.z << std::endl;
        std::cout << "        Optical density: " << materials_[i].optical_density << std::endl;
        std::cout << "        Metal fuzz: " << materials_[i].metal_fuzz << std::endl;
        // is simple material
        if (materials_[i].tex_obj) {
            std::cout << "        Texture: Yes" << std::endl;
        }
    }
    #endif
    // copy materials to device
    cudaMalloc(&d_materials_, materials_.size() * sizeof(Material));
    cudaMemcpy(d_materials_, materials_.data(), materials_.size() * sizeof(Material), cudaMemcpyHostToDevice);
    return 0;
}

}