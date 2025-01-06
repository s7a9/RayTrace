#ifndef SCENELOADER_H
#define SCENELOADER_H
/* This file contains the scene loader which loads the scene data from a file.
 * The scene loader would read the configuration file and load the scene data
 * Then convert the scene data to the format that the worker can use.
*/

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include "renderobject.cuh"
#include "transform.cuh"

namespace vrt {

class SceneLoader {
    std::vector<RenderObject> objects_;
    std::vector<ModelTransform> transforms_;
    std::vector<Material> materials_;
    RenderObject* d_objects_;
    Material* d_materials_;
    RenderConfig config_;
    int n_objects_;

    int load_object(std::ifstream& file);
    int load_texture(const std::string& filename, Material& material);

public:
    SceneLoader();
    ~SceneLoader();

    int load_scene(const std::string& filename);

    const RenderObject* host_object() const { return objects_.data(); }
    const RenderObject* device_objects() const { return d_objects_; }
    int num_object() const { return n_objects_; }

    const Material* device_materials() const { return d_materials_; }
    int num_material() const { return materials_.size(); }

    RenderConfig& config() { return config_; }
};

}

#endif // !SCENELOADER_H