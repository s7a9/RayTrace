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
    RenderObject* d_objects_;
    RenderConfig config_;
    int n_objects_;

    int load_object(std::ifstream& file);

public:
    SceneLoader();
    ~SceneLoader();

    int load_scene(const std::string& filename);

    const RenderObject* device_objects() const { return d_objects_; }
    int size() const { return n_objects_; }
    RenderConfig& config() { return config_; }
};

}

#endif // !SCENELOADER_H