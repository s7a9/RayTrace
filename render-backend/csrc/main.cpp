#include "sceneloader.h"
#include "worker.h"

#include <thread>

using namespace vrt;
using namespace std;

int main(int argc, char** argv) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <scene_file> <output_dir>" << endl;
        return 0;
    }
    Worker worker(argv[1], argv[2]);
    worker.run();
    worker.update_camera(make_float3(0.0f, 0.0f, 0.0f), make_float3(0.0f, 0.0f, 0.0f));
    std::this_thread::sleep_for(std::chrono::seconds(1));
    worker.stop();
    return 0;
}