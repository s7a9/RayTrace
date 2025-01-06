#include "sceneloader.h"
#include "worker.h"

#include <thread>

using namespace vrt;
using namespace std;

int main(int argc, char** argv) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <scene_file> <output_dir> <batch_size>" << endl;
        return 0;
    }
    int batch_size = atoi(argv[3]);
    Worker worker(argv[1], argv[2], batch_size);
    worker.run();
    worker.force_render();
    worker.stop();
    return 0;
}