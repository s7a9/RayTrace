#include <thread>
#include <iostream>

#include "worker.h"
#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#ifndef __attribute_maybe_unused__
#define __attribute_maybe_unused__ __attribute__((__unused__))
#endif

// handle sigpipe and ignore it
void sigpipe_handler(int signum) {
    std::cerr << "SIGPIPE " << signum << " caught" << std::endl;
}

int main(int argc, char** argv) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <scene_file> <batch_size> <address> <port>" << std::endl;
        return 0;
    }
    std::string scene_file = argv[1];
    int batch_size = atoi(argv[2]);
    std::string address = argv[3];
    int port = atoi(argv[4]);
    std::string output_dir = "./assets/static";
    std::string index_file = "./assets/static/index.html";
    vrt::Worker worker(scene_file, output_dir, batch_size);
    worker.run();

    // ignore sigpipe
    signal(SIGPIPE, sigpipe_handler);

    worker.force_render();

    using namespace httplib;
    // SSLServer svr("./bin/cert.pem", "./bin/key.pem");
    Server svr;

    svr.Get("/", [index_file](__attribute_maybe_unused__ const Request& req, Response& res) {
        std::ifstream file(index_file);
        if (!file.is_open()) {
            res.status = 404;
            res.set_content("File not found", "text/plain");
            return;
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        res.set_content(content, "text/html");
        std::cout << "Serving index.html" << std::endl;
    });

    // Serve static files
    svr.set_base_dir(output_dir);

    // Extract values from HTTP headers and URL query params
    svr.Post("/update-camera", [&](const Request& req, Response& res) {
        constexpr float scale = 1.f;
        // Parse the value dx,dy,dz
        float3 delta_pos = make_float3(0.0f, 0.0f, 0.0f);
        float3 delta_dir = make_float3(0.0f, 0.0f, 0.0f);
        std::string body_str(req.body);
        std::istringstream body_stream(body_str);
        char comma;
        body_stream >> delta_pos.x >> comma >> delta_pos.y >> comma >> delta_pos.z >> comma;
        body_stream >> delta_dir.x >> comma >> delta_dir.y >> comma >> delta_dir.z;
        delta_pos.x *= scale; delta_pos.y *= scale; delta_pos.z *= scale;
        worker.update_camera(delta_pos, delta_dir);
        res.set_content("updated", "text/plain");
        auto& config = worker.config();
        std::cout << "Camera updated to " << config.camera_pos.x << "," << config.camera_pos.y << "," << config.camera_pos.z << std::endl;
    });

    // Send the most recent rendered image
    svr.Get("/image", [&](__attribute_maybe_unused__ const Request& req, Response& res) {
        char filename[vrt::MaxFilenameLength];
        worker.get_output_filename(filename);
        // load the file and send it back
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            res.status = 404;
            res.set_content("File not found", "text/plain");
            return;
        }
        std::string content((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        res.set_content(content, "image/png");
    });

    // reset camera
    svr.Post("/reset-camera", [&](__attribute_maybe_unused__ const Request& req, Response& res) {
        worker.reset_camera();
        res.set_content("Camera reset", "text/plain");
    });

    // get camera position
    svr.Get("/camera-pos", [&](__attribute_maybe_unused__ const Request& req, Response& res) {
        auto& config = worker.config();
        char buffer[128];
        snprintf(buffer, 128, "%f,%f,%f", config.camera_pos.x, config.camera_pos.y, config.camera_pos.z);
        res.set_content(buffer, "text/plain");
    });

    svr.Get("/stop", [&](__attribute_maybe_unused__ const Request& req, Response& res) {
        worker.stop();
        svr.stop();
        std::cout << "Server stopped" << std::endl;
        res.set_content("Server stopped", "text/plain");
    });

    std::cout << "Starting server at " << address << ":" << port << std::endl;
    if (!svr.listen(address, port)) {
        std::cerr << "Failed to start server" << std::endl;
        return 1;
    }

    return 0;
}
