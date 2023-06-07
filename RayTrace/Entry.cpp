#include "opencv2/opencv.hpp"
#include <iostream>
#include <string>
#include <atomic>
#include <random>
#include <fstream>

#include "RenderObject.h"
#include "LoadScene.hpp"
#include "RayTracer.h"

// === Random float generator ===//
std::uniform_real_distribution<dtype> dis(0.0, 1.0);
std::random_device random_device;
std::mt19937 rand_gen(random_device());

dtype randf() { return dis(rand_gen); }
// ============================= //

size_t Height = 1080, Width = 1920, THREAD_N = 16;
cv::Mat* image;

std::vector<const RenderObject*> scene;
Camera* cam;
// vec3 envir_color{0.5, 0.5, 0.5};
int sample_n = 64;


// Multithread render acceleration //
std::atomic<int> cur_row;

void render_line() {
	while (cur_row < Height) {
		int row = cur_row++;
		if (row * 10 % Height == 0) std::cout << (row * 10) / Height << ' ';
		for (int i = 0; i < Width; ++i) {
			Ray ray = cam->cast(row, i);
			color_t color;
			RayTrace(scene, ray, sample_n, randf, envir_color, color);
			for (int k = 0; k < 3; ++k) {
				image->ptr<uchar>(row, i)[k] = 255 * color[k];
			}
		}
	}
}

int main(int argc, const char** argv) {
	image = new cv::Mat(Height, Width, CV_8UC3);
	cur_row = 0;
	scene = load_scene();
	cam = load_camera(Width, Height);
	std::thread** thdpool = new std::thread*[THREAD_N];
	for (int i = 0; i < THREAD_N; ++i) {
		thdpool[i] = new std::thread(render_line);
		std::this_thread::sleep_for(std::chrono::milliseconds(20));
	}
	for (int i = 0; i < THREAD_N; ++i) {
		thdpool[i]->join();
		delete thdpool[i];
	}
	delete cam;
	delete[] thdpool;
	cv::imshow("show1", *image);
	cv::imwrite("output.jpg", *image);
	cv::waitKey(0);
	delete image;
	return 0;
}