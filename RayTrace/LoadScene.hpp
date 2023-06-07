#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "RenderObject.h"
#include "Camera.h"

vec3 rect1_vs[4] = { {30, 0, -10}, {30, 0, 10}, {0, 8, 10}, {0, 8, -10} };
vec3 rect2_vs[4] = { {30, 0, -10}, {30, 0, 10}, {0, -8, 10}, {0, -8, -10} };
vec3 rect3_vs[4] = { {0, -100, -10}, {0, 100, -10}, {100, 100, -10}, {100, -100, -10} };
vec3 rect4_vs[4] = { {10, -5, -9.5}, {10, 5, -9.5}, {0, 5, -9.5}, {0, -5, -9.5} };

vec3 envir_color{0.2, 0.2, 0.2};

cv::Mat texture1 = cv::imread(".\\1.bmp");

std::vector<const RenderObject*> load_scene() {
	RenderObject* rect1 = new Polygon<4>(rect1_vs, { 1.0, 1.0, 1.0 }, 
		RenderObject::Reflective);
	RenderObject* rect2 = new Polygon<4>(rect2_vs, { 1.0, 1.0, 1.0 },
		RenderObject::Reflective);
	rect1->opticalTraitValue[1] = rect2->opticalTraitValue[1] = 0.8;
	RenderObject* rect3 = new Polygon<4>(rect3_vs, { 1.0, 0, 0 },
		RenderObject::Diffuse);
	RenderObject* rect4 = new Texture(rect4_vs, RenderObject::Emissive, &texture1);
	RenderObject* tri1 = new Triangle({ 30, 0, -4 }, { 15, -5, -10 }, { 15, 5, -10 }, { 0, 0, 1 },
		RenderObject::Emissive);
	rect3->opticalTraitValue[3] = 0.6;
	RenderObject* sphere1 = new SphereLen({ 4, 0, -5 }, 2, { 1, 1, 1 }, RenderObject::Refractive, 1.5);
	return std::vector<const RenderObject*> {
		rect1, rect2, sphere1, rect3, tri1, rect4
	};
}

Camera* load_camera(int width, int height) {
	return new PerspectiveCamera(width, height, { -5, 0, 3 }, { 1, 0, -0.6 }, 56);
}