#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include "RenderObject.h"
#include "Camera.h"

vec3 rect1_vs[4] = { {-10, 0, 0}, {-10, 0, 20}, {-10, 20, 20}, {-10, 20, 0} };
vec3 rect2_vs[4] = { {-10, 20, 20}, {10, 20, 20}, {10, 20, 0}, {-10, 20, 0} };
vec3 rect3_vs[4] = { {10, 20, 20}, {10, 0, 20}, {10, 0, 0}, {10, 20, 0} };
vec3 rect4_vs[4] = { {-10, 20, 0}, {10, 20, 0}, {10, 0, 0}, {-10, 0, 0} };
vec3 rect5_vs[4] = { {-10, 20, 20}, {10, 20, 20}, {10, 0, 20}, {-10, 0, 20} };
vec3 rect6_vs[4] = { {-5, 15, 19.999}, {5, 15, 19.999}, {5, 5, 19.999}, {-5, 5, 19.999} };

vec3 envir_color;

cv::Mat texture1 = cv::imread(".\\1.png");

std::vector<const RenderObject*> load_scene() {
	RenderObject* upper = new Rectangle(rect5_vs, { 1, 1, 1 }, RenderObject::Diffuse);
	RenderObject* light = new Rectangle(rect6_vs, { 1, 1, 1 }, RenderObject::Emissive);
	RenderObject* left = new Rectangle(rect1_vs, { 1, 1, 1 }, RenderObject::Diffuse);
	RenderObject* right = new Rectangle(rect3_vs, { 1, 1, 1 }, RenderObject::Diffuse);
	RenderObject* bottom = new Rectangle(rect4_vs, { 1, 1, 1 }, RenderObject::Diffuse);
	//RenderObject* front = new Rectangle(rect2_vs, { 1, 1, 1 }, RenderObject::Diffuse);
	RenderObject* front = new Texture(rect2_vs, RenderObject::Emissive, &texture1);
	RenderObject* sphere1 = new Sphere({ 6, 5, 3 }, 3, { 1, 1, 0 }, RenderObject::Reflective);
	RenderObject* sphere3 = new Sphere({ -5, 10, 2 }, 2, { 0, 1, 1 }, RenderObject::Reflective | RenderObject::Emissive);
	sphere3->opticalTraitValue[0] = 0.2;
	sphere3->opticalTraitValue[1] = 0.8;
	RenderObject* sphere2 = new SphereLen({ 0, 5, 10 }, 3, {1, 1, 1}, 0, 1.5);
	return std::vector<const RenderObject*> {
		upper, left, right, bottom, front, sphere1, sphere3, sphere2
	};
}

Camera* load_camera(int width, int height) {
	return new PerspectiveCamera(width, height, { 0, -10, 10 }, { 0, 1, 0 }, 60);
}