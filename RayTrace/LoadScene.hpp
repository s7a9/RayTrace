#pragma once

#include <vector>
#include "ReflectObject.h"
#include "Camera.h"

vec3 rect1_vs[4] = { {30, 0, -10}, {30, 0, 10}, {0, 8, 10}, {0, 8, -10} };
vec3 rect2_vs[4] = { {30, 0, -10}, {30, 0, 10}, {0, -8, 10}, {0, -8, -10} };
vec3 rect3_vs[4] = { {0, -100, -10}, {0, 100, -10}, {100, 100, -10}, {100, -100, -10} };

vec3 envir_color{0.2, 0.2, 0.2};

std::vector<const RenderObject*> load_scene() {
	RenderObject* rect1 = new Polygon<4>(rect1_vs, { 1.0, 1.0, 1.0 }, 
		RenderObject::Reflective);
	RenderObject* rect2 = new Polygon<4>(rect2_vs, { 1.0, 1.0, 1.0 },
		RenderObject::Reflective);
	rect1->opticalTraitValue[1] = rect2->opticalTraitValue[1] = 0.8;
	RenderObject* rect3 = new Polygon<4>(rect3_vs, { 0, 0, 1.0 },
		RenderObject::Diffuse);
	RenderObject* tri1 = new Triangle({ 30, 0, 0 }, { 15, -5, -10 }, { 15, 5, -10 }, { 0, 0, 1 },
		RenderObject::Emissive);
	rect3->opticalTraitValue[3] = 0.6;
	RenderObject* sphere1 = new Sphere({ 6, 0, -3 }, 2, { 1, 1, 1 }, RenderObject::Emissive);
	return std::vector<const RenderObject*> {
		rect1, rect2, sphere1, rect3, tri1
	};
}

Camera* load_camera(int width, int height) {
	return new PerspectiveCamera(width, height, { -5, 0, 5 }, { 1, 0, -0.5 }, 56);
}