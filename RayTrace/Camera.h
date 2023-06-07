#pragma once

#include "types.h"

class Camera
{
protected:
	size_t Width = 0, Height = 0;

public:
	virtual Ray cast(size_t i, size_t j) const = 0;
};

class ParallelCamera : public Camera {
private:
	vec3 source, up, left, dir;

public:
	ParallelCamera(size_t width, size_t height, vec3 src, vec3 up, vec3 left);

	Ray cast(size_t i, size_t j) const;
};

class PerspectiveCamera : public Camera {
private:
	vec3 source, direction, up, left;

public:
	PerspectiveCamera(size_t width, size_t height, 
		vec3 src, vec3 dir, 
		double angle, double rotate = 0);

	Ray cast(size_t i, size_t j) const;
};