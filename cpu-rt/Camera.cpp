#include "Camera.h"
#include <cmath>


ParallelCamera::ParallelCamera(size_t width, size_t height, vec3 src, vec3 up, vec3 left)
	:source(src), up(up), left(left) {
	Width = width; Height = height;
	dir = left.cross(up);
}

Ray ParallelCamera::cast(size_t i, size_t j) const
{
	Ray ray;
	dtype xdif = Width, ydif = Height;
	xdif = (xdif / 2 - j - 0.5) / xdif * 2;
	ydif = (i - ydif / 2 + 0.5) / ydif * 2;
	ray.source = source + up * ydif + left * xdif;
	ray.direction = dir;
	return ray;
}

PerspectiveCamera::PerspectiveCamera(size_t width, size_t height, vec3 src, vec3 dir, double angle, double rotate):
	source(src), direction(dir)
{
	Width = width; Height = height;
	double dirlen = sqrt(direction.ddot(direction));
	direction /= dirlen;
	vec3 Z(.0f, .0f, 1.f);
	left = Z.cross(direction);
	rotate *= CV_PI / 180;
	angle *= CV_PI / 180;
	left = cos(rotate) * left + 
		(1 - cos(rotate)) * direction.dot(left) * direction + 
		sin(rotate) * direction.cross(left);
	up = direction.cross(left);
	double leftlen = sqrt(left.ddot(left)), uplen = sqrt(up.ddot(up));
	left *= tan(angle) / leftlen;
	up *= tan(angle) / uplen;
}

Ray PerspectiveCamera::cast(size_t i, size_t j) const
{
	Ray ray;
	dtype xdif = Width, ydif = Height;
	ydif = (ydif / 2 - i - 0.5) / xdif * 2;
	xdif = (xdif / 2 - j - 0.5) / xdif * 2;
	ray.direction = direction + up * ydif + left * xdif;
	ray.source = source;
	return ray;
}
