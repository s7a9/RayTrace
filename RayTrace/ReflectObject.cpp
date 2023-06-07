#include "ReflectObject.h"

#include <cmath>
constexpr dtype EPS = 1e-5;

inline bool checkTriangle(const Ray& ray, vec3& hitPos, vec3& normal, 
	const vec3& p0, const vec3& p1, const vec3& p2) {
	vec3 v1 = p2 - p0, v2 = p1 - p0,
		 v1s = p0 - ray.source;
	normal = v1.cross(v2);
	dtype f1 = normal.dot(ray.direction);
	if (std::abs(f1) < EPS) return false;
	dtype f2 = normal.dot(v1s) / f1;
	if (f2 < 0) return false;
	hitPos = ray.source + f2 * ray.direction;
	if (normal.dot((hitPos - p1).cross(hitPos - p0)) < 0 ||
		normal.dot((hitPos - p2).cross(hitPos - p1)) < 0 || 
		normal.dot((hitPos - p0).cross(hitPos - p2)) < 0) 
		return false;
	if (f1 > 0) normal = -normal;
	return true;
}

bool Triangle::shoot(const Ray& ray, vec3& hitPos, vec3& normal) const {
	return checkTriangle(ray, hitPos, normal, vertices[0], vertices[1], vertices[2]);
}

Sphere::Sphere(vec3 center, dtype radius, color_t color, uint trait):
	center(center), color(color), radius_sqr(radius * radius), RenderObject(trait) {}

bool Sphere::shoot(const Ray& ray, vec3& hitPos, vec3& normal) const {
	vec3 po = center - ray.source;
	double dl = ray.direction.ddot(ray.direction);
	double k = po.ddot(ray.direction) / dl;
	vec3 h = po - ray.direction * k;
	double a = radius_sqr - h.ddot(h);
	if (a <= 0) return false;
	hitPos = ray.source + ray.direction * (k - sqrt(a / dl));
	normal = hitPos - center;
	return true;
}

RenderObject::RenderObject(uint trait)
{
	opticalTrait = trait;
	for (int i = 0; i < 4; ++i)
		opticalTraitValue[i] = (trait & 1 << i) ? 1.0 : 0.0;
}
