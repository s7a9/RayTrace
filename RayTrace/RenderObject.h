#pragma once

#include "types.h"
#include <initializer_list>

class RenderObject
{
public:
	enum OpticalTrait {
		Emissive	= 0x01,
		Reflective	= 0x02,
		Refractive	= 0x04,
		Diffuse		= 0x08,
	};

	dtype opticalTraitValue[4];

	uint opticalTrait;

	explicit RenderObject(uint trait);

	virtual bool shoot(const Ray& ray, vec3& hitPos, vec3& normal) const = 0;

	virtual color_t get_color(const vec3& pos) const = 0;

	virtual dtype get_refraction_index(const vec3& pos) const { return 0.0; }
};

class Triangle : public RenderObject {
public:
	vec3 vertices[3];

	color_t color;

	Triangle(vec3 p1, vec3 p2, vec3 p3, color_t clr, uint trait):
		color(clr), RenderObject(trait) {
		vertices[0] = p1;
		vertices[1] = p2;
		vertices[2] = p3;
	}

	bool shoot(const Ray& ray, vec3& hitPos, vec3& normal) const;

	virtual color_t get_color(const vec3& pos) const { return color; }
};

template <size_t V>
class Polygon : public RenderObject {
protected:
	vec3 vertices[V];

	color_t color;

public:
	Polygon(vec3* vts, color_t color, uint trait) : 
		color(color), RenderObject(trait) {
		for (size_t i = 0; i < V; ++i) vertices[i] = vts[i];
		opticalTrait = trait;
	}

	bool shoot(const Ray& ray, vec3& hitPos, vec3& normal) const;

	virtual color_t get_color(const vec3& pos) const { return color; }
};

using Rectangle = Polygon<4>;

class Sphere : public RenderObject {
public:
	vec3 center;

	dtype radius_sqr;

	color_t color;

	Sphere(vec3 center, dtype radius, color_t color, uint trait);

	bool shoot(const Ray& ray, vec3& hitPos, vec3& normal) const;

	virtual color_t get_color(const vec3& pos) const { return color; }
};

inline bool checkTriangle(const Ray& ray, vec3& hitPos, vec3& normal,
	const vec3& p0, const vec3& p1, const vec3& p2);

template<size_t V>
inline bool Polygon<V>::shoot(const Ray& ray, vec3& hitPos, vec3& normal) const {
	for (size_t i = 1; i + 1 < V; ++i) {
		if (checkTriangle(ray, hitPos, normal, vertices[0], vertices[i], vertices[i + 1]))
			return true;
	}
	return false;
}

class Texture : public Polygon<4> {
private:
	cv::Mat* img;

	dtype wlen2, hlen2;

	vec3 v1, v2;

public:
	Texture(vec3* vts, uint trait, cv::Mat* image);

	color_t get_color(const vec3& pos) const;
};

class SphereLen : public Sphere {
private:
	dtype refraction_index;

public:
	SphereLen(vec3 center, dtype radius, color_t color, uint trait, dtype refraction_index);

	dtype get_refraction_index(const vec3& pos) const;
};