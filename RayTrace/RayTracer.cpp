#include "RayTracer.h"

#include <cmath>

inline void refract(vec3 src, const vec3& normal, vec3& refl, dtype ri) {
	src /= sqrtf(src.dot(src));
	float sdn = src.dot(normal);
	float x = 1 - ri * ri * (1 - sdn * sdn);
	if (x < 0) refl = src - 2.0f * sdn * normal;
	else refl = ri * src - (ri * sdn + sqrtf(x)) * normal;
}

inline const RenderObject* hit_all(const std::vector<const RenderObject*>& objs,
	const Ray& ray, vec3& min_hitp, vec3& min_norm, double& min_dis) {
	min_dis = -1.0;
	const RenderObject* ret = nullptr;
	bool flag = false;
	for (auto iter = objs.begin(); iter != objs.end(); ++iter) {
		vec3 norm, hitp, dif;
		double dis;
		if ((*iter)->shoot(ray, hitp, norm)) {
			dif = hitp - ray.source;
			dis = dif.ddot(dif);
			if (dis < 1e-5) continue;
			if (dis < min_dis || !flag) {
				min_dis = dis;
				min_norm = norm;
				min_hitp = hitp;
				ret = *iter;
				flag = true;
			}
		}
	}
	return ret;
}

color_t RayTrace_recursive(const std::vector<const RenderObject*>& objs, 
	const Ray& ray, int depth, const color_t& envir) {
	if (depth == 0) return envir;
	Ray new_ray;
	vec3 norm;
	double dis;
	const RenderObject* obj = hit_all(objs, ray, new_ray.source, norm, dis);
	if (obj) {
		color_t color;
		color_t obj_color = obj->get_color(new_ray.source);
		if (obj->opticalTrait & RenderObject::Emissive) {
			color += obj->opticalTraitValue[0] * obj_color;
		}
		norm /= sqrtf(norm.dot(norm));
		if (obj->opticalTrait & (RenderObject::Reflective + RenderObject::Diffuse)) {
			new_ray.direction = ray.direction - 2.0f * ray.direction.dot(norm) * norm;
			color += (obj->opticalTraitValue[1] + 0.2 * obj->opticalTraitValue[3]) *
				RayTrace_recursive(objs, new_ray, depth - 1, envir).mul(obj_color);
		}
		if (obj->opticalTrait & RenderObject::Refractive) {
			refract(ray.direction, norm, new_ray.direction,
				obj->get_refraction_index(new_ray.source, ray));
			new_ray.source += 0.01 * new_ray.direction;
			color += obj->opticalTraitValue[2] * 
				RayTrace_recursive(objs, new_ray, depth - 1, envir).mul(obj_color);
		}
		return color;
	}
	return envir;
}

bool RayTrace(const std::vector<const RenderObject*>& objs, const Ray& ray,
	int sample_n, dtype(*rand_float)(), const color_t& envir, color_t& color, int depth) {
	if (depth == 0) {
		color = envir;
		return false;
	}
	Ray new_ray;
	vec3 norm, rd;
	double dis;
	const RenderObject* obj = hit_all(objs, ray, new_ray.source, norm, dis);
	if (!obj) {
		color = envir;
		return false;
	}
	color *= 0.0;
	color_t obj_color = obj->get_color(new_ray.source), color_tmp;
	if (obj->opticalTrait & RenderObject::Emissive) {
		color += obj->opticalTraitValue[0] * obj_color;
	}
	norm /= sqrtf(norm.dot(norm));
	if (obj->opticalTrait & (RenderObject::Reflective + RenderObject::Refractive)) {
		if (obj->opticalTrait & RenderObject::Reflective) {
			new_ray.direction = ray.direction - 2.0f * ray.direction.dot(norm) * norm;
			RayTrace(objs, new_ray, sample_n , rand_float, envir, color_tmp, depth - 1);
			color += obj->opticalTraitValue[1] * color_tmp.mul(obj_color);
		}
		if (obj->opticalTrait & RenderObject::Refractive) {
			refract(ray.direction, norm, new_ray.direction,
				obj->get_refraction_index(new_ray.source, ray));
			new_ray.source += 0.01 * new_ray.direction;
			RayTrace(objs, new_ray, sample_n, rand_float, envir, color_tmp, depth - 1);
			color += obj->opticalTraitValue[2] * color_tmp.mul(obj_color);
		}
	}
	if (obj->opticalTrait & RenderObject::Diffuse) {
		color_t diffuse_color_sum;
		for (int i = 0; i < sample_n; ++i) {
			do {
				rd = 2.0 * vec3(rand_float(), rand_float(), rand_float()) - vec3(1.0, 1.0, 1.0);
			} while (rd.dot(rd) > 1);
			new_ray.direction = norm + rd;
			diffuse_color_sum += RayTrace_recursive(objs, new_ray, depth, envir);
		}
		diffuse_color_sum /= sample_n;
		color += obj->opticalTraitValue[3] * diffuse_color_sum.mul(obj_color);
	}
	return true;
}