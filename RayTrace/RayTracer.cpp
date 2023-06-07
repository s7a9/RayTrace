#include "RayTracer.h"

#include <cmath>

const RenderObject* hit_all(const std::vector<const RenderObject*>& objs,
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
	if (depth == 8) return envir;
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
		// make norm dot (dir + norm) equals zero
		if (obj->opticalTrait & 14) {
			norm *= -ray.direction.ddot(norm) / norm.ddot(norm);
		}
		if (obj->opticalTrait & (RenderObject::Reflective + RenderObject::Diffuse)) {
			new_ray.direction = 2 * norm + ray.direction;
			color += (obj->opticalTraitValue[1] + 0.2 * obj->opticalTraitValue[3]) *
				RayTrace_recursive(objs, new_ray, depth + 1, envir).mul(obj_color);
		}
		if (obj->opticalTrait & RenderObject::Refractive) {
			dtype obj_refration_index = obj->get_refraction_index(new_ray.source);
			new_ray.direction = ray.direction * (2.0 - obj_refration_index) +
				(1.0 - obj_refration_index) * norm;
			color += obj->opticalTraitValue[2] * 
				RayTrace_recursive(objs, new_ray, depth + 1, envir).mul(obj_color);
		}
		return color;
	}
	return envir;
}

bool RayTrace(const std::vector<const RenderObject*>& objs, const Ray& ray,
	int sample_n, dtype(*rand_float)(), const color_t& envir, color_t& color) {
	Ray new_ray;
	vec3 norm, rd;
	double dis;
	const RenderObject* obj = hit_all(objs, ray, new_ray.source, norm, dis);
	if (!obj) {
		color = envir;
		return false;
	}
	color *= 0.0;
	color_t obj_color = obj->get_color(new_ray.source);
	if (obj->opticalTrait & RenderObject::Emissive) {
		color += obj->opticalTraitValue[0] * obj_color;
	}
	if (obj->opticalTrait & (RenderObject::Reflective + RenderObject::Refractive)) {
		norm *= -ray.direction.ddot(norm) / norm.ddot(norm);
		if (obj->opticalTrait & RenderObject::Reflective) {
			new_ray.direction = 2 * norm + ray.direction;
			color += (obj->opticalTraitValue[1]) *
				RayTrace_recursive(objs, new_ray, 1, envir).mul(obj_color);
		}
		if (obj->opticalTrait & RenderObject::Refractive) {
			dtype obj_refration_index = obj->get_refraction_index(new_ray.source);
			new_ray.direction = ray.direction * (2.0 - obj_refration_index) +
				(1.0 - obj_refration_index) * norm;
			color += obj->opticalTraitValue[2] * RayTrace_recursive(objs, new_ray, 1, envir).mul(obj_color);
		}
	}
	if (obj->opticalTrait & RenderObject::Diffuse) {
		color_t diffuse_color_sum;
		dis = sqrt(norm.ddot(norm));
		norm /= dis;
		for (int i = 0; i < sample_n; ++i) {
			do {
				rd = 2.0 * vec3(rand_float(), rand_float(), rand_float()) - vec3(1.0, 1.0, 1.0);
			} while (rd.dot(rd) > 1);
			new_ray.direction = norm + rd;
			diffuse_color_sum += RayTrace_recursive(objs, new_ray, 4, envir);
		}
		diffuse_color_sum /= sample_n;
		color += obj->opticalTraitValue[3] * diffuse_color_sum.mul(obj_color);
	}
	return true;
}