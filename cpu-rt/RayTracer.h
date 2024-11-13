#pragma once

#include <vector>
#include "RenderObject.h"

bool RayTrace(const std::vector<const RenderObject*>& objs,
const Ray& ray, int sample_n, dtype(*rand_float)(), const color_t& envir, color_t& result,
int depth = 6);
