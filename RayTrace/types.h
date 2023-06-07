#pragma once

#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>

using dtype = float;
using vec3 = typename cv::Vec<dtype, 3>;
using color_t = cv::Scalar;

struct Ray {
	vec3 source;

	vec3 direction;
};