#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>

#include <thrust/device_vector.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>

namespace vrt {

constexpr int BLOCK_SIZE = 8;
constexpr int MAX_MAT_PER_OBJECT = 8;
constexpr int MAX_HIT_STACK_SIZE = 32;

struct RenderConfig {
    bool blend;
    int msaa;
    int max_depth;
    int max_surface;
    int width;
    int height;
    int n_samples;
    int batch_size;
    float fov;
    float gamma;
    float russian_roulette;
    float3 background;
    float3 camera_pos;
    float3 camera_dir;
    float3 camera_up;
};


template<class T> struct needs_qivs_arithmetic_operators : public std::false_type {};
template<> struct needs_qivs_arithmetic_operators<float3> : public std::true_type {};
template<> struct needs_qivs_arithmetic_operators<float4> : public std::true_type {};

#define ATTRIBUTES __forceinline__ __host__ __device__

// float3
ATTRIBUTES float3& operator+=(float3& a, const float3& b) {
    a.x += b.x; a.y += b.y; a.z += b.z;
    return a;
}

ATTRIBUTES float3& operator*=(float3& a, float b) {
    a.x *= b; a.y *= b; a.z *= b;
    return a;
}

ATTRIBUTES float3& operator*=(float3& a, const float3& b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z;
    return a;
}

// float4
ATTRIBUTES float4& operator+=(float4& a, const float4& b) {
    a.x += b.x; a.y += b.y; a.z += b.z; a.w += b.w;
    return a;
}

ATTRIBUTES float4& operator*=(float4& a, float b) {
    a.x *= b; a.y *= b; a.z *= b; a.w *= b;
    return a;
}

ATTRIBUTES float4& operator*=(float4& a, const float4& b) {
    a.x *= b.x; a.y *= b.y; a.z *= b.z; a.w *= b.w;
    return a;
}

// normailize
ATTRIBUTES float3 normalize(const float3& a) {
    float norm = sqrtf(a.x*a.x + a.y*a.y + a.z*a.z);
    return {a.x/norm, a.y/norm, a.z/norm};
}

ATTRIBUTES float dot3(const float3& a, const float3& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z;
}

ATTRIBUTES float dot4(const float4& a, const float4& b) {
    return a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
}

ATTRIBUTES float3 cross(const float3& a, const float3& b) {
    return make_float3(
        a.y*b.z - a.z*b.y,
        a.z*b.x - a.x*b.z,
        a.x*b.y - a.y*b.x
    );
}

// Generalize += and *= to +, -=, -, *, /= and /
template<typename Pt> ATTRIBUTES
typename std::enable_if<needs_qivs_arithmetic_operators<Pt>::value, float>::type
get(const Pt& a, int i) {
    return (&a.x)[i];
}

template<typename Pt> ATTRIBUTES
typename std::enable_if<needs_qivs_arithmetic_operators<Pt>::value, Pt>::type
operator+(const Pt& a, const Pt& b) {
    auto sum = a;
    sum += b;
    return sum;
}

template<typename Pt> ATTRIBUTES
typename std::enable_if<needs_qivs_arithmetic_operators<Pt>::value, Pt>::type&
operator-=(Pt& a, const Pt& b) {
    a += make_float3(-b.x, -b.y, -b.z);
    return a;
}

template<typename Pt> ATTRIBUTES
typename std::enable_if<needs_qivs_arithmetic_operators<Pt>::value, Pt>::type
operator-(const Pt& a, const Pt& b) {
    auto diff = a;
    diff -= b;
    return diff;
}

template<typename Pt> ATTRIBUTES
typename std::enable_if<needs_qivs_arithmetic_operators<Pt>::value, Pt>::type
operator-(const Pt& a) {
    return -1.0f * a;
}

template<typename Pt> ATTRIBUTES
typename std::enable_if<needs_qivs_arithmetic_operators<Pt>::value, Pt>::type
operator*(const Pt& a, float b) {
    auto prod = a;
    prod *= b;
    return prod;
}

template<typename Pt> ATTRIBUTES
typename std::enable_if<needs_qivs_arithmetic_operators<Pt>::value, Pt>::type
operator*(float b, const Pt& a) {
    auto prod = a;
    prod *= b;
    return prod;
}

template<typename Pt> ATTRIBUTES
typename std::enable_if<needs_qivs_arithmetic_operators<Pt>::value, Pt>::type&
operator/=(Pt& a, float b) {
    a *= 1.f/b;
    return a;
}

template<typename Pt> ATTRIBUTES
typename std::enable_if<needs_qivs_arithmetic_operators<Pt>::value, Pt>::type
operator/(const Pt& a, float b) {
    auto quot = a;
    quot /= b;
    return quot;
}

#undef ATTRIBUTES

}

#endif // !COMMON_CUH