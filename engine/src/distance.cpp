#include "vectordb/distance.hpp"
#include <cmath>

namespace vectordb {

float l2_squared(const float* a, const float* b, uint32_t dim) noexcept {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        float d0 = a[i]     - b[i];
        float d1 = a[i + 1] - b[i + 1];
        float d2 = a[i + 2] - b[i + 2];
        float d3 = a[i + 3] - b[i + 3];
        float d4 = a[i + 4] - b[i + 4];
        float d5 = a[i + 5] - b[i + 5];
        float d6 = a[i + 6] - b[i + 6];
        float d7 = a[i + 7] - b[i + 7];
        s0 += d0 * d0;
        s1 += d1 * d1;
        s2 += d2 * d2;
        s3 += d3 * d3;
        s4 += d4 * d4;
        s5 += d5 * d5;
        s6 += d6 * d6;
        s7 += d7 * d7;
    }
    float total = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        total += d * d;
    }
    return total;
}

float dot_product(const float* a, const float* b, uint32_t dim) noexcept {
    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    float s4 = 0.0f, s5 = 0.0f, s6 = 0.0f, s7 = 0.0f;
    uint32_t i = 0;
    for (; i + 8 <= dim; i += 8) {
        s0 += a[i]     * b[i];
        s1 += a[i + 1] * b[i + 1];
        s2 += a[i + 2] * b[i + 2];
        s3 += a[i + 3] * b[i + 3];
        s4 += a[i + 4] * b[i + 4];
        s5 += a[i + 5] * b[i + 5];
        s6 += a[i + 6] * b[i + 6];
        s7 += a[i + 7] * b[i + 7];
    }
    float total = (s0 + s1) + (s2 + s3) + (s4 + s5) + (s6 + s7);
    for (; i < dim; ++i) {
        total += a[i] * b[i];
    }
    return total;
}

float vector_norm_sq(const float* a, uint32_t dim) noexcept {
    return dot_product(a, a, dim);
}

float vector_norm(const float* a, uint32_t dim) noexcept {
    return std::sqrt(vector_norm_sq(a, dim));
}

float cosine_distance_with_norms(const float* a, float norm_a, const float* b, float norm_b, uint32_t dim) noexcept {
    constexpr float kEps = 1e-12f;
    if (norm_a <= kEps || norm_b <= kEps) {
        return 1.0f;
    }
    float cos_sim = dot_product(a, b, dim) / (norm_a * norm_b);
    if (cos_sim > 1.0f) cos_sim = 1.0f;
    if (cos_sim < -1.0f) cos_sim = -1.0f;
    return 1.0f - cos_sim;
}

float cosine_distance(const float* a, const float* b, uint32_t dim) noexcept {
    float s_ab = 0.0f, s_aa = 0.0f, s_bb = 0.0f;
    float s_ab2 = 0.0f, s_aa2 = 0.0f, s_bb2 = 0.0f;
    uint32_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float a0 = a[i], a1 = a[i + 1], a2 = a[i + 2], a3 = a[i + 3];
        float b0 = b[i], b1 = b[i + 1], b2 = b[i + 2], b3 = b[i + 3];
        s_ab  += a0 * b0 + a1 * b1;
        s_ab2 += a2 * b2 + a3 * b3;
        s_aa  += a0 * a0 + a1 * a1;
        s_aa2 += a2 * a2 + a3 * a3;
        s_bb  += b0 * b0 + b1 * b1;
        s_bb2 += b2 * b2 + b3 * b3;
    }
    float ab = s_ab + s_ab2;
    float aa = s_aa + s_aa2;
    float bb = s_bb + s_bb2;
    for (; i < dim; ++i) {
        ab += a[i] * b[i];
        aa += a[i] * a[i];
        bb += b[i] * b[i];
    }
    constexpr float kEps = 1e-12f;
    if (aa <= kEps || bb <= kEps) {
        return 1.0f;
    }
    float cos_sim = ab / std::sqrt(aa * bb);
    if (cos_sim > 1.0f) cos_sim = 1.0f;
    if (cos_sim < -1.0f) cos_sim = -1.0f;
    return 1.0f - cos_sim;
}

float compute_distance(const float* a, const float* b, uint32_t dim, MetricType metric) noexcept {
    switch (metric) {
        case MetricType::L2:
            return l2_squared(a, b, dim);
        case MetricType::COSINE:
            return cosine_distance(a, b, dim);
        case MetricType::DOT:
            return -dot_product(a, b, dim);
    }
    return l2_squared(a, b, dim);
}

float distance_to_score(float distance, MetricType metric) noexcept {
    switch (metric) {
        case MetricType::L2:
            return 1.0f / (1.0f + distance);
        case MetricType::COSINE:
            return 1.0f - distance;
        case MetricType::DOT:
            return -distance;
    }
    return -distance;
}

void normalize_vector(float* vec, uint32_t dim) noexcept {
    float n = vector_norm(vec, dim);
    if (n <= 1e-12f) return;
    float inv = 1.0f / n;
    for (uint32_t i = 0; i < dim; ++i) {
        vec[i] *= inv;
    }
}

} // namespace vectordb
