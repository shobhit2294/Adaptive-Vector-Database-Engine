#pragma once

#include "vectordb/types.hpp"
#include <cstddef>
#include <vector>

namespace vectordb {

float l2_squared(const float* a, const float* b, uint32_t dim) noexcept;
float dot_product(const float* a, const float* b, uint32_t dim) noexcept;
float vector_norm_sq(const float* a, uint32_t dim) noexcept;
float vector_norm(const float* a, uint32_t dim) noexcept;

// Cosine distance in [0, 2]: 1.0 - cosine_similarity. Zero-norm vectors yield 1.0.
float cosine_distance(const float* a, const float* b, uint32_t dim) noexcept;
float cosine_distance_with_norms(const float* a, float norm_a, const float* b, float norm_b, uint32_t dim) noexcept;

// Unified distance where smaller is closer for all metrics.
float compute_distance(const float* a, const float* b, uint32_t dim, MetricType metric) noexcept;

// Converts internal distance into user-facing similarity score (higher = better).
float distance_to_score(float distance, MetricType metric) noexcept;

void normalize_vector(float* vec, uint32_t dim) noexcept;

} // namespace vectordb
