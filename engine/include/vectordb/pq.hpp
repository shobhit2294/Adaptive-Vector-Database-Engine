#pragma once

#include "vectordb/types.hpp"
#include <cstdint>
#include <string>
#include <vector>

namespace vectordb {

class ProductQuantizer {
public:
    ProductQuantizer() = default;
    ProductQuantizer(uint32_t dim,
                     MetricType metric,
                     uint32_t requested_subspaces = 16,
                     uint32_t centroids = 256);

    // Trains k-means++ codebooks on flattened vectors [count * dim].
    void train(const float* vectors, uint32_t count, uint32_t max_iters = 15, uint64_t seed = 1337);

    // Encodes a single vector into M_pq uint8 codes.
    void encode(const float* vec, uint8_t* out_codes) const;
    std::vector<uint8_t> encode_batch(const float* vectors, uint32_t count) const;

    // Reconstructs approximate vector into out[dim].
    void decode(const uint8_t* codes, float* out) const;

    // Builds [M_pq * K_pq] asymmetric distance lookup table for a query.
    void build_adc_table(const float* query, std::vector<float>& out_table) const;

    // Computes approximate distance using precomputed ADC table.
    float adc_distance(const uint8_t* codes, const float* adc_table) const noexcept;

    bool is_trained() const { return trained_; }
    uint32_t dim() const { return dim_; }
    uint32_t subspaces() const { return m_pq_; }
    uint32_t centroids() const { return k_pq_; }
    uint32_t sub_dim() const { return sub_dim_; }
    MetricType metric() const { return metric_; }
    uint64_t codebook_ram_bytes() const;

    bool save(const std::string& path) const;
    bool load(const std::string& path);

private:
    uint32_t dim_ = 0;
    MetricType metric_ = MetricType::L2;
    uint32_t m_pq_ = 1;
    uint32_t k_pq_ = 256;
    uint32_t sub_dim_ = 0;
    bool trained_ = false;

    // Layout: [m][k][sub_dim] -> m * k_pq_ * sub_dim_ + k * sub_dim_ + d
    std::vector<float> codebooks_;

    static uint32_t choose_subspaces(uint32_t dim, uint32_t requested);
};

} // namespace vectordb
