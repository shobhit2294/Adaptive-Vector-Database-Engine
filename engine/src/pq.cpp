#include "vectordb/pq.hpp"
#include "vectordb/distance.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace vectordb {

namespace {
constexpr uint32_t kPqMagic = 0x50514342; // "PQCB"
constexpr uint32_t kPqVersion = 1;
} // namespace

uint32_t ProductQuantizer::choose_subspaces(uint32_t dim, uint32_t requested) {
    if (dim == 0) return 1;
    uint32_t m = std::min(dim, std::max(1u, requested));
    while (m > 1 && (dim % m) != 0) {
        --m;
    }
    return m;
}

ProductQuantizer::ProductQuantizer(uint32_t dim,
                                   MetricType metric,
                                   uint32_t requested_subspaces,
                                   uint32_t centroids)
    : dim_(dim),
      metric_(metric),
      m_pq_(choose_subspaces(dim, requested_subspaces)),
      k_pq_(std::clamp(centroids, 2u, 256u)),
      sub_dim_(dim_ / m_pq_) {
    if (dim_ == 0) {
        throw std::invalid_argument("PQ dimension must be > 0");
    }
}

void ProductQuantizer::train(const float* vectors, uint32_t count, uint32_t max_iters, uint64_t seed) {
    if (count == 0 || vectors == nullptr) {
        throw std::invalid_argument("Cannot train PQ on zero vectors");
    }

    // Subsample up to 4096 training vectors for fast codebook construction
    std::mt19937_64 rng(seed);
    const uint32_t sample_n = std::min(count, 4096u);
    std::vector<uint32_t> indices(count);
    std::iota(indices.begin(), indices.end(), 0);
    if (count > sample_n) {
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(sample_n);
    }

    // Prepare normalized copies if metric is COSINE
    std::vector<float> train_buf(static_cast<size_t>(sample_n) * dim_);
    for (uint32_t i = 0; i < sample_n; ++i) {
        const float* src = vectors + static_cast<size_t>(indices[i]) * dim_;
        float* dst = train_buf.data() + static_cast<size_t>(i) * dim_;
        std::copy(src, src + dim_, dst);
        if (metric_ == MetricType::COSINE) {
            normalize_vector(dst, dim_);
        }
    }

    const uint32_t actual_k = std::min(k_pq_, sample_n);
    k_pq_ = std::max(1u, actual_k);
    codebooks_.assign(static_cast<size_t>(m_pq_) * k_pq_ * sub_dim_, 0.0f);

    std::vector<float> min_dist_sq(sample_n);
    std::vector<uint32_t> assign(sample_n);
    std::vector<float> sums(static_cast<size_t>(k_pq_) * sub_dim_);
    std::vector<uint32_t> counts(k_pq_);

    for (uint32_t m = 0; m < m_pq_; ++m) {
        const uint32_t sub_off = m * sub_dim_;
        float* cb = codebooks_.data() + static_cast<size_t>(m) * k_pq_ * sub_dim_;

        // K-means++ initialization on subspace m
        uint32_t first = static_cast<uint32_t>(rng() % sample_n);
        const float* first_ptr = train_buf.data() + static_cast<size_t>(first) * dim_ + sub_off;
        std::copy(first_ptr, first_ptr + sub_dim_, cb);

        for (uint32_t i = 0; i < sample_n; ++i) {
            const float* p = train_buf.data() + static_cast<size_t>(i) * dim_ + sub_off;
            min_dist_sq[i] = l2_squared(p, cb, sub_dim_);
        }

        for (uint32_t c = 1; c < k_pq_; ++c) {
            double total = 0.0;
            for (float d : min_dist_sq) total += d;

            uint32_t chosen = static_cast<uint32_t>(rng() % sample_n);
            if (total > 1e-12) {
                std::uniform_real_distribution<double> udist(0.0, total);
                double target = udist(rng);
                double acc = 0.0;
                for (uint32_t i = 0; i < sample_n; ++i) {
                    acc += min_dist_sq[i];
                    if (acc >= target) {
                        chosen = i;
                        break;
                    }
                }
            }
            const float* cptr = train_buf.data() + static_cast<size_t>(chosen) * dim_ + sub_off;
            std::copy(cptr, cptr + sub_dim_, cb + static_cast<size_t>(c) * sub_dim_);

            for (uint32_t i = 0; i < sample_n; ++i) {
                const float* p = train_buf.data() + static_cast<size_t>(i) * dim_ + sub_off;
                float d = l2_squared(p, cb + static_cast<size_t>(c) * sub_dim_, sub_dim_);
                if (d < min_dist_sq[i]) min_dist_sq[i] = d;
            }
        }

        // Lloyd iterations
        for (uint32_t iter = 0; iter < max_iters; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0u);

            for (uint32_t i = 0; i < sample_n; ++i) {
                const float* p = train_buf.data() + static_cast<size_t>(i) * dim_ + sub_off;
                uint32_t best_c = 0;
                float best_d = std::numeric_limits<float>::max();
                for (uint32_t c = 0; c < k_pq_; ++c) {
                    float d = l2_squared(p, cb + static_cast<size_t>(c) * sub_dim_, sub_dim_);
                    if (d < best_d) {
                        best_d = d;
                        best_c = c;
                    }
                }
                assign[i] = best_c;
                ++counts[best_c];
                float* s_ptr = sums.data() + static_cast<size_t>(best_c) * sub_dim_;
                for (uint32_t d = 0; d < sub_dim_; ++d) {
                    s_ptr[d] += p[d];
                }
            }

            for (uint32_t c = 0; c < k_pq_; ++c) {
                float* c_ptr = cb + static_cast<size_t>(c) * sub_dim_;
                if (counts[c] > 0) {
                    float inv = 1.0f / static_cast<float>(counts[c]);
                    const float* s_ptr = sums.data() + static_cast<size_t>(c) * sub_dim_;
                    for (uint32_t d = 0; d < sub_dim_; ++d) {
                        c_ptr[d] = s_ptr[d] * inv;
                    }
                } else {
                    // Re-seed empty cluster from a random training sample with slight perturbation
                    uint32_t ridx = static_cast<uint32_t>(rng() % sample_n);
                    const float* r_ptr = train_buf.data() + static_cast<size_t>(ridx) * dim_ + sub_off;
                    std::copy(r_ptr, r_ptr + sub_dim_, c_ptr);
                }
            }
        }
    }

    trained_ = true;
}

void ProductQuantizer::encode(const float* vec, uint8_t* out_codes) const {
    if (!trained_) {
        throw std::logic_error("ProductQuantizer must be trained before encode");
    }
    std::vector<float> norm_copy;
    const float* src = vec;
    if (metric_ == MetricType::COSINE) {
        norm_copy.assign(vec, vec + dim_);
        normalize_vector(norm_copy.data(), dim_);
        src = norm_copy.data();
    }

    for (uint32_t m = 0; m < m_pq_; ++m) {
        const float* p = src + static_cast<size_t>(m) * sub_dim_;
        const float* cb = codebooks_.data() + static_cast<size_t>(m) * k_pq_ * sub_dim_;
        uint32_t best_c = 0;
        float best_d = std::numeric_limits<float>::max();
        for (uint32_t c = 0; c < k_pq_; ++c) {
            float d = l2_squared(p, cb + static_cast<size_t>(c) * sub_dim_, sub_dim_);
            if (d < best_d) {
                best_d = d;
                best_c = c;
            }
        }
        out_codes[m] = static_cast<uint8_t>(best_c);
    }
}

std::vector<uint8_t> ProductQuantizer::encode_batch(const float* vectors, uint32_t count) const {
    std::vector<uint8_t> codes(static_cast<size_t>(count) * m_pq_);
    for (uint32_t i = 0; i < count; ++i) {
        encode(vectors + static_cast<size_t>(i) * dim_, codes.data() + static_cast<size_t>(i) * m_pq_);
    }
    return codes;
}

void ProductQuantizer::decode(const uint8_t* codes, float* out) const {
    if (!trained_) {
        throw std::logic_error("ProductQuantizer must be trained before decode");
    }
    for (uint32_t m = 0; m < m_pq_; ++m) {
        uint32_t c = std::min<uint32_t>(codes[m], k_pq_ - 1);
        const float* cb = codebooks_.data() + (static_cast<size_t>(m) * k_pq_ + c) * sub_dim_;
        std::copy(cb, cb + sub_dim_, out + static_cast<size_t>(m) * sub_dim_);
    }
}

void ProductQuantizer::build_adc_table(const float* query, std::vector<float>& out_table) const {
    if (!trained_) {
        throw std::logic_error("ProductQuantizer must be trained before building ADC table");
    }
    std::vector<float> q_copy;
    const float* q = query;
    if (metric_ == MetricType::COSINE) {
        q_copy.assign(query, query + dim_);
        normalize_vector(q_copy.data(), dim_);
        q = q_copy.data();
    }

    out_table.resize(static_cast<size_t>(m_pq_) * k_pq_);
    for (uint32_t m = 0; m < m_pq_; ++m) {
        const float* q_sub = q + static_cast<size_t>(m) * sub_dim_;
        const float* cb = codebooks_.data() + static_cast<size_t>(m) * k_pq_ * sub_dim_;
        float* t_row = out_table.data() + static_cast<size_t>(m) * k_pq_;
        for (uint32_t c = 0; c < k_pq_; ++c) {
            const float* c_ptr = cb + static_cast<size_t>(c) * sub_dim_;
            if (metric_ == MetricType::L2) {
                t_row[c] = l2_squared(q_sub, c_ptr, sub_dim_);
            } else {
                // Both COSINE (on normalized vectors) and DOT use negative inner product across subspaces
                t_row[c] = -dot_product(q_sub, c_ptr, sub_dim_);
            }
        }
    }
}

float ProductQuantizer::adc_distance(const uint8_t* codes, const float* adc_table) const noexcept {
    float sum = 0.0f;
    for (uint32_t m = 0; m < m_pq_; ++m) {
        sum += adc_table[static_cast<size_t>(m) * k_pq_ + codes[m]];
    }
    if (metric_ == MetricType::COSINE) {
        return 1.0f + sum; // since table stored -dot(q_norm, c_norm)
    }
    return sum;
}

uint64_t ProductQuantizer::codebook_ram_bytes() const {
    return codebooks_.capacity() * sizeof(float);
}

bool ProductQuantizer::save(const std::string& path) const {
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) return false;
    uint8_t m_byte = static_cast<uint8_t>(metric_);
    uint8_t tr = trained_ ? 1 : 0;
    os.write(reinterpret_cast<const char*>(&kPqMagic), sizeof(kPqMagic));
    os.write(reinterpret_cast<const char*>(&kPqVersion), sizeof(kPqVersion));
    os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    os.write(reinterpret_cast<const char*>(&m_byte), sizeof(m_byte));
    os.write(reinterpret_cast<const char*>(&m_pq_), sizeof(m_pq_));
    os.write(reinterpret_cast<const char*>(&k_pq_), sizeof(k_pq_));
    os.write(reinterpret_cast<const char*>(&sub_dim_), sizeof(sub_dim_));
    os.write(reinterpret_cast<const char*>(&tr), sizeof(tr));
    if (trained_) {
        os.write(reinterpret_cast<const char*>(codebooks_.data()),
                 static_cast<std::streamsize>(codebooks_.size() * sizeof(float)));
    }
    return static_cast<bool>(os);
}

bool ProductQuantizer::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) return false;
    uint32_t magic = 0, version = 0;
    uint8_t m_byte = 0, tr = 0;
    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    is.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!is || magic != kPqMagic || version != kPqVersion) return false;

    is.read(reinterpret_cast<char*>(&dim_), sizeof(dim_));
    is.read(reinterpret_cast<char*>(&m_byte), sizeof(m_byte));
    is.read(reinterpret_cast<char*>(&m_pq_), sizeof(m_pq_));
    is.read(reinterpret_cast<char*>(&k_pq_), sizeof(k_pq_));
    is.read(reinterpret_cast<char*>(&sub_dim_), sizeof(sub_dim_));
    is.read(reinterpret_cast<char*>(&tr), sizeof(tr));
    metric_ = static_cast<MetricType>(m_byte);
    trained_ = (tr != 0);
    if (trained_) {
        codebooks_.resize(static_cast<size_t>(m_pq_) * k_pq_ * sub_dim_);
        is.read(reinterpret_cast<char*>(codebooks_.data()),
                static_cast<std::streamsize>(codebooks_.size() * sizeof(float)));
    }
    return static_cast<bool>(is);
}

} // namespace vectordb
