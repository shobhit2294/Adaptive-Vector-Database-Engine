#include "vectordb/brute_force.hpp"
#include "vectordb/distance.hpp"
#include <algorithm>
#include <mutex>
#include <queue>
#include <stdexcept>

namespace vectordb {

BruteForceIndex::BruteForceIndex(uint32_t dim, MetricType metric)
    : dim_(dim), metric_(metric) {
    if (dim_ == 0) {
        throw std::invalid_argument("Dimension must be > 0");
    }
}

void BruteForceIndex::upsert(VectorId id, const std::vector<float>& vec) {
    if (vec.size() != dim_) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    float norm = vector_norm(vec.data(), dim_);
    std::unique_lock lock(mutex_);
    auto it = id_to_slot_.find(id);
    if (it != id_to_slot_.end()) {
        uint32_t slot = it->second;
        std::copy(vec.begin(), vec.end(), data_.begin() + static_cast<size_t>(slot) * dim_);
        norms_[slot] = norm;
        return;
    }

    uint32_t slot = 0;
    if (!free_slots_.empty()) {
        slot = free_slots_.back();
        free_slots_.pop_back();
        std::copy(vec.begin(), vec.end(), data_.begin() + static_cast<size_t>(slot) * dim_);
        norms_[slot] = norm;
        slot_to_id_[slot] = id;
        alive_[slot] = 1;
    } else {
        slot = static_cast<uint32_t>(slot_to_id_.size());
        data_.insert(data_.end(), vec.begin(), vec.end());
        norms_.push_back(norm);
        slot_to_id_.push_back(id);
        alive_.push_back(1);
    }
    id_to_slot_[id] = slot;
    ++live_count_;
}

bool BruteForceIndex::remove(VectorId id) {
    std::unique_lock lock(mutex_);
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) {
        return false;
    }
    uint32_t slot = it->second;
    alive_[slot] = 0;
    free_slots_.push_back(slot);
    id_to_slot_.erase(it);
    --live_count_;
    return true;
}

bool BruteForceIndex::get(VectorId id, std::vector<float>& out) const {
    std::shared_lock lock(mutex_);
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) {
        return false;
    }
    uint32_t slot = it->second;
    const float* ptr = data_.data() + static_cast<size_t>(slot) * dim_;
    out.assign(ptr, ptr + dim_);
    return true;
}

bool BruteForceIndex::contains(VectorId id) const {
    std::shared_lock lock(mutex_);
    return id_to_slot_.find(id) != id_to_slot_.end();
}

std::vector<SearchResult> BruteForceIndex::search(
    const std::vector<float>& query,
    uint32_t k,
    const FilterFn& filter) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("Query dimension mismatch");
    }
    if (k == 0) {
        return {};
    }

    float q_norm = vector_norm(query.data(), dim_);
    std::shared_lock lock(mutex_);

    struct Candidate {
        float dist;
        VectorId id;
        bool operator<(const Candidate& other) const {
            if (dist != other.dist) return dist < other.dist; // max-heap by dist
            return id < other.id;
        }
    };

    std::priority_queue<Candidate> top_k;
    const size_t total_slots = slot_to_id_.size();
    for (size_t slot = 0; slot < total_slots; ++slot) {
        if (!alive_[slot]) continue;
        VectorId id = slot_to_id_[slot];
        if (filter && !filter(id)) continue;

        const float* vec = data_.data() + slot * dim_;
        float dist = 0.0f;
        if (metric_ == MetricType::COSINE) {
            dist = cosine_distance_with_norms(query.data(), q_norm, vec, norms_[slot], dim_);
        } else {
            dist = compute_distance(query.data(), vec, dim_, metric_);
        }

        if (top_k.size() < k) {
            top_k.push({dist, id});
        } else if (dist < top_k.top().dist || (dist == top_k.top().dist && id < top_k.top().id)) {
            top_k.pop();
            top_k.push({dist, id});
        }
    }

    std::vector<SearchResult> results;
    results.reserve(top_k.size());
    while (!top_k.empty()) {
        auto c = top_k.top();
        top_k.pop();
        results.push_back(SearchResult{c.id, c.dist, distance_to_score(c.dist, metric_), SegmentTier::HOT_RAM_HNSW});
    }
    std::reverse(results.begin(), results.end());
    return results;
}

std::vector<SearchResult> BruteForceIndex::search_with_allowlist(
    const std::vector<float>& query,
    uint32_t k,
    const std::unordered_set<VectorId>& allowed_ids) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("Query dimension mismatch");
    }
    if (k == 0 || allowed_ids.empty()) {
        return {};
    }

    float q_norm = vector_norm(query.data(), dim_);
    std::shared_lock lock(mutex_);

    struct Candidate {
        float dist;
        VectorId id;
        bool operator<(const Candidate& other) const {
            if (dist != other.dist) return dist < other.dist;
            return id < other.id;
        }
    };
    std::priority_queue<Candidate> top_k;

    if (allowed_ids.size() * 4 < live_count_) {
        for (VectorId id : allowed_ids) {
            auto it = id_to_slot_.find(id);
            if (it == id_to_slot_.end()) continue;
            uint32_t slot = it->second;
            const float* vec = data_.data() + static_cast<size_t>(slot) * dim_;
            float dist = (metric_ == MetricType::COSINE)
                ? cosine_distance_with_norms(query.data(), q_norm, vec, norms_[slot], dim_)
                : compute_distance(query.data(), vec, dim_, metric_);
            if (top_k.size() < k) {
                top_k.push({dist, id});
            } else if (dist < top_k.top().dist) {
                top_k.pop();
                top_k.push({dist, id});
            }
        }
    } else {
        for (size_t slot = 0; slot < slot_to_id_.size(); ++slot) {
            if (!alive_[slot]) continue;
            VectorId id = slot_to_id_[slot];
            if (allowed_ids.find(id) == allowed_ids.end()) continue;
            const float* vec = data_.data() + slot * dim_;
            float dist = (metric_ == MetricType::COSINE)
                ? cosine_distance_with_norms(query.data(), q_norm, vec, norms_[slot], dim_)
                : compute_distance(query.data(), vec, dim_, metric_);
            if (top_k.size() < k) {
                top_k.push({dist, id});
            } else if (dist < top_k.top().dist) {
                top_k.pop();
                top_k.push({dist, id});
            }
        }
    }

    std::vector<SearchResult> results;
    results.reserve(top_k.size());
    while (!top_k.empty()) {
        auto c = top_k.top();
        top_k.pop();
        results.push_back(SearchResult{c.id, c.dist, distance_to_score(c.dist, metric_), SegmentTier::HOT_RAM_HNSW});
    }
    std::reverse(results.begin(), results.end());
    return results;
}

size_t BruteForceIndex::size() const {
    std::shared_lock lock(mutex_);
    return live_count_;
}

uint64_t BruteForceIndex::ram_bytes() const {
    std::shared_lock lock(mutex_);
    return data_.capacity() * sizeof(float)
         + norms_.capacity() * sizeof(float)
         + slot_to_id_.capacity() * sizeof(VectorId)
         + alive_.capacity() * sizeof(uint8_t)
         + id_to_slot_.size() * (sizeof(VectorId) + sizeof(uint32_t) + 24);
}

} // namespace vectordb
