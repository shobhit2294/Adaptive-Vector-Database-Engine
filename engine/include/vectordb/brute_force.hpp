#pragma once

#include "vectordb/types.hpp"
#include <functional>
#include <shared_mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vectordb {

class BruteForceIndex {
public:
    using FilterFn = std::function<bool(VectorId)>;

    BruteForceIndex(uint32_t dim, MetricType metric);

    void upsert(VectorId id, const std::vector<float>& vec);
    bool remove(VectorId id);
    bool get(VectorId id, std::vector<float>& out) const;
    bool contains(VectorId id) const;

    std::vector<SearchResult> search(
        const std::vector<float>& query,
        uint32_t k,
        const FilterFn& filter = nullptr) const;

    std::vector<SearchResult> search_with_allowlist(
        const std::vector<float>& query,
        uint32_t k,
        const std::unordered_set<VectorId>& allowed_ids) const;

    size_t size() const;
    uint32_t dimension() const { return dim_; }
    MetricType metric() const { return metric_; }
    uint64_t ram_bytes() const;

private:
    uint32_t dim_;
    MetricType metric_;
    mutable std::shared_mutex mutex_;

    // Contiguous vector buffer + free slot recycling for cache-friendly scans
    std::vector<float> data_;
    std::vector<float> norms_;
    std::vector<VectorId> slot_to_id_;
    std::vector<uint8_t> alive_;
    std::vector<uint32_t> free_slots_;
    std::unordered_map<VectorId, uint32_t> id_to_slot_;
    size_t live_count_ = 0;
};

} // namespace vectordb
