#pragma once

#include "vectordb/metrics.hpp"
#include "vectordb/mmap_storage.hpp"
#include "vectordb/pq.hpp"
#include "vectordb/segment.hpp"
#include "vectordb/types.hpp"
#include <cstdint>
#include <memory>
#include <vector>

namespace vectordb {

struct WahiDecision {
    SegmentId segment_id = 0;
    SegmentTier from_tier = SegmentTier::HOT_RAM_HNSW;
    SegmentTier to_tier = SegmentTier::HOT_RAM_HNSW;
    double utility_score = 0.0;
};

class WAHIController {
public:
    WAHIController() = default;

    // Runs one adaptation pass across sealed segments under cfg.ram_budget_bytes.
    // Returns list of tier transitions executed.
    std::vector<WahiDecision> reconcile(
        const DatabaseConfig& cfg,
        const std::vector<std::shared_ptr<Segment>>& sealed_segments,
        uint64_t memtable_ram_bytes,
        const ProductQuantizer* shared_pq,
        BoundedVectorCache& cache,
        MetricsRegistry& metrics,
        uint64_t current_tick);
};

} // namespace vectordb
