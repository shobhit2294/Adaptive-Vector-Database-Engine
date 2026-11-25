#include "vectordb/wahi_controller.hpp"
#include <algorithm>
#include <cmath>

namespace vectordb {

std::vector<WahiDecision> WAHIController::reconcile(
    const DatabaseConfig& cfg,
    const std::vector<std::shared_ptr<Segment>>& sealed_segments,
    uint64_t memtable_ram_bytes,
    const ProductQuantizer* shared_pq,
    BoundedVectorCache& cache,
    MetricsRegistry& metrics,
    uint64_t current_tick) {
    std::vector<WahiDecision> transitions;
    if (sealed_segments.empty()) return transitions;

    const bool pq_ready = (shared_pq != nullptr && shared_pq->is_trained());

    struct PlanItem {
        std::shared_ptr<Segment> seg;
        SegmentTelemetry tel;
        uint64_t hot_bytes = 0;
        uint64_t warm_bytes = 0;
        uint64_t cold_bytes = 0;
        double raw_utility = 0.0;
        double sort_score = 0.0;
        SegmentTier target_tier = SegmentTier::HOT_RAM_HNSW;
    };

    std::vector<PlanItem> items;
    items.reserve(sealed_segments.size());
    uint64_t total_cold_bytes = 0;

    for (const auto& seg : sealed_segments) {
        if (!seg->is_sealed()) continue;
        if (cfg.wahi_mode == WahiPolicyMode::WAHI_ADAPTIVE) {
            seg->decay_telemetry(cfg.wahi_ewma_alpha);
        }
        PlanItem item;
        item.seg = seg;
        item.tel = seg->telemetry();
        item.hot_bytes = seg->estimated_hot_ram_bytes();
        item.warm_bytes = seg->estimated_warm_ram_bytes();
        item.cold_bytes = seg->estimated_cold_ram_bytes();
        total_cold_bytes += item.cold_bytes;

        double age = (current_tick >= item.tel.last_access_tick)
            ? static_cast<double>(current_tick - item.tel.last_access_tick)
            : 0.0;
        double recency = 1.0 / (1.0 + 0.25 * age);
        double demand = 3.0 * item.tel.useful_hits_ewma
                      + 0.35 * item.tel.scan_freq_ewma
                      + 1.5 * item.tel.write_freq_ewma;
        double marginal_mb = std::max(
            0.05,
            static_cast<double>(item.hot_bytes > item.cold_bytes ? (item.hot_bytes - item.cold_bytes) : 1)
                / (1024.0 * 1024.0));

        item.raw_utility = ((demand + 0.05) * (0.6 + 0.4 * recency)) / marginal_mb;
        item.sort_score = item.raw_utility;
        if (item.tel.tier == SegmentTier::HOT_RAM_HNSW) {
            item.sort_score *= cfg.wahi_hysteresis_margin;
        }
        seg->set_utility_score(item.raw_utility);
        items.push_back(std::move(item));
    }

    if (items.empty()) return transitions;

    if (cfg.wahi_mode == WahiPolicyMode::WAHI_ADAPTIVE && cfg.ram_budget_bytes > 0) {
        uint64_t capped_cache = std::min(cfg.disk_cache_bytes, cfg.ram_budget_bytes / 5);
        cache.set_capacity(capped_cache);
    } else {
        cache.set_capacity(cfg.disk_cache_bytes);
    }

    uint64_t cache_bytes = cache.capacity();
    uint64_t pq_cb_bytes = pq_ready ? shared_pq->codebook_ram_bytes() : 0;
    uint64_t overhead = memtable_ram_bytes + cache_bytes + pq_cb_bytes;
    uint64_t sealed_budget = (cfg.ram_budget_bytes > overhead)
        ? (cfg.ram_budget_bytes - overhead)
        : 0;

    if (cfg.wahi_mode == WahiPolicyMode::STATIC_ALL_RAM) {
        for (auto& it : items) it.target_tier = SegmentTier::HOT_RAM_HNSW;
    } else if (cfg.wahi_mode == WahiPolicyMode::STATIC_ALL_WARM_MMAP) {
        for (auto& it : items) it.target_tier = SegmentTier::WARM_MMAP_HNSW;
    } else if (cfg.wahi_mode == WahiPolicyMode::STATIC_ALL_DISK_PQ) {
        for (auto& it : items) {
            it.target_tier = pq_ready ? SegmentTier::COLD_DISK_PQ : SegmentTier::WARM_MMAP_HNSW;
        }
    } else if (cfg.wahi_mode == WahiPolicyMode::STATIC_LRU) {
        std::sort(items.begin(), items.end(), [](const PlanItem& a, const PlanItem& b) {
            if (a.tel.last_hit_tick != b.tel.last_hit_tick) {
                return a.tel.last_hit_tick > b.tel.last_hit_tick;
            }
            return a.tel.segment_id > b.tel.segment_id;
        });
        uint64_t extra_budget = (sealed_budget > total_cold_bytes) ? (sealed_budget - total_cold_bytes) : 0;
        uint64_t extra_used = 0;
        for (auto& it : items) {
            uint64_t hot_delta = (it.hot_bytes > it.cold_bytes) ? (it.hot_bytes - it.cold_bytes) : 0;
            if (extra_used + hot_delta <= extra_budget) {
                it.target_tier = SegmentTier::HOT_RAM_HNSW;
                extra_used += hot_delta;
            } else {
                it.target_tier = pq_ready ? SegmentTier::COLD_DISK_PQ : SegmentTier::WARM_MMAP_HNSW;
            }
        }
    } else {
        // WAHI_ADAPTIVE: 3-tier utility-per-byte knapsack with hysteresis & cooldown
        std::sort(items.begin(), items.end(), [](const PlanItem& a, const PlanItem& b) {
            if (a.sort_score != b.sort_score) {
                return a.sort_score > b.sort_score;
            }
            return a.tel.last_access_tick > b.tel.last_access_tick;
        });

        uint64_t extra_budget = (sealed_budget > total_cold_bytes) ? (sealed_budget - total_cold_bytes) : 0;
        uint64_t extra_used = 0;

        // Pass 1: Allocate HOT_RAM_HNSW to top-utility segments
        for (auto& it : items) {
            uint64_t hot_delta = (it.hot_bytes > it.cold_bytes) ? (it.hot_bytes - it.cold_bytes) : 0;
            if (extra_used + hot_delta <= extra_budget) {
                it.target_tier = SegmentTier::HOT_RAM_HNSW;
                extra_used += hot_delta;
            } else {
                it.target_tier = pq_ready ? SegmentTier::COLD_DISK_PQ : SegmentTier::WARM_MMAP_HNSW;
            }
        }

        // Pass 2: Upgrade remaining highest-utility segments to WARM_MMAP_HNSW if residual RAM allows
        for (auto& it : items) {
            if (it.target_tier == SegmentTier::HOT_RAM_HNSW) continue;
            uint64_t warm_delta = (it.warm_bytes > it.cold_bytes) ? (it.warm_bytes - it.cold_bytes) : 0;
            if (extra_used + warm_delta <= extra_budget) {
                it.target_tier = SegmentTier::WARM_MMAP_HNSW;
                extra_used += warm_delta;
            }
        }
    }

    // Execute demotions first to stay strictly below RAM ceiling during transitions
    for (auto& it : items) {
        SegmentTier cur = it.tel.tier;
        if (static_cast<uint8_t>(it.target_tier) > static_cast<uint8_t>(cur)) {
            if (it.seg->transition_to(it.target_tier, shared_pq, cache)) {
                it.seg->set_last_transition_tick(current_tick);
                metrics.record_demotion();
                transitions.push_back(WahiDecision{it.tel.segment_id, cur, it.target_tier, it.raw_utility});
            }
        }
    }

    // Execute promotions (respecting cooldown for adaptive mode unless cold->warm/hot is explicitly warranted)
    for (auto& it : items) {
        SegmentTier cur = it.tel.tier;
        if (static_cast<uint8_t>(it.target_tier) < static_cast<uint8_t>(cur)) {
            if (cfg.wahi_mode == WahiPolicyMode::WAHI_ADAPTIVE &&
                it.tel.last_transition_tick > 0 &&
                current_tick >= it.tel.last_transition_tick &&
                (current_tick - it.tel.last_transition_tick) < cfg.wahi_cooldown_ticks) {
                // Only hold back promotion if segment just transitioned; allow urgent promotion if utility is strongly elevated
                if (it.raw_utility < 2.0) continue;
            }
            if (it.seg->transition_to(it.target_tier, shared_pq, cache)) {
                it.seg->set_last_transition_tick(current_tick);
                metrics.record_promotion();
                transitions.push_back(WahiDecision{it.tel.segment_id, cur, it.target_tier, it.raw_utility});
            }
        }
    }

    return transitions;
}

} // namespace vectordb
