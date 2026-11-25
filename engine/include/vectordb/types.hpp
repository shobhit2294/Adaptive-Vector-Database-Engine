#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace vectordb {

using VectorId = uint64_t;
using SegmentId = uint32_t;
using SlotId = uint32_t;
using Lsn = uint64_t;

enum class MetricType : uint8_t {
    L2 = 0,
    COSINE = 1,
    DOT = 2
};

inline const char* to_string(MetricType m) {
    switch (m) {
        case MetricType::L2: return "l2";
        case MetricType::COSINE: return "cosine";
        case MetricType::DOT: return "dot";
    }
    return "unknown";
}

inline MetricType metric_from_string(const std::string& s) {
    if (s == "cosine" || s == "COSINE") return MetricType::COSINE;
    if (s == "dot" || s == "DOT" || s == "ip") return MetricType::DOT;
    return MetricType::L2;
}

enum class SegmentTier : uint8_t {
    HOT_MEMTABLE = 0,
    HOT_RAM_HNSW = 1,
    WARM_MMAP_HNSW = 2,
    COLD_DISK_PQ = 3
};

inline const char* to_string(SegmentTier t) {
    switch (t) {
        case SegmentTier::HOT_MEMTABLE: return "memtable";
        case SegmentTier::HOT_RAM_HNSW: return "hot_ram";
        case SegmentTier::WARM_MMAP_HNSW: return "warm_mmap";
        case SegmentTier::COLD_DISK_PQ: return "cold_disk_pq";
    }
    return "unknown";
}

enum class FsyncPolicy : uint8_t {
    ALWAYS = 0,
    BATCHED = 1,
    NONE = 2
};

enum class WahiPolicyMode : uint8_t {
    WAHI_ADAPTIVE = 0,
    STATIC_ALL_RAM = 1,
    STATIC_ALL_DISK_PQ = 2,
    STATIC_LRU = 3,
    STATIC_ALL_WARM_MMAP = 4
};

inline const char* to_string(WahiPolicyMode m) {
    switch (m) {
        case WahiPolicyMode::WAHI_ADAPTIVE: return "wahi_adaptive";
        case WahiPolicyMode::STATIC_ALL_RAM: return "static_all_ram";
        case WahiPolicyMode::STATIC_ALL_DISK_PQ: return "static_all_disk_pq";
        case WahiPolicyMode::STATIC_LRU: return "static_lru";
        case WahiPolicyMode::STATIC_ALL_WARM_MMAP: return "static_all_warm_mmap";
    }
    return "unknown";
}

inline WahiPolicyMode policy_from_string(const std::string& s) {
    if (s == "static_all_ram") return WahiPolicyMode::STATIC_ALL_RAM;
    if (s == "static_all_disk_pq") return WahiPolicyMode::STATIC_ALL_DISK_PQ;
    if (s == "static_lru") return WahiPolicyMode::STATIC_LRU;
    if (s == "static_all_warm_mmap") return WahiPolicyMode::STATIC_ALL_WARM_MMAP;
    return WahiPolicyMode::WAHI_ADAPTIVE;
}

struct SearchResult {
    VectorId id = 0;
    float distance = 0.0f;
    float score = 0.0f;
    SegmentTier source_tier = SegmentTier::HOT_RAM_HNSW;
};

struct SegmentTelemetry {
    SegmentId segment_id = 0;
    SegmentTier tier = SegmentTier::HOT_MEMTABLE;
    uint32_t vector_count = 0;
    uint32_t live_count = 0;
    uint32_t tombstone_count = 0;
    double scan_freq_ewma = 0.0;
    double useful_hits_ewma = 0.0;
    double write_freq_ewma = 0.0;
    double utility_score = 0.0;
    uint64_t last_access_tick = 0;
    uint64_t last_hit_tick = 0;
    uint64_t last_transition_tick = 0;
    uint64_t ram_bytes_vectors = 0;
    uint64_t ram_bytes_hnsw = 0;
    uint64_t ram_bytes_pq = 0;
    uint64_t disk_bytes = 0;
    uint64_t total_ram_bytes() const {
        return ram_bytes_vectors + ram_bytes_hnsw + ram_bytes_pq;
    }
};

struct DatabaseConfig {
    std::string data_dir = "./data";
    uint32_t dimension = 128;
    MetricType metric = MetricType::COSINE;

    // HNSW parameters
    uint32_t hnsw_m = 16;
    uint32_t hnsw_ef_construction = 128;
    uint32_t hnsw_ef_search = 64;

    // Segment & Storage parameters
    uint32_t segment_capacity = 4096;
    FsyncPolicy wal_fsync = FsyncPolicy::BATCHED;
    uint64_t wal_checkpoint_bytes = 32ULL * 1024ULL * 1024ULL;
    double compaction_tombstone_ratio = 0.25;

    // PQ parameters
    uint32_t pq_subspaces = 16;   // M_pq (must divide dimension)
    uint32_t pq_centroids = 256;  // K_pq (<= 256 for uint8 codes)
    uint32_t pq_train_iters = 15;
    uint32_t cold_rerank_factor = 4; // Rerank top-(factor * k) candidates from disk

    // Disk LRU page/vector cache
    uint64_t disk_cache_bytes = 16ULL * 1024ULL * 1024ULL;

    // WAHI Adaptive Controller parameters
    WahiPolicyMode wahi_mode = WahiPolicyMode::WAHI_ADAPTIVE;
    uint64_t ram_budget_bytes = 64ULL * 1024ULL * 1024ULL;
    uint32_t wahi_reconcile_interval_queries = 200;
    uint32_t wahi_cooldown_ticks = 2;
    double wahi_ewma_alpha = 0.25;
    double wahi_hysteresis_margin = 1.15;
    bool auto_background_adapt = true;
};

struct SearchOptions {
    uint32_t k = 10;
    uint32_t ef_search = 0;          // 0 = use database default
    uint32_t cold_rerank_factor = 0; // 0 = use database default
    bool bypass_cold_rerank = false;
};

struct SearchDiagnostics {
    double latency_ms = 0.0;
    uint32_t segments_scanned = 0;
    uint32_t hot_segments_scanned = 0;
    uint32_t warm_segments_scanned = 0;
    uint32_t cold_segments_scanned = 0;
    uint64_t disk_reads = 0;
    uint64_t disk_bytes_read = 0;
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint32_t candidates_scored = 0;
};

} // namespace vectordb
