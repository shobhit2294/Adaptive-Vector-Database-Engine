#pragma once

#include "vectordb/types.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace vectordb {

struct EngineSnapshot {
    uint64_t live_vectors = 0;
    uint64_t tombstones = 0;
    uint32_t segments_memtable = 0;
    uint32_t segments_hot_ram = 0;
    uint32_t segments_warm_mmap = 0;
    uint32_t segments_cold_disk_pq = 0;
    uint64_t ram_bytes_vectors = 0;
    uint64_t ram_bytes_hnsw = 0;
    uint64_t ram_bytes_pq = 0;
    uint64_t ram_bytes_cache = 0;
    uint64_t ram_bytes_total = 0;
    uint64_t ram_budget_bytes = 0;
    uint64_t disk_bytes_total = 0;
    uint64_t wal_bytes = 0;
    uint64_t wal_records = 0;
    WahiPolicyMode wahi_mode = WahiPolicyMode::WAHI_ADAPTIVE;
};

struct LatencySummary {
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double avg_ms = 0.0;
    uint64_t count = 0;
};

class MetricsRegistry {
public:
    static constexpr size_t kNumBuckets = 10;
    static constexpr std::array<double, kNumBuckets - 1> kBucketBoundsMs = {
        0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0
    };

    MetricsRegistry();

    void record_query(double latency_ms, bool filtered, const SearchDiagnostics& diag);
    void record_insert();
    void record_update();
    void record_delete();
    void record_promotion();
    void record_demotion();
    void record_compaction();
    void record_checkpoint();

    LatencySummary latency_summary() const;
    void reset();

    std::string export_prometheus(const EngineSnapshot& snap) const;

    std::atomic<uint64_t> queries_total{0};
    std::atomic<uint64_t> filtered_queries_total{0};
    std::atomic<uint64_t> inserts_total{0};
    std::atomic<uint64_t> updates_total{0};
    std::atomic<uint64_t> deletes_total{0};
    std::atomic<uint64_t> disk_reads_total{0};
    std::atomic<uint64_t> disk_bytes_read_total{0};
    std::atomic<uint64_t> cache_hits_total{0};
    std::atomic<uint64_t> cache_misses_total{0};
    std::atomic<uint64_t> wahi_promotions_total{0};
    std::atomic<uint64_t> wahi_demotions_total{0};
    std::atomic<uint64_t> compactions_total{0};
    std::atomic<uint64_t> wal_checkpoints_total{0};

private:
    static constexpr size_t kRingSize = 2048;
    mutable std::mutex lat_mutex_;
    std::vector<double> ring_;
    size_t ring_pos_ = 0;
    bool ring_full_ = false;
    double latency_sum_ms_ = 0.0;
    std::array<uint64_t, kNumBuckets> buckets_{};
};

} // namespace vectordb
