#include "vectordb/metrics.hpp"
#include <algorithm>
#include <sstream>

namespace vectordb {

MetricsRegistry::MetricsRegistry() : ring_(kRingSize, 0.0) {}

void MetricsRegistry::record_query(double latency_ms, bool filtered, const SearchDiagnostics& diag) {
    queries_total.fetch_add(1, std::memory_order_relaxed);
    if (filtered) {
        filtered_queries_total.fetch_add(1, std::memory_order_relaxed);
    }
    disk_reads_total.fetch_add(diag.disk_reads, std::memory_order_relaxed);
    disk_bytes_read_total.fetch_add(diag.disk_bytes_read, std::memory_order_relaxed);
    cache_hits_total.fetch_add(diag.cache_hits, std::memory_order_relaxed);
    cache_misses_total.fetch_add(diag.cache_misses, std::memory_order_relaxed);

    std::lock_guard lock(lat_mutex_);
    ring_[ring_pos_] = latency_ms;
    ring_pos_ = (ring_pos_ + 1) % kRingSize;
    if (ring_pos_ == 0) ring_full_ = true;
    latency_sum_ms_ += latency_ms;

    for (size_t i = 0; i < kBucketBoundsMs.size(); ++i) {
        if (latency_ms <= kBucketBoundsMs[i]) {
            ++buckets_[i];
        }
    }
    ++buckets_[kNumBuckets - 1]; // +Inf cumulative bucket
}

void MetricsRegistry::record_insert() { inserts_total.fetch_add(1, std::memory_order_relaxed); }
void MetricsRegistry::record_update() { updates_total.fetch_add(1, std::memory_order_relaxed); }
void MetricsRegistry::record_delete() { deletes_total.fetch_add(1, std::memory_order_relaxed); }
void MetricsRegistry::record_promotion() { wahi_promotions_total.fetch_add(1, std::memory_order_relaxed); }
void MetricsRegistry::record_demotion() { wahi_demotions_total.fetch_add(1, std::memory_order_relaxed); }
void MetricsRegistry::record_compaction() { compactions_total.fetch_add(1, std::memory_order_relaxed); }
void MetricsRegistry::record_checkpoint() { wal_checkpoints_total.fetch_add(1, std::memory_order_relaxed); }

LatencySummary MetricsRegistry::latency_summary() const {
    std::lock_guard lock(lat_mutex_);
    size_t n = ring_full_ ? kRingSize : ring_pos_;
    LatencySummary sum;
    sum.count = queries_total.load(std::memory_order_relaxed);
    if (n == 0) return sum;

    std::vector<double> sorted(ring_.begin(), ring_.begin() + static_cast<ptrdiff_t>(n));
    std::sort(sorted.begin(), sorted.end());
    auto pct = [&](double p) -> double {
        size_t idx = static_cast<size_t>(p * static_cast<double>(n - 1));
        return sorted[std::min(idx, n - 1)];
    };
    sum.p50_ms = pct(0.50);
    sum.p95_ms = pct(0.95);
    sum.p99_ms = pct(0.99);
    double total = 0.0;
    for (double v : sorted) total += v;
    sum.avg_ms = total / static_cast<double>(n);
    return sum;
}

void MetricsRegistry::reset() {
    queries_total = 0;
    filtered_queries_total = 0;
    inserts_total = 0;
    updates_total = 0;
    deletes_total = 0;
    disk_reads_total = 0;
    disk_bytes_read_total = 0;
    cache_hits_total = 0;
    cache_misses_total = 0;
    wahi_promotions_total = 0;
    wahi_demotions_total = 0;
    compactions_total = 0;
    wal_checkpoints_total = 0;

    std::lock_guard lock(lat_mutex_);
    std::fill(ring_.begin(), ring_.end(), 0.0);
    ring_pos_ = 0;
    ring_full_ = false;
    latency_sum_ms_ = 0.0;
    buckets_.fill(0);
}

std::string MetricsRegistry::export_prometheus(const EngineSnapshot& snap) const {
    auto lat = latency_summary();
    std::ostringstream ss;

    ss << "# HELP vectordb_vectors_total Total live vectors in the database\n"
       << "# TYPE vectordb_vectors_total gauge\n"
       << "vectordb_vectors_total " << snap.live_vectors << "\n";

    ss << "# HELP vectordb_tombstones_total Total tombstoned slots awaiting compaction\n"
       << "# TYPE vectordb_tombstones_total gauge\n"
       << "vectordb_tombstones_total " << snap.tombstones << "\n";

    ss << "# HELP vectordb_segments_count Number of segments per storage tier\n"
       << "# TYPE vectordb_segments_count gauge\n"
       << "vectordb_segments_count{tier=\"memtable\"} " << snap.segments_memtable << "\n"
       << "vectordb_segments_count{tier=\"hot_ram\"} " << snap.segments_hot_ram << "\n"
       << "vectordb_segments_count{tier=\"warm_mmap\"} " << snap.segments_warm_mmap << "\n"
       << "vectordb_segments_count{tier=\"cold_disk_pq\"} " << snap.segments_cold_disk_pq << "\n";

    ss << "# HELP vectordb_ram_bytes Tracked RAM consumption in bytes by component\n"
       << "# TYPE vectordb_ram_bytes gauge\n"
       << "vectordb_ram_bytes{component=\"vectors\"} " << snap.ram_bytes_vectors << "\n"
       << "vectordb_ram_bytes{component=\"hnsw\"} " << snap.ram_bytes_hnsw << "\n"
       << "vectordb_ram_bytes{component=\"pq\"} " << snap.ram_bytes_pq << "\n"
       << "vectordb_ram_bytes{component=\"cache\"} " << snap.ram_bytes_cache << "\n"
       << "vectordb_ram_bytes{component=\"total\"} " << snap.ram_bytes_total << "\n";

    ss << "# HELP vectordb_ram_budget_bytes Configured WAHI RAM budget in bytes\n"
       << "# TYPE vectordb_ram_budget_bytes gauge\n"
       << "vectordb_ram_budget_bytes " << snap.ram_budget_bytes << "\n";

    ss << "# HELP vectordb_disk_bytes_total Persistent disk footprint in bytes\n"
       << "# TYPE vectordb_disk_bytes_total gauge\n"
       << "vectordb_disk_bytes_total " << snap.disk_bytes_total << "\n";

    ss << "# HELP vectordb_queries_total Total search queries executed\n"
       << "# TYPE vectordb_queries_total counter\n"
       << "vectordb_queries_total{mode=\"pure\"} "
       << (queries_total.load() - filtered_queries_total.load()) << "\n"
       << "vectordb_queries_total{mode=\"filtered\"} "
       << filtered_queries_total.load() << "\n";

    ss << "# HELP vectordb_mutations_total Total write operations by type\n"
       << "# TYPE vectordb_mutations_total counter\n"
       << "vectordb_mutations_total{op=\"insert\"} " << inserts_total.load() << "\n"
       << "vectordb_mutations_total{op=\"update\"} " << updates_total.load() << "\n"
       << "vectordb_mutations_total{op=\"delete\"} " << deletes_total.load() << "\n";

    ss << "# HELP vectordb_disk_reads_total Cold segment disk reads\n"
       << "# TYPE vectordb_disk_reads_total counter\n"
       << "vectordb_disk_reads_total " << disk_reads_total.load() << "\n";

    ss << "# HELP vectordb_disk_bytes_read_total Total bytes read from disk for cold reranking\n"
       << "# TYPE vectordb_disk_bytes_read_total counter\n"
       << "vectordb_disk_bytes_read_total " << disk_bytes_read_total.load() << "\n";

    ss << "# HELP vectordb_cache_hits_total LRU vector cache hits\n"
       << "# TYPE vectordb_cache_hits_total counter\n"
       << "vectordb_cache_hits_total " << cache_hits_total.load() << "\n";

    ss << "# HELP vectordb_cache_misses_total LRU vector cache misses\n"
       << "# TYPE vectordb_cache_misses_total counter\n"
       << "vectordb_cache_misses_total " << cache_misses_total.load() << "\n";

    ss << "# HELP vectordb_wahi_transitions_total WAHI segment tier transitions\n"
       << "# TYPE vectordb_wahi_transitions_total counter\n"
       << "vectordb_wahi_transitions_total{direction=\"promote\"} " << wahi_promotions_total.load() << "\n"
       << "vectordb_wahi_transitions_total{direction=\"demote\"} " << wahi_demotions_total.load() << "\n";

    ss << "# HELP vectordb_compactions_total Segment compactions performed\n"
       << "# TYPE vectordb_compactions_total counter\n"
       << "vectordb_compactions_total " << compactions_total.load() << "\n";

    ss << "# HELP vectordb_search_latency_quantile_ms Search query latency summary quantiles\n"
       << "# TYPE vectordb_search_latency_quantile_ms gauge\n"
       << "vectordb_search_latency_quantile_ms{quantile=\"0.5\"} " << lat.p50_ms << "\n"
       << "vectordb_search_latency_quantile_ms{quantile=\"0.95\"} " << lat.p95_ms << "\n"
       << "vectordb_search_latency_quantile_ms{quantile=\"0.99\"} " << lat.p99_ms << "\n";

    {
        std::lock_guard lock(lat_mutex_);
        ss << "# HELP vectordb_search_latency_ms Search query latency histogram in milliseconds\n"
           << "# TYPE vectordb_search_latency_ms histogram\n";
        for (size_t i = 0; i < kBucketBoundsMs.size(); ++i) {
            ss << "vectordb_search_latency_ms_bucket{le=\"" << kBucketBoundsMs[i] << "\"} " << buckets_[i] << "\n";
        }
        ss << "vectordb_search_latency_ms_bucket{le=\"+Inf\"} " << buckets_[kNumBuckets - 1] << "\n"
           << "vectordb_search_latency_ms_sum " << latency_sum_ms_ << "\n"
           << "vectordb_search_latency_ms_count " << buckets_[kNumBuckets - 1] << "\n";
    }

    return ss.str();
}

} // namespace vectordb
