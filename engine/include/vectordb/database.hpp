#pragma once

#include "vectordb/metrics.hpp"
#include "vectordb/mmap_storage.hpp"
#include "vectordb/pq.hpp"
#include "vectordb/segment.hpp"
#include "vectordb/types.hpp"
#include "vectordb/wahi_controller.hpp"
#include "vectordb/wal.hpp"
#include <atomic>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vectordb {

class VectorDatabase {
public:
    explicit VectorDatabase(DatabaseConfig config);
    ~VectorDatabase();

    VectorDatabase(const VectorDatabase&) = delete;
    VectorDatabase& operator=(const VectorDatabase&) = delete;

    bool open();
    void close();

    // CRUD operations (WAL-logged + crash-safe)
    void upsert(VectorId id, const std::vector<float>& vec);
    void upsert_batch(const std::vector<std::pair<VectorId, std::vector<float>>>& items);
    bool remove(VectorId id);
    bool get(VectorId id, std::vector<float>& out) const;
    bool contains(VectorId id) const;

    // Similarity search (pure or allowlist-filtered)
    std::vector<SearchResult> search(
        const std::vector<float>& query,
        const SearchOptions& opts = SearchOptions{},
        SearchDiagnostics* out_diag = nullptr,
        const std::unordered_set<VectorId>* allowed_ids = nullptr);

    // Explicit maintenance & adaptive control
    bool flush_and_seal();
    size_t compact_tombstones();
    std::vector<WahiDecision> trigger_wahi_reconcile();
    void set_wahi_policy(WahiPolicyMode mode, uint64_t ram_budget_bytes = 0);
    void set_ram_budget(uint64_t ram_budget_bytes);

    // Inspection & observability
    EngineSnapshot snapshot() const;
    std::vector<SegmentTelemetry> segment_telemetries() const;
    std::string prometheus_metrics() const;
    const DatabaseConfig& config() const { return config_; }
    MetricsRegistry& metrics() { return metrics_; }
    const MetricsRegistry& metrics() const { return metrics_; }

private:
    DatabaseConfig config_;
    bool is_open_ = false;

    mutable std::mutex write_mutex_;
    mutable std::mutex adapt_mutex_;
    mutable std::shared_mutex db_mutex_;

    WriteAheadLog wal_;
    std::unique_ptr<ProductQuantizer> shared_pq_;
    BoundedVectorCache disk_cache_;
    WAHIController wahi_;
    mutable MetricsRegistry metrics_;

    std::shared_ptr<Segment> memtable_;
    std::vector<std::shared_ptr<Segment>> sealed_segments_;
    std::unordered_map<VectorId, std::shared_ptr<Segment>> id_to_segment_;

    SegmentId next_segment_id_ = 1;
    Lsn last_checkpoint_lsn_ = 0;
    std::atomic<uint64_t> access_tick_{0};
    std::atomic<uint64_t> query_count_since_adapt_{0};

    std::string segments_dir() const { return config_.data_dir + "/segments"; }
    std::string wal_path() const { return config_.data_dir + "/wal.log"; }
    std::string manifest_path() const { return config_.data_dir + "/db_manifest.json"; }
    std::string pq_codebook_path() const { return config_.data_dir + "/pq_codebook.bin"; }

    void apply_upsert_in_memory(VectorId id, const std::vector<float>& vec, bool during_recovery);
    bool apply_delete_in_memory(VectorId id);
    bool seal_memtable_locked(Lsn current_lsn);
    void ensure_pq_trained_locked();
    bool save_manifest_locked() const;
    bool load_manifest_and_segments();
};

} // namespace vectordb
