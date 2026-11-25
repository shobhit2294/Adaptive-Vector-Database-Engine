#pragma once

#include "vectordb/hnsw.hpp"
#include "vectordb/mmap_storage.hpp"
#include "vectordb/pq.hpp"
#include "vectordb/types.hpp"
#include <cstdint>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vectordb {

class Segment {
public:
    Segment(SegmentId id,
            const std::string& seg_dir,
            uint32_t dim,
            MetricType metric,
            uint32_t hnsw_m,
            uint32_t hnsw_ef_construction,
            uint32_t hnsw_ef_search,
            uint32_t pq_subspaces);

    // Load an existing persisted segment from seg_dir.
    static std::shared_ptr<Segment> load_from_disk(
        SegmentId id,
        const std::string& seg_dir,
        uint32_t hnsw_m,
        uint32_t hnsw_ef_construction,
        uint32_t hnsw_ef_search,
        uint32_t pq_subspaces,
        SegmentTier initial_tier = SegmentTier::HOT_RAM_HNSW);

    // Mutable memtable operations
    void insert_memtable(VectorId id, const std::vector<float>& vec);
    bool remove(VectorId id);
    bool contains(VectorId id) const;
    bool get(VectorId id, std::vector<float>& out) const;

    // Seals memtable to immutable persistent disk segment (HOT_RAM_HNSW).
    bool seal(const ProductQuantizer* shared_pq = nullptr);

    // Tier migration (HOT_RAM_HNSW <-> WARM_MMAP_HNSW <-> COLD_DISK_PQ)
    bool transition_to(SegmentTier target,
                       const ProductQuantizer* shared_pq,
                       BoundedVectorCache& cache);

    // Search within segment
    std::vector<SearchResult> search(
        const std::vector<float>& query,
        const SearchOptions& opts,
        const ProductQuantizer* shared_pq,
        const float* shared_adc_table,
        BoundedVectorCache& cache,
        SearchDiagnostics& diag,
        const std::unordered_set<VectorId>* allowed_ids = nullptr) const;

    // Export all live (non-tombstoned) vectors for compaction
    std::vector<std::pair<VectorId, std::vector<float>>> export_live_vectors() const;

    // Telemetry updates (called by database / WAHI controller)
    void record_scan(uint64_t tick);
    void record_useful_hit(uint64_t tick, double weight = 1.0);
    void record_write(uint64_t tick);
    void decay_telemetry(double alpha);
    void set_utility_score(double u) { utility_score_ = u; }
    void set_last_transition_tick(uint64_t t) { last_transition_tick_ = t; }

    SegmentId id() const { return id_; }
    SegmentTier tier() const;
    uint32_t total_slots() const;
    uint32_t live_count() const;
    uint32_t tombstone_count() const;
    double tombstone_ratio() const;
    bool is_sealed() const;

    uint64_t estimated_hot_ram_bytes() const;
    uint64_t estimated_warm_ram_bytes() const;
    uint64_t estimated_cold_ram_bytes() const;
    SegmentTelemetry telemetry() const;

    void remove_disk_files() const;

private:
    SegmentId id_;
    std::string seg_dir_;
    uint32_t dim_;
    MetricType metric_;
    uint32_t hnsw_m_;
    uint32_t hnsw_ef_construction_;
    uint32_t hnsw_ef_search_;
    uint32_t pq_subspaces_;

    mutable std::shared_mutex mutex_;
    SegmentTier tier_ = SegmentTier::HOT_MEMTABLE;
    bool sealed_ = false;

    uint32_t total_slots_ = 0;
    uint32_t live_count_ = 0;
    uint32_t tombstone_count_ = 0;

    std::vector<VectorId> slot_to_id_;
    std::vector<uint8_t> deleted_;
    std::unordered_map<VectorId, uint32_t> id_to_slot_;

    // Tier-specific storage
    std::unique_ptr<HnswIndex> hnsw_;
    MmapVectorFile mmap_vecs_;
    DiskVectorReader disk_reader_;
    std::vector<uint8_t> pq_codes_;
    uint64_t cached_graph_bytes_ = 0;
    uint64_t disk_bytes_ = 0;

    // WAHI EWMA telemetry
    double scan_freq_ewma_ = 0.0;
    double useful_hits_ewma_ = 0.0;
    double write_freq_ewma_ = 0.0;
    double utility_score_ = 0.0;
    uint64_t last_access_tick_ = 0;
    uint64_t last_hit_tick_ = 0;
    uint64_t last_transition_tick_ = 0;

    std::string vec_path() const;
    std::string hnsw_path() const;
    std::string meta_path() const;
    std::string pq_path() const;

    bool save_meta_unlocked() const;
    bool load_meta_unlocked();
    bool ensure_pq_codes_unlocked(const ProductQuantizer* shared_pq);
    bool load_owned_vectors_unlocked(std::vector<float>& out_vecs) const;
};

} // namespace vectordb
