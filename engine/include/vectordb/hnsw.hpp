#pragma once

#include "vectordb/types.hpp"
#include <atomic>
#include <cstdint>
#include <functional>
#include <iosfwd>
#include <memory>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace vectordb {

class HnswIndex {
public:
    using FilterFn = std::function<bool(VectorId)>;

    HnswIndex(uint32_t dim,
              MetricType metric,
              uint32_t M = 16,
              uint32_t ef_construction = 128,
              uint32_t ef_search = 64,
              uint64_t seed = 42);

    void upsert(VectorId id, const std::vector<float>& vec);
    bool remove(VectorId id);
    bool contains(VectorId id) const;
    bool get(VectorId id, std::vector<float>& out) const;

    std::vector<SearchResult> search(
        const std::vector<float>& query,
        uint32_t k,
        uint32_t ef_override = 0,
        const FilterFn& filter = nullptr) const;

    std::vector<SearchResult> search_with_allowlist(
        const std::vector<float>& query,
        uint32_t k,
        const std::unordered_set<VectorId>& allowed_ids,
        uint32_t ef_override = 0) const;

    // Rebuilds index compacting away all tombstoned slots
    void compact();

    // Persistence (graph + vectors)
    bool save_to_file(const std::string& path) const;
    static std::unique_ptr<HnswIndex> load_from_file(const std::string& path);

    // Graph-only persistence + external mmap pointer support for WARM_MMAP_HNSW tier
    bool save_graph_only(const std::string& path) const;
    static std::unique_ptr<HnswIndex> load_graph_with_external_vectors(
        const std::string& graph_path,
        const float* external_vectors,
        uint32_t expected_slots);

    void set_external_vectors(const float* external_vectors, bool release_owned_vectors);
    void restore_owned_vectors(std::vector<float>&& owned_vectors);
    void sync_tombstones(const std::vector<uint8_t>& external_deleted);
    const std::vector<float>& owned_vectors() const { return owned_vectors_; }

    size_t size() const;
    size_t total_slots() const;
    size_t tombstone_count() const;
    double tombstone_ratio() const;
    uint32_t dimension() const { return dim_; }
    MetricType metric() const { return metric_; }
    uint32_t max_m() const { return M_; }
    uint32_t ef_search() const { return ef_search_; }
    void set_ef_search(uint32_t ef) { ef_search_ = ef; }

    uint64_t vector_ram_bytes() const;
    uint64_t graph_ram_bytes() const;
    uint64_t total_ram_bytes() const { return vector_ram_bytes() + graph_ram_bytes(); }

    // Access all live vector IDs and raw vector pointers (caller must keep index immutable/locked externally if concurrent)
    std::vector<std::pair<VectorId, std::vector<float>>> export_live_vectors() const;

private:
    struct DistSlot {
        float dist;
        uint32_t slot;
    };

    uint32_t dim_;
    MetricType metric_;
    uint32_t M_;
    uint32_t M_max0_;
    uint32_t ef_construction_;
    uint32_t ef_search_;
    double m_l_;
    uint64_t rng_state_;

    mutable std::shared_mutex mutex_;

    uint32_t entry_slot_ = UINT32_MAX;
    int max_level_ = -1;
    size_t live_count_ = 0;
    size_t tombstone_count_ = 0;

    std::vector<float> owned_vectors_;
    const float* external_vectors_ = nullptr;
    std::vector<float> norms_;
    std::vector<VectorId> slot_to_id_;
    std::vector<uint8_t> levels_;
    std::vector<uint8_t> deleted_;
    std::unordered_map<VectorId, uint32_t> id_to_slot_;

    // Layer 0 flat adjacency: each slot uses (M_max0_ + 1) uint32_t entries: [deg, n0, n1, ...]
    std::vector<uint32_t> layer0_links_;
    // Upper layers (layer >= 1): upper_links_[slot][layer - 1]
    std::vector<std::vector<std::vector<uint32_t>>> upper_links_;

    const float* vec_ptr(uint32_t slot) const noexcept {
        const float* base = external_vectors_ ? external_vectors_ : owned_vectors_.data();
        return base + static_cast<size_t>(slot) * dim_;
    }

    float dist_query_slot(const float* query, float q_norm, uint32_t slot) const noexcept;
    float dist_slots(uint32_t a, uint32_t b) const noexcept;

    int random_level();
    uint32_t greedy_search_layer(const float* query, float q_norm, uint32_t entry, int layer) const;

    std::vector<DistSlot> search_layer_for_insert(
        const float* query, float q_norm, uint32_t entry, uint32_t ef, int layer) const;

    void select_neighbors_heuristic(
        uint32_t target_slot,
        const float* target_vec,
        float target_norm,
        std::vector<DistSlot>& candidates,
        uint32_t m_limit,
        std::vector<uint32_t>& out) const;

    void connect_new_node(uint32_t new_slot, const std::vector<uint32_t>& neighbors, int layer);
    uint32_t* layer0_adj(uint32_t slot) noexcept {
        return layer0_links_.data() + static_cast<size_t>(slot) * (M_max0_ + 1);
    }
    const uint32_t* layer0_adj(uint32_t slot) const noexcept {
        return layer0_links_.data() + static_cast<size_t>(slot) * (M_max0_ + 1);
    }
};

} // namespace vectordb
