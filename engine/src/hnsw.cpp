#include "vectordb/hnsw.hpp"
#include "vectordb/distance.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <mutex>
#include <queue>
#include <stdexcept>

namespace vectordb {

namespace {

constexpr uint32_t kHnswMagic = 0x484E5357; // "HNSW"
constexpr uint32_t kHnswVersion = 1;

struct VisitedTracker {
    std::vector<uint32_t> marks;
    uint32_t epoch = 1;

    explicit VisitedTracker(size_t n = 0) : marks(n, 0) {}

    void reset(size_t n) {
        if (marks.size() < n) {
            marks.assign(n, 0);
            epoch = 1;
            return;
        }
        ++epoch;
        if (epoch == 0) {
            std::fill(marks.begin(), marks.end(), 0);
            epoch = 1;
        }
    }

    bool visit(uint32_t slot) {
        if (marks[slot] == epoch) return false;
        marks[slot] = epoch;
        return true;
    }
};

} // namespace

HnswIndex::HnswIndex(uint32_t dim,
                     MetricType metric,
                     uint32_t M,
                     uint32_t ef_construction,
                     uint32_t ef_search,
                     uint64_t seed)
    : dim_(dim),
      metric_(metric),
      M_(std::max(4u, M)),
      M_max0_(std::max(8u, M * 2u)),
      ef_construction_(std::max(ef_construction, M_)),
      ef_search_(std::max(1u, ef_search)),
      m_l_(1.0 / std::log(static_cast<double>(M_))),
      rng_state_(seed == 0 ? 0x9e3779b97f4a7c15ULL : seed) {
    if (dim_ == 0) {
        throw std::invalid_argument("HNSW dimension must be > 0");
    }
}

int HnswIndex::random_level() {
    // xorshift64*
    uint64_t x = rng_state_;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    rng_state_ = x;
    double u = static_cast<double>((x * 0x2545F4914F6CDD1DULL) >> 11) * (1.0 / 9007199254740992.0);
    if (u < 1e-12) u = 1e-12;
    int lvl = static_cast<int>(-std::log(u) * m_l_);
    return std::min(lvl, 16);
}

float HnswIndex::dist_query_slot(const float* query, float q_norm, uint32_t slot) const noexcept {
    const float* v = vec_ptr(slot);
    if (metric_ == MetricType::COSINE) {
        return cosine_distance_with_norms(query, q_norm, v, norms_[slot], dim_);
    }
    return compute_distance(query, v, dim_, metric_);
}

float HnswIndex::dist_slots(uint32_t a, uint32_t b) const noexcept {
    const float* va = vec_ptr(a);
    const float* vb = vec_ptr(b);
    if (metric_ == MetricType::COSINE) {
        return cosine_distance_with_norms(va, norms_[a], vb, norms_[b], dim_);
    }
    return compute_distance(va, vb, dim_, metric_);
}

uint32_t HnswIndex::greedy_search_layer(const float* query, float q_norm, uint32_t entry, int layer) const {
    uint32_t curr = entry;
    float curr_dist = dist_query_slot(query, q_norm, curr);
    bool improved = true;
    while (improved) {
        improved = false;
        if (layer == 0) {
            const uint32_t* adj = layer0_adj(curr);
            uint32_t deg = adj[0];
            for (uint32_t i = 1; i <= deg; ++i) {
                uint32_t nb = adj[i];
                float d = dist_query_slot(query, q_norm, nb);
                if (d < curr_dist) {
                    curr_dist = d;
                    curr = nb;
                    improved = true;
                }
            }
        } else {
            const auto& adj = upper_links_[curr][static_cast<size_t>(layer - 1)];
            for (uint32_t nb : adj) {
                float d = dist_query_slot(query, q_norm, nb);
                if (d < curr_dist) {
                    curr_dist = d;
                    curr = nb;
                    improved = true;
                }
            }
        }
    }
    return curr;
}

std::vector<HnswIndex::DistSlot> HnswIndex::search_layer_for_insert(
    const float* query, float q_norm, uint32_t entry, uint32_t ef, int layer) const {
    thread_local VisitedTracker visited;
    visited.reset(slot_to_id_.size());
    visited.visit(entry);

    struct MinCmp {
        bool operator()(const DistSlot& a, const DistSlot& b) const { return a.dist > b.dist; }
    };
    struct MaxCmp {
        bool operator()(const DistSlot& a, const DistSlot& b) const { return a.dist < b.dist; }
    };

    std::priority_queue<DistSlot, std::vector<DistSlot>, MinCmp> candidates;
    std::priority_queue<DistSlot, std::vector<DistSlot>, MaxCmp> top_ef;

    float d0 = dist_query_slot(query, q_norm, entry);
    candidates.push({d0, entry});
    top_ef.push({d0, entry});

    while (!candidates.empty()) {
        DistSlot c = candidates.top();
        candidates.pop();
        if (top_ef.size() >= ef && c.dist > top_ef.top().dist) {
            break;
        }

        if (layer == 0) {
            const uint32_t* adj = layer0_adj(c.slot);
            uint32_t deg = adj[0];
            for (uint32_t i = 1; i <= deg; ++i) {
                uint32_t nb = adj[i];
                if (!visited.visit(nb)) continue;
                float d = dist_query_slot(query, q_norm, nb);
                if (top_ef.size() < ef || d < top_ef.top().dist) {
                    candidates.push({d, nb});
                    top_ef.push({d, nb});
                    if (top_ef.size() > ef) {
                        top_ef.pop();
                    }
                }
            }
        } else {
            const auto& adj = upper_links_[c.slot][static_cast<size_t>(layer - 1)];
            for (uint32_t nb : adj) {
                if (!visited.visit(nb)) continue;
                float d = dist_query_slot(query, q_norm, nb);
                if (top_ef.size() < ef || d < top_ef.top().dist) {
                    candidates.push({d, nb});
                    top_ef.push({d, nb});
                    if (top_ef.size() > ef) {
                        top_ef.pop();
                    }
                }
            }
        }
    }

    std::vector<DistSlot> out;
    out.reserve(top_ef.size());
    while (!top_ef.empty()) {
        out.push_back(top_ef.top());
        top_ef.pop();
    }
    std::sort(out.begin(), out.end(), [](const DistSlot& a, const DistSlot& b) {
        return a.dist < b.dist;
    });
    return out;
}

void HnswIndex::select_neighbors_heuristic(
    uint32_t target_slot,
    const float* target_vec,
    float target_norm,
    std::vector<DistSlot>& candidates,
    uint32_t m_limit,
    std::vector<uint32_t>& out) const {
    (void)target_vec;
    (void)target_norm;
    std::sort(candidates.begin(), candidates.end(), [](const DistSlot& a, const DistSlot& b) {
        return a.dist < b.dist;
    });

    out.clear();
    out.reserve(m_limit);
    std::vector<DistSlot> discarded;

    for (const auto& c : candidates) {
        if (c.slot == target_slot) continue;
        if (deleted_[c.slot]) continue; // Do not preferentially wire new edges to tombstoned nodes
        bool good = true;
        for (uint32_t r : out) {
            if (dist_slots(c.slot, r) < c.dist) {
                good = false;
                break;
            }
        }
        if (good) {
            out.push_back(c.slot);
            if (out.size() >= m_limit) break;
        } else {
            discarded.push_back(c);
        }
    }

    // Keep pruned connections to prevent sparse disconnected regions
    for (const auto& c : discarded) {
        if (out.size() >= m_limit) break;
        out.push_back(c.slot);
    }
}

void HnswIndex::connect_new_node(uint32_t new_slot, const std::vector<uint32_t>& neighbors, int layer) {
    uint32_t m_limit = (layer == 0) ? M_max0_ : M_;
    if (layer == 0) {
        uint32_t* new_adj = layer0_adj(new_slot);
        uint32_t deg = std::min<uint32_t>(static_cast<uint32_t>(neighbors.size()), m_limit);
        new_adj[0] = deg;
        for (uint32_t i = 0; i < deg; ++i) {
            new_adj[i + 1] = neighbors[i];
        }
        for (uint32_t i = 0; i < deg; ++i) {
            uint32_t nb = neighbors[i];
            uint32_t* nb_adj = layer0_adj(nb);
            uint32_t nb_deg = nb_adj[0];
            if (nb_deg < m_limit) {
                nb_adj[nb_deg + 1] = new_slot;
                nb_adj[0] = nb_deg + 1;
            } else {
                std::vector<DistSlot> cand;
                cand.reserve(nb_deg + 1);
                for (uint32_t j = 1; j <= nb_deg; ++j) {
                    cand.push_back({dist_slots(nb, nb_adj[j]), nb_adj[j]});
                }
                cand.push_back({dist_slots(nb, new_slot), new_slot});
                std::vector<uint32_t> selected;
                select_neighbors_heuristic(nb, vec_ptr(nb), norms_[nb], cand, m_limit, selected);
                nb_adj[0] = static_cast<uint32_t>(selected.size());
                for (uint32_t j = 0; j < selected.size(); ++j) {
                    nb_adj[j + 1] = selected[j];
                }
            }
        }
    } else {
        auto& new_adj = upper_links_[new_slot][static_cast<size_t>(layer - 1)];
        new_adj.assign(neighbors.begin(), neighbors.begin() + std::min(neighbors.size(), static_cast<size_t>(m_limit)));
        for (uint32_t nb : new_adj) {
            auto& nb_adj = upper_links_[nb][static_cast<size_t>(layer - 1)];
            if (nb_adj.size() < m_limit) {
                nb_adj.push_back(new_slot);
            } else {
                std::vector<DistSlot> cand;
                cand.reserve(nb_adj.size() + 1);
                for (uint32_t existing : nb_adj) {
                    cand.push_back({dist_slots(nb, existing), existing});
                }
                cand.push_back({dist_slots(nb, new_slot), new_slot});
                std::vector<uint32_t> selected;
                select_neighbors_heuristic(nb, vec_ptr(nb), norms_[nb], cand, m_limit, selected);
                nb_adj = std::move(selected);
            }
        }
    }
}

void HnswIndex::upsert(VectorId id, const std::vector<float>& vec) {
    if (vec.size() != dim_) {
        throw std::invalid_argument("HNSW vector dimension mismatch");
    }
    if (external_vectors_ != nullptr) {
        throw std::logic_error("Cannot insert into HNSW attached to external read-only mmap vectors");
    }

    float norm = vector_norm(vec.data(), dim_);
    std::unique_lock lock(mutex_);

    auto old_it = id_to_slot_.find(id);
    if (old_it != id_to_slot_.end()) {
        uint32_t old_slot = old_it->second;
        deleted_[old_slot] = 1;
        id_to_slot_.erase(old_it);
        --live_count_;
        ++tombstone_count_;
    }

    uint32_t new_slot = static_cast<uint32_t>(slot_to_id_.size());
    int level = random_level();

    owned_vectors_.insert(owned_vectors_.end(), vec.begin(), vec.end());
    norms_.push_back(norm);
    slot_to_id_.push_back(id);
    levels_.push_back(static_cast<uint8_t>(level));
    deleted_.push_back(0);
    layer0_links_.resize(layer0_links_.size() + (M_max0_ + 1), 0);
    upper_links_.emplace_back(static_cast<size_t>(std::max(0, level)));
    id_to_slot_[id] = new_slot;
    ++live_count_;

    if (entry_slot_ == UINT32_MAX) {
        entry_slot_ = new_slot;
        max_level_ = level;
        return;
    }

    uint32_t curr = entry_slot_;
    for (int l = max_level_; l > level; --l) {
        curr = greedy_search_layer(vec.data(), norm, curr, l);
    }

    std::vector<uint32_t> selected;
    for (int l = std::min(level, max_level_); l >= 0; --l) {
        auto cands = search_layer_for_insert(vec.data(), norm, curr, ef_construction_, l);
        uint32_t m_limit = (l == 0) ? M_max0_ : M_;
        select_neighbors_heuristic(new_slot, vec.data(), norm, cands, m_limit, selected);
        connect_new_node(new_slot, selected, l);
        if (!cands.empty()) {
            curr = cands[0].slot;
        }
    }

    if (level > max_level_) {
        entry_slot_ = new_slot;
        max_level_ = level;
    }
}

bool HnswIndex::remove(VectorId id) {
    std::unique_lock lock(mutex_);
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) {
        return false;
    }
    uint32_t slot = it->second;
    deleted_[slot] = 1;
    id_to_slot_.erase(it);
    --live_count_;
    ++tombstone_count_;
    return true;
}

bool HnswIndex::contains(VectorId id) const {
    std::shared_lock lock(mutex_);
    return id_to_slot_.find(id) != id_to_slot_.end();
}

bool HnswIndex::get(VectorId id, std::vector<float>& out) const {
    std::shared_lock lock(mutex_);
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) {
        return false;
    }
    const float* p = vec_ptr(it->second);
    out.assign(p, p + dim_);
    return true;
}

std::vector<SearchResult> HnswIndex::search(
    const std::vector<float>& query,
    uint32_t k,
    uint32_t ef_override,
    const FilterFn& filter) const {
    if (query.size() != dim_) {
        throw std::invalid_argument("HNSW query dimension mismatch");
    }
    if (k == 0) return {};

    float q_norm = vector_norm(query.data(), dim_);
    std::shared_lock lock(mutex_);
    if (entry_slot_ == UINT32_MAX || live_count_ == 0) {
        return {};
    }

    uint32_t ef = ef_override > 0 ? ef_override : ef_search_;
    if (filter) {
        ef = std::max(ef, k * 8u);
    } else if (tombstone_count_ > 0) {
        ef = std::max(ef, k + static_cast<uint32_t>(k * tombstone_count_ / std::max<size_t>(1, live_count_) + 4));
    }
    ef = std::max(ef, k);

    uint32_t curr = entry_slot_;
    for (int l = max_level_; l > 0; --l) {
        curr = greedy_search_layer(query.data(), q_norm, curr, l);
    }

    thread_local VisitedTracker visited;
    visited.reset(slot_to_id_.size());
    visited.visit(curr);

    struct MinCmp {
        bool operator()(const DistSlot& a, const DistSlot& b) const { return a.dist > b.dist; }
    };
    struct MaxCmp {
        bool operator()(const DistSlot& a, const DistSlot& b) const { return a.dist < b.dist; }
    };

    std::priority_queue<DistSlot, std::vector<DistSlot>, MinCmp> candidates;
    std::priority_queue<DistSlot, std::vector<DistSlot>, MaxCmp> beam;
    std::priority_queue<DistSlot, std::vector<DistSlot>, MaxCmp> valid_topk;

    float d0 = dist_query_slot(query.data(), q_norm, curr);
    candidates.push({d0, curr});
    beam.push({d0, curr});
    if (!deleted_[curr] && (!filter || filter(slot_to_id_[curr]))) {
        valid_topk.push({d0, curr});
        if (valid_topk.size() > k) valid_topk.pop();
    }

    while (!candidates.empty()) {
        DistSlot c = candidates.top();
        candidates.pop();
        if (beam.size() >= ef && c.dist > beam.top().dist) {
            if (!filter || valid_topk.size() >= k) {
                break;
            }
            // If filtering and still haven't collected k valid matches, cap extra exploration
            if (visited.epoch > 0 && beam.size() >= ef * 2) {
                break;
            }
        }

        const uint32_t* adj = layer0_adj(c.slot);
        uint32_t deg = adj[0];
        for (uint32_t i = 1; i <= deg; ++i) {
            uint32_t nb = adj[i];
            if (!visited.visit(nb)) continue;
            float d = dist_query_slot(query.data(), q_norm, nb);
            if (beam.size() < ef || d < beam.top().dist) {
                candidates.push({d, nb});
                beam.push({d, nb});
                if (beam.size() > ef) {
                    beam.pop();
                }
                if (!deleted_[nb] && (!filter || filter(slot_to_id_[nb]))) {
                    if (valid_topk.size() < k) {
                        valid_topk.push({d, nb});
                    } else if (d < valid_topk.top().dist) {
                        valid_topk.pop();
                        valid_topk.push({d, nb});
                    }
                }
            } else if (filter && valid_topk.size() < k && !deleted_[nb] && filter(slot_to_id_[nb])) {
                valid_topk.push({d, nb});
                candidates.push({d, nb});
            }
        }
    }

    std::vector<SearchResult> results;
    results.reserve(valid_topk.size());
    while (!valid_topk.empty()) {
        DistSlot ds = valid_topk.top();
        valid_topk.pop();
        results.push_back(SearchResult{
            slot_to_id_[ds.slot],
            ds.dist,
            distance_to_score(ds.dist, metric_),
            external_vectors_ ? SegmentTier::WARM_MMAP_HNSW : SegmentTier::HOT_RAM_HNSW
        });
    }
    std::reverse(results.begin(), results.end());
    return results;
}

std::vector<SearchResult> HnswIndex::search_with_allowlist(
    const std::vector<float>& query,
    uint32_t k,
    const std::unordered_set<VectorId>& allowed_ids,
    uint32_t ef_override) const {
    if (allowed_ids.empty()) return {};
    // Fast exact path when allowlist is small compared to segment size
    {
        std::shared_lock lock(mutex_);
        if (allowed_ids.size() <= 256 || allowed_ids.size() * 8 < live_count_) {
            float q_norm = vector_norm(query.data(), dim_);
            struct MaxCmp {
                bool operator()(const DistSlot& a, const DistSlot& b) const { return a.dist < b.dist; }
            };
            std::priority_queue<DistSlot, std::vector<DistSlot>, MaxCmp> topk;
            for (VectorId id : allowed_ids) {
                auto it = id_to_slot_.find(id);
                if (it == id_to_slot_.end()) continue;
                uint32_t slot = it->second;
                if (deleted_[slot]) continue;
                float d = dist_query_slot(query.data(), q_norm, slot);
                if (topk.size() < k) {
                    topk.push({d, slot});
                } else if (d < topk.top().dist) {
                    topk.pop();
                    topk.push({d, slot});
                }
            }
            std::vector<SearchResult> res;
            res.reserve(topk.size());
            while (!topk.empty()) {
                DistSlot ds = topk.top();
                topk.pop();
                res.push_back(SearchResult{
                    slot_to_id_[ds.slot],
                    ds.dist,
                    distance_to_score(ds.dist, metric_),
                    external_vectors_ ? SegmentTier::WARM_MMAP_HNSW : SegmentTier::HOT_RAM_HNSW
                });
            }
            std::reverse(res.begin(), res.end());
            return res;
        }
    }
    return search(query, k, ef_override, [&allowed_ids](VectorId id) {
        return allowed_ids.find(id) != allowed_ids.end();
    });
}

void HnswIndex::compact() {
    if (external_vectors_ != nullptr) {
        throw std::logic_error("Cannot compact HNSW with external vectors");
    }
    std::unique_lock lock(mutex_);
    if (tombstone_count_ == 0) return;

    std::vector<std::pair<VectorId, std::vector<float>>> live;
    live.reserve(live_count_);
    for (uint32_t s = 0; s < slot_to_id_.size(); ++s) {
        if (!deleted_[s]) {
            const float* p = vec_ptr(s);
            live.emplace_back(slot_to_id_[s], std::vector<float>(p, p + dim_));
        }
    }

    entry_slot_ = UINT32_MAX;
    max_level_ = -1;
    live_count_ = 0;
    tombstone_count_ = 0;
    owned_vectors_.clear();
    norms_.clear();
    slot_to_id_.clear();
    levels_.clear();
    deleted_.clear();
    id_to_slot_.clear();
    layer0_links_.clear();
    upper_links_.clear();
    lock.unlock();

    for (const auto& item : live) {
        upsert(item.first, item.second);
    }
}

void HnswIndex::set_external_vectors(const float* external_vectors, bool release_owned_vectors) {
    std::unique_lock lock(mutex_);
    external_vectors_ = external_vectors;
    if (release_owned_vectors) {
        std::vector<float>().swap(owned_vectors_);
    }
}

void HnswIndex::restore_owned_vectors(std::vector<float>&& owned_vectors) {
    std::unique_lock lock(mutex_);
    owned_vectors_ = std::move(owned_vectors);
    external_vectors_ = nullptr;
}

void HnswIndex::sync_tombstones(const std::vector<uint8_t>& external_deleted) {
    std::unique_lock lock(mutex_);
    size_t n = std::min(deleted_.size(), external_deleted.size());
    for (size_t s = 0; s < n; ++s) {
        if (external_deleted[s] && !deleted_[s]) {
            deleted_[s] = 1;
            id_to_slot_.erase(slot_to_id_[s]);
            if (live_count_ > 0) --live_count_;
            ++tombstone_count_;
        }
    }
}

size_t HnswIndex::size() const {
    std::shared_lock lock(mutex_);
    return live_count_;
}

size_t HnswIndex::total_slots() const {
    std::shared_lock lock(mutex_);
    return slot_to_id_.size();
}

size_t HnswIndex::tombstone_count() const {
    std::shared_lock lock(mutex_);
    return tombstone_count_;
}

double HnswIndex::tombstone_ratio() const {
    std::shared_lock lock(mutex_);
    if (slot_to_id_.empty()) return 0.0;
    return static_cast<double>(tombstone_count_) / static_cast<double>(slot_to_id_.size());
}

uint64_t HnswIndex::vector_ram_bytes() const {
    std::shared_lock lock(mutex_);
    return owned_vectors_.capacity() * sizeof(float);
}

uint64_t HnswIndex::graph_ram_bytes() const {
    std::shared_lock lock(mutex_);
    uint64_t bytes = norms_.capacity() * sizeof(float)
                   + slot_to_id_.capacity() * sizeof(VectorId)
                   + levels_.capacity() * sizeof(uint8_t)
                   + deleted_.capacity() * sizeof(uint8_t)
                   + layer0_links_.capacity() * sizeof(uint32_t)
                   + id_to_slot_.size() * (sizeof(VectorId) + sizeof(uint32_t) + 24);
    for (const auto& node_layers : upper_links_) {
        bytes += node_layers.capacity() * sizeof(std::vector<uint32_t>);
        for (const auto& l : node_layers) {
            bytes += l.capacity() * sizeof(uint32_t);
        }
    }
    return bytes;
}

std::vector<std::pair<VectorId, std::vector<float>>> HnswIndex::export_live_vectors() const {
    std::shared_lock lock(mutex_);
    std::vector<std::pair<VectorId, std::vector<float>>> out;
    out.reserve(live_count_);
    for (uint32_t s = 0; s < slot_to_id_.size(); ++s) {
        if (!deleted_[s]) {
            const float* p = vec_ptr(s);
            out.emplace_back(slot_to_id_[s], std::vector<float>(p, p + dim_));
        }
    }
    return out;
}

namespace {
template <typename T>
void write_pod(std::ostream& os, const T& val) {
    os.write(reinterpret_cast<const char*>(&val), sizeof(T));
}

template <typename T>
bool read_pod(std::istream& is, T& val) {
    return static_cast<bool>(is.read(reinterpret_cast<char*>(&val), sizeof(T)));
}
} // namespace

bool HnswIndex::save_to_file(const std::string& path) const {
    std::shared_lock lock(mutex_);
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) return false;

    uint8_t has_vectors = (external_vectors_ == nullptr && !owned_vectors_.empty()) ? 1 : 0;
    uint32_t slots = static_cast<uint32_t>(slot_to_id_.size());
    uint8_t m_byte = static_cast<uint8_t>(metric_);

    write_pod(os, kHnswMagic);
    write_pod(os, kHnswVersion);
    write_pod(os, dim_);
    write_pod(os, m_byte);
    write_pod(os, M_);
    write_pod(os, M_max0_);
    write_pod(os, ef_construction_);
    write_pod(os, ef_search_);
    write_pod(os, entry_slot_);
    write_pod(os, max_level_);
    write_pod(os, slots);
    uint64_t lc = live_count_;
    uint64_t tc = tombstone_count_;
    write_pod(os, lc);
    write_pod(os, tc);
    write_pod(os, has_vectors);

    if (slots > 0) {
        if (has_vectors) {
            os.write(reinterpret_cast<const char*>(owned_vectors_.data()),
                     static_cast<std::streamsize>(owned_vectors_.size() * sizeof(float)));
        }
        os.write(reinterpret_cast<const char*>(norms_.data()),
                 static_cast<std::streamsize>(slots * sizeof(float)));
        os.write(reinterpret_cast<const char*>(slot_to_id_.data()),
                 static_cast<std::streamsize>(slots * sizeof(VectorId)));
        os.write(reinterpret_cast<const char*>(levels_.data()),
                 static_cast<std::streamsize>(slots * sizeof(uint8_t)));
        os.write(reinterpret_cast<const char*>(deleted_.data()),
                 static_cast<std::streamsize>(slots * sizeof(uint8_t)));
        os.write(reinterpret_cast<const char*>(layer0_links_.data()),
                 static_cast<std::streamsize>(layer0_links_.size() * sizeof(uint32_t)));

        for (uint32_t s = 0; s < slots; ++s) {
            uint32_t n_upper = static_cast<uint32_t>(upper_links_[s].size());
            write_pod(os, n_upper);
            for (const auto& vec_l : upper_links_[s]) {
                uint32_t deg = static_cast<uint32_t>(vec_l.size());
                write_pod(os, deg);
                if (deg > 0) {
                    os.write(reinterpret_cast<const char*>(vec_l.data()),
                             static_cast<std::streamsize>(deg * sizeof(uint32_t)));
                }
            }
        }
    }
    return static_cast<bool>(os);
}

bool HnswIndex::save_graph_only(const std::string& path) const {
    std::shared_lock lock(mutex_);
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os) return false;

    uint8_t has_vectors = 0;
    uint32_t slots = static_cast<uint32_t>(slot_to_id_.size());
    uint8_t m_byte = static_cast<uint8_t>(metric_);

    write_pod(os, kHnswMagic);
    write_pod(os, kHnswVersion);
    write_pod(os, dim_);
    write_pod(os, m_byte);
    write_pod(os, M_);
    write_pod(os, M_max0_);
    write_pod(os, ef_construction_);
    write_pod(os, ef_search_);
    write_pod(os, entry_slot_);
    write_pod(os, max_level_);
    write_pod(os, slots);
    uint64_t lc = live_count_;
    uint64_t tc = tombstone_count_;
    write_pod(os, lc);
    write_pod(os, tc);
    write_pod(os, has_vectors);

    if (slots > 0) {
        os.write(reinterpret_cast<const char*>(norms_.data()),
                 static_cast<std::streamsize>(slots * sizeof(float)));
        os.write(reinterpret_cast<const char*>(slot_to_id_.data()),
                 static_cast<std::streamsize>(slots * sizeof(VectorId)));
        os.write(reinterpret_cast<const char*>(levels_.data()),
                 static_cast<std::streamsize>(slots * sizeof(uint8_t)));
        os.write(reinterpret_cast<const char*>(deleted_.data()),
                 static_cast<std::streamsize>(slots * sizeof(uint8_t)));
        os.write(reinterpret_cast<const char*>(layer0_links_.data()),
                 static_cast<std::streamsize>(layer0_links_.size() * sizeof(uint32_t)));

        for (uint32_t s = 0; s < slots; ++s) {
            uint32_t n_upper = static_cast<uint32_t>(upper_links_[s].size());
            write_pod(os, n_upper);
            for (const auto& vec_l : upper_links_[s]) {
                uint32_t deg = static_cast<uint32_t>(vec_l.size());
                write_pod(os, deg);
                if (deg > 0) {
                    os.write(reinterpret_cast<const char*>(vec_l.data()),
                             static_cast<std::streamsize>(deg * sizeof(uint32_t)));
                }
            }
        }
    }
    return static_cast<bool>(os);
}

std::unique_ptr<HnswIndex> HnswIndex::load_from_file(const std::string& path) {
    return load_graph_with_external_vectors(path, nullptr, 0);
}

std::unique_ptr<HnswIndex> HnswIndex::load_graph_with_external_vectors(
    const std::string& graph_path,
    const float* external_vectors,
    uint32_t expected_slots) {
    std::ifstream is(graph_path, std::ios::binary);
    if (!is) return nullptr;

    uint32_t magic = 0, version = 0, dim = 0, M = 0, M_max0 = 0, ef_c = 0, ef_s = 0, entry = 0, slots = 0;
    uint8_t m_byte = 0, has_vectors = 0;
    int max_level = -1;
    uint64_t lc = 0, tc = 0;

    if (!read_pod(is, magic) || magic != kHnswMagic) return nullptr;
    if (!read_pod(is, version) || version != kHnswVersion) return nullptr;
    if (!read_pod(is, dim) || !read_pod(is, m_byte) || !read_pod(is, M) || !read_pod(is, M_max0)) return nullptr;
    if (!read_pod(is, ef_c) || !read_pod(is, ef_s) || !read_pod(is, entry) || !read_pod(is, max_level)) return nullptr;
    if (!read_pod(is, slots) || !read_pod(is, lc) || !read_pod(is, tc) || !read_pod(is, has_vectors)) return nullptr;

    if (external_vectors && expected_slots != slots) {
        return nullptr;
    }

    auto idx = std::make_unique<HnswIndex>(dim, static_cast<MetricType>(m_byte), M, ef_c, ef_s);
    idx->M_max0_ = M_max0;
    idx->entry_slot_ = entry;
    idx->max_level_ = max_level;
    idx->live_count_ = static_cast<size_t>(lc);
    idx->tombstone_count_ = static_cast<size_t>(tc);
    idx->external_vectors_ = external_vectors;

    if (slots > 0) {
        if (has_vectors) {
            idx->owned_vectors_.resize(static_cast<size_t>(slots) * dim);
            is.read(reinterpret_cast<char*>(idx->owned_vectors_.data()),
                    static_cast<std::streamsize>(idx->owned_vectors_.size() * sizeof(float)));
        }
        idx->norms_.resize(slots);
        idx->slot_to_id_.resize(slots);
        idx->levels_.resize(slots);
        idx->deleted_.resize(slots);
        idx->layer0_links_.resize(static_cast<size_t>(slots) * (M_max0 + 1));
        idx->upper_links_.resize(slots);

        is.read(reinterpret_cast<char*>(idx->norms_.data()), static_cast<std::streamsize>(slots * sizeof(float)));
        is.read(reinterpret_cast<char*>(idx->slot_to_id_.data()), static_cast<std::streamsize>(slots * sizeof(VectorId)));
        is.read(reinterpret_cast<char*>(idx->levels_.data()), static_cast<std::streamsize>(slots * sizeof(uint8_t)));
        is.read(reinterpret_cast<char*>(idx->deleted_.data()), static_cast<std::streamsize>(slots * sizeof(uint8_t)));
        is.read(reinterpret_cast<char*>(idx->layer0_links_.data()),
                static_cast<std::streamsize>(idx->layer0_links_.size() * sizeof(uint32_t)));

        for (uint32_t s = 0; s < slots; ++s) {
            if (!idx->deleted_[s]) {
                idx->id_to_slot_[idx->slot_to_id_[s]] = s;
            }
            uint32_t n_upper = 0;
            if (!read_pod(is, n_upper)) return nullptr;
            idx->upper_links_[s].resize(n_upper);
            for (uint32_t l = 0; l < n_upper; ++l) {
                uint32_t deg = 0;
                if (!read_pod(is, deg)) return nullptr;
                idx->upper_links_[s][l].resize(deg);
                if (deg > 0) {
                    is.read(reinterpret_cast<char*>(idx->upper_links_[s][l].data()),
                            static_cast<std::streamsize>(deg * sizeof(uint32_t)));
                }
            }
        }
    }
    if (!is) return nullptr;
    return idx;
}

} // namespace vectordb
