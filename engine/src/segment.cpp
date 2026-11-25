#include "vectordb/segment.hpp"
#include "vectordb/distance.hpp"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <queue>
#include <sstream>

namespace vectordb {

namespace {
constexpr uint32_t kMetaMagic = 0x534D4554; // "SMET"
constexpr uint32_t kMetaVersion = 1;
constexpr uint32_t kPqCodesMagic = 0x50514344; // "PQCD"
} // namespace

Segment::Segment(SegmentId id,
                 const std::string& seg_dir,
                 uint32_t dim,
                 MetricType metric,
                 uint32_t hnsw_m,
                 uint32_t hnsw_ef_construction,
                 uint32_t hnsw_ef_search,
                 uint32_t pq_subspaces)
    : id_(id),
      seg_dir_(seg_dir),
      dim_(dim),
      metric_(metric),
      hnsw_m_(hnsw_m),
      hnsw_ef_construction_(hnsw_ef_construction),
      hnsw_ef_search_(hnsw_ef_search),
      pq_subspaces_(std::min(dim, std::max(1u, pq_subspaces))) {
    while (pq_subspaces_ > 1 && (dim_ % pq_subspaces_) != 0) {
        --pq_subspaces_;
    }
    hnsw_ = std::make_unique<HnswIndex>(dim_, metric_, hnsw_m_, hnsw_ef_construction_, hnsw_ef_search_, 42ULL + id_);
}

std::string Segment::vec_path() const {
    return seg_dir_ + "/seg_" + std::to_string(id_) + ".vec";
}
std::string Segment::hnsw_path() const {
    return seg_dir_ + "/seg_" + std::to_string(id_) + ".hnsw";
}
std::string Segment::meta_path() const {
    return seg_dir_ + "/seg_" + std::to_string(id_) + ".meta";
}
std::string Segment::pq_path() const {
    return seg_dir_ + "/seg_" + std::to_string(id_) + ".pq";
}

bool Segment::save_meta_unlocked() const {
    std::filesystem::create_directories(seg_dir_);
    const std::string tmp = meta_path() + ".tmp";
    std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
    if (!os) return false;
    uint8_t m_byte = static_cast<uint8_t>(metric_);
    os.write(reinterpret_cast<const char*>(&kMetaMagic), sizeof(kMetaMagic));
    os.write(reinterpret_cast<const char*>(&kMetaVersion), sizeof(kMetaVersion));
    os.write(reinterpret_cast<const char*>(&id_), sizeof(id_));
    os.write(reinterpret_cast<const char*>(&dim_), sizeof(dim_));
    os.write(reinterpret_cast<const char*>(&m_byte), sizeof(m_byte));
    os.write(reinterpret_cast<const char*>(&total_slots_), sizeof(total_slots_));
    os.write(reinterpret_cast<const char*>(&live_count_), sizeof(live_count_));
    os.write(reinterpret_cast<const char*>(&tombstone_count_), sizeof(tombstone_count_));
    if (total_slots_ > 0) {
        os.write(reinterpret_cast<const char*>(slot_to_id_.data()),
                 static_cast<std::streamsize>(total_slots_ * sizeof(VectorId)));
        os.write(reinterpret_cast<const char*>(deleted_.data()),
                 static_cast<std::streamsize>(total_slots_ * sizeof(uint8_t)));
    }
    os.flush();
    if (!os) return false;
    os.close();
    std::error_code ec;
    std::filesystem::rename(tmp, meta_path(), ec);
    return !ec;
}

bool Segment::load_meta_unlocked() {
    std::ifstream is(meta_path(), std::ios::binary);
    if (!is) return false;
    uint32_t magic = 0, ver = 0, sid = 0, dim = 0;
    uint8_t m_byte = 0;
    is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    is.read(reinterpret_cast<char*>(&ver), sizeof(ver));
    if (!is || magic != kMetaMagic || ver != kMetaVersion) return false;
    is.read(reinterpret_cast<char*>(&sid), sizeof(sid));
    is.read(reinterpret_cast<char*>(&dim), sizeof(dim));
    is.read(reinterpret_cast<char*>(&m_byte), sizeof(m_byte));
    is.read(reinterpret_cast<char*>(&total_slots_), sizeof(total_slots_));
    is.read(reinterpret_cast<char*>(&live_count_), sizeof(live_count_));
    is.read(reinterpret_cast<char*>(&tombstone_count_), sizeof(tombstone_count_));

    dim_ = dim;
    metric_ = static_cast<MetricType>(m_byte);
    slot_to_id_.resize(total_slots_);
    deleted_.resize(total_slots_);
    id_to_slot_.clear();
    if (total_slots_ > 0) {
        is.read(reinterpret_cast<char*>(slot_to_id_.data()),
                static_cast<std::streamsize>(total_slots_ * sizeof(VectorId)));
        is.read(reinterpret_cast<char*>(deleted_.data()),
                static_cast<std::streamsize>(total_slots_ * sizeof(uint8_t)));
        for (uint32_t s = 0; s < total_slots_; ++s) {
            if (!deleted_[s]) {
                id_to_slot_[slot_to_id_[s]] = s;
            }
        }
    }
    return static_cast<bool>(is);
}

std::shared_ptr<Segment> Segment::load_from_disk(
    SegmentId id,
    const std::string& seg_dir,
    uint32_t hnsw_m,
    uint32_t hnsw_ef_construction,
    uint32_t hnsw_ef_search,
    uint32_t pq_subspaces,
    SegmentTier initial_tier) {
    auto seg = std::make_shared<Segment>(id, seg_dir, 1, MetricType::L2, hnsw_m, hnsw_ef_construction, hnsw_ef_search, pq_subspaces);
    std::unique_lock lock(seg->mutex_);
    if (!seg->load_meta_unlocked()) return nullptr;
    seg->pq_subspaces_ = std::min(seg->dim_, std::max(1u, pq_subspaces));
    while (seg->pq_subspaces_ > 1 && (seg->dim_ % seg->pq_subspaces_) != 0) {
        --seg->pq_subspaces_;
    }
    seg->sealed_ = true;
    seg->hnsw_.reset();

    std::error_code ec;
    uint64_t dbytes = 0;
    for (const auto& p : {seg->vec_path(), seg->hnsw_path(), seg->meta_path(), seg->pq_path()}) {
        if (std::filesystem::exists(p, ec)) {
            dbytes += std::filesystem::file_size(p, ec);
        }
    }
    seg->disk_bytes_ = dbytes;
    if (std::filesystem::exists(seg->hnsw_path(), ec)) {
        seg->cached_graph_bytes_ = std::filesystem::file_size(seg->hnsw_path(), ec);
    }

    if (initial_tier == SegmentTier::WARM_MMAP_HNSW) {
        if (!seg->mmap_vecs_.open(seg->vec_path())) return nullptr;
        seg->hnsw_ = HnswIndex::load_graph_with_external_vectors(
            seg->hnsw_path(), seg->mmap_vecs_.vectors(), seg->total_slots_);
        if (!seg->hnsw_) return nullptr;
        seg->hnsw_->sync_tombstones(seg->deleted_);
        seg->tier_ = SegmentTier::WARM_MMAP_HNSW;
    } else if (initial_tier == SegmentTier::COLD_DISK_PQ) {
        if (!seg->disk_reader_.open(seg->vec_path())) return nullptr;
        std::ifstream pq_is(seg->pq_path(), std::ios::binary);
        if (pq_is) {
            uint32_t magic = 0, slots = 0, m_pq = 0;
            pq_is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            pq_is.read(reinterpret_cast<char*>(&slots), sizeof(slots));
            pq_is.read(reinterpret_cast<char*>(&m_pq), sizeof(m_pq));
            if (magic == kPqCodesMagic && slots == seg->total_slots_) {
                seg->pq_subspaces_ = m_pq;
                seg->pq_codes_.resize(static_cast<size_t>(slots) * m_pq);
                pq_is.read(reinterpret_cast<char*>(seg->pq_codes_.data()),
                           static_cast<std::streamsize>(seg->pq_codes_.size()));
            }
        }
        seg->tier_ = SegmentTier::COLD_DISK_PQ;
    } else {
        std::vector<float> vecs;
        if (!seg->load_owned_vectors_unlocked(vecs)) return nullptr;
        seg->hnsw_ = HnswIndex::load_graph_with_external_vectors(seg->hnsw_path(), nullptr, seg->total_slots_);
        if (!seg->hnsw_) return nullptr;
        seg->hnsw_->restore_owned_vectors(std::move(vecs));
        seg->hnsw_->sync_tombstones(seg->deleted_);
        seg->tier_ = SegmentTier::HOT_RAM_HNSW;
    }
    return seg;
}

void Segment::insert_memtable(VectorId id, const std::vector<float>& vec) {
    std::unique_lock lock(mutex_);
    if (sealed_ || tier_ != SegmentTier::HOT_MEMTABLE) {
        throw std::logic_error("Cannot insert into sealed segment");
    }
    auto it = id_to_slot_.find(id);
    if (it != id_to_slot_.end()) {
        deleted_[it->second] = 1;
        id_to_slot_.erase(it);
        --live_count_;
        ++tombstone_count_;
    }
    hnsw_->upsert(id, vec);
    uint32_t new_slot = static_cast<uint32_t>(slot_to_id_.size());
    slot_to_id_.push_back(id);
    deleted_.push_back(0);
    id_to_slot_[id] = new_slot;
    total_slots_ = static_cast<uint32_t>(slot_to_id_.size());
    ++live_count_;
}

bool Segment::remove(VectorId id) {
    std::unique_lock lock(mutex_);
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) return false;
    uint32_t slot = it->second;
    deleted_[slot] = 1;
    id_to_slot_.erase(it);
    --live_count_;
    ++tombstone_count_;
    if (hnsw_) {
        hnsw_->remove(id);
    }
    if (sealed_) {
        save_meta_unlocked();
    }
    return true;
}

bool Segment::contains(VectorId id) const {
    std::shared_lock lock(mutex_);
    return id_to_slot_.find(id) != id_to_slot_.end();
}

bool Segment::get(VectorId id, std::vector<float>& out) const {
    std::shared_lock lock(mutex_);
    auto it = id_to_slot_.find(id);
    if (it == id_to_slot_.end()) return false;
    uint32_t slot = it->second;
    if (hnsw_) {
        return hnsw_->get(id, out);
    }
    if (disk_reader_.is_open()) {
        out.resize(dim_);
        return disk_reader_.read_slot(slot, out.data());
    }
    return false;
}

bool Segment::load_owned_vectors_unlocked(std::vector<float>& out_vecs) const {
    DiskVectorReader r;
    if (!r.open(vec_path())) return false;
    out_vecs.resize(static_cast<size_t>(r.count()) * r.dim());
    for (uint32_t s = 0; s < r.count(); ++s) {
        if (!r.read_slot(s, out_vecs.data() + static_cast<size_t>(s) * r.dim())) {
            return false;
        }
    }
    return true;
}

bool Segment::ensure_pq_codes_unlocked(const ProductQuantizer* shared_pq) {
    if (!pq_codes_.empty()) return true;

    // Try loading precomputed .pq file
    {
        std::ifstream is(pq_path(), std::ios::binary);
        if (is) {
            uint32_t magic = 0, slots = 0, m_pq = 0;
            is.read(reinterpret_cast<char*>(&magic), sizeof(magic));
            is.read(reinterpret_cast<char*>(&slots), sizeof(slots));
            is.read(reinterpret_cast<char*>(&m_pq), sizeof(m_pq));
            if (is && magic == kPqCodesMagic && slots == total_slots_ &&
                (!shared_pq || m_pq == shared_pq->subspaces())) {
                pq_subspaces_ = m_pq;
                pq_codes_.resize(static_cast<size_t>(slots) * m_pq);
                is.read(reinterpret_cast<char*>(pq_codes_.data()),
                        static_cast<std::streamsize>(pq_codes_.size()));
                if (is) return true;
                pq_codes_.clear();
            }
        }
    }

    if (!shared_pq || !shared_pq->is_trained()) {
        return false;
    }

    const float* vec_data = nullptr;
    std::vector<float> tmp_vecs;
    if (mmap_vecs_.is_open()) {
        vec_data = mmap_vecs_.vectors();
    } else if (hnsw_ && !hnsw_->owned_vectors().empty()) {
        vec_data = hnsw_->owned_vectors().data();
    } else {
        if (!load_owned_vectors_unlocked(tmp_vecs)) return false;
        vec_data = tmp_vecs.data();
    }

    pq_subspaces_ = shared_pq->subspaces();
    pq_codes_ = shared_pq->encode_batch(vec_data, total_slots_);

    // Persist PQ codes to disk so future HOT<->COLD transitions are instantaneous
    std::filesystem::create_directories(seg_dir_);
    std::ofstream os(pq_path(), std::ios::binary | std::ios::trunc);
    if (os) {
        os.write(reinterpret_cast<const char*>(&kPqCodesMagic), sizeof(kPqCodesMagic));
        os.write(reinterpret_cast<const char*>(&total_slots_), sizeof(total_slots_));
        os.write(reinterpret_cast<const char*>(&pq_subspaces_), sizeof(pq_subspaces_));
        os.write(reinterpret_cast<const char*>(pq_codes_.data()),
                 static_cast<std::streamsize>(pq_codes_.size()));
    }
    return true;
}

bool Segment::seal(const ProductQuantizer* shared_pq) {
    std::unique_lock lock(mutex_);
    if (sealed_) return true;
    std::filesystem::create_directories(seg_dir_);

    total_slots_ = static_cast<uint32_t>(hnsw_->total_slots());
    if (!write_vector_file(vec_path(), hnsw_->owned_vectors().data(), total_slots_, dim_)) {
        return false;
    }
    if (!hnsw_->save_graph_only(hnsw_path())) {
        return false;
    }
    if (!save_meta_unlocked()) {
        return false;
    }

    cached_graph_bytes_ = hnsw_->graph_ram_bytes();
    if (shared_pq && shared_pq->is_trained()) {
        // Precompute and persist .pq file on disk without retaining pq_codes_ in RAM while hot
        ensure_pq_codes_unlocked(shared_pq);
        std::vector<uint8_t>().swap(pq_codes_);
    }

    std::error_code ec;
    uint64_t dbytes = 0;
    for (const auto& p : {vec_path(), hnsw_path(), meta_path(), pq_path()}) {
        if (std::filesystem::exists(p, ec)) dbytes += std::filesystem::file_size(p, ec);
    }
    disk_bytes_ = dbytes;
    sealed_ = true;
    tier_ = SegmentTier::HOT_RAM_HNSW;
    return true;
}

bool Segment::transition_to(SegmentTier target,
                            const ProductQuantizer* shared_pq,
                            BoundedVectorCache& cache) {
    if (target == SegmentTier::HOT_MEMTABLE) return false;
    std::unique_lock lock(mutex_);
    if (!sealed_) return false;
    if (tier_ == target) return true;

    if (target == SegmentTier::COLD_DISK_PQ) {
        if (!ensure_pq_codes_unlocked(shared_pq)) {
            return false;
        }
        if (hnsw_) {
            cached_graph_bytes_ = hnsw_->graph_ram_bytes();
            hnsw_.reset();
        }
        if (mmap_vecs_.is_open()) {
            mmap_vecs_.advise_dontneed();
            mmap_vecs_.close();
        }
        if (!disk_reader_.is_open()) {
            if (!disk_reader_.open(vec_path())) return false;
        }
        tier_ = SegmentTier::COLD_DISK_PQ;
        return true;
    }

    if (target == SegmentTier::WARM_MMAP_HNSW) {
        if (!mmap_vecs_.is_open()) {
            if (!mmap_vecs_.open(vec_path())) return false;
        }
        if (tier_ == SegmentTier::HOT_RAM_HNSW && hnsw_) {
            cached_graph_bytes_ = hnsw_->graph_ram_bytes();
            hnsw_->set_external_vectors(mmap_vecs_.vectors(), true);
        } else {
            hnsw_ = HnswIndex::load_graph_with_external_vectors(
                hnsw_path(), mmap_vecs_.vectors(), total_slots_);
            if (!hnsw_) {
                mmap_vecs_.close();
                return false;
            }
            hnsw_->sync_tombstones(deleted_);
            cached_graph_bytes_ = hnsw_->graph_ram_bytes();
        }
        std::vector<uint8_t>().swap(pq_codes_);
        disk_reader_.close();
        cache.invalidate_segment(id_);
        tier_ = SegmentTier::WARM_MMAP_HNSW;
        return true;
    }

    if (target == SegmentTier::HOT_RAM_HNSW) {
        std::vector<float> vecs;
        if (tier_ == SegmentTier::WARM_MMAP_HNSW && mmap_vecs_.is_open() && hnsw_) {
            size_t n_floats = static_cast<size_t>(total_slots_) * dim_;
            vecs.assign(mmap_vecs_.vectors(), mmap_vecs_.vectors() + n_floats);
            hnsw_->restore_owned_vectors(std::move(vecs));
            mmap_vecs_.close();
        } else {
            if (!load_owned_vectors_unlocked(vecs)) return false;
            hnsw_ = HnswIndex::load_graph_with_external_vectors(hnsw_path(), nullptr, total_slots_);
            if (!hnsw_) return false;
            hnsw_->restore_owned_vectors(std::move(vecs));
            hnsw_->sync_tombstones(deleted_);
        }
        cached_graph_bytes_ = hnsw_->graph_ram_bytes();
        std::vector<uint8_t>().swap(pq_codes_);
        disk_reader_.close();
        cache.invalidate_segment(id_);
        tier_ = SegmentTier::HOT_RAM_HNSW;
        return true;
    }
    return false;
}

std::vector<SearchResult> Segment::search(
    const std::vector<float>& query,
    const SearchOptions& opts,
    const ProductQuantizer* shared_pq,
    const float* shared_adc_table,
    BoundedVectorCache& cache,
    SearchDiagnostics& diag,
    const std::unordered_set<VectorId>* allowed_ids) const {
    std::shared_lock lock(mutex_);
    if (live_count_ == 0 || opts.k == 0) return {};

    // Fast skip when allowlist has zero overlap with this segment
    if (allowed_ids != nullptr) {
        bool any = false;
        if (allowed_ids->size() <= id_to_slot_.size()) {
            for (VectorId vid : *allowed_ids) {
                if (id_to_slot_.find(vid) != id_to_slot_.end()) {
                    any = true;
                    break;
                }
            }
        } else {
            any = true;
        }
        if (!any) return {};
    }

    ++diag.segments_scanned;
    if (tier_ == SegmentTier::HOT_MEMTABLE || tier_ == SegmentTier::HOT_RAM_HNSW) {
        ++diag.hot_segments_scanned;
        if (allowed_ids) {
            return hnsw_->search_with_allowlist(query, opts.k, *allowed_ids, opts.ef_search);
        }
        return hnsw_->search(query, opts.k, opts.ef_search);
    }

    if (tier_ == SegmentTier::WARM_MMAP_HNSW) {
        ++diag.warm_segments_scanned;
        auto res = allowed_ids
            ? hnsw_->search_with_allowlist(query, opts.k, *allowed_ids, opts.ef_search)
            : hnsw_->search(query, opts.k, opts.ef_search);
        for (auto& r : res) r.source_tier = SegmentTier::WARM_MMAP_HNSW;
        return res;
    }

    // COLD_DISK_PQ path: ADC scan in L1/L2 cache + budgeted exact rerank from disk via LRU cache
    ++diag.cold_segments_scanned;
    if (!shared_pq || !shared_adc_table || pq_codes_.empty()) {
        return {};
    }

    uint32_t rerank_factor = opts.cold_rerank_factor > 0 ? opts.cold_rerank_factor : 4;
    uint32_t cand_limit = opts.bypass_cold_rerank
        ? opts.k
        : std::max(opts.k, opts.k * rerank_factor);

    struct ApproxCand {
        float dist;
        uint32_t slot;
        bool operator<(const ApproxCand& o) const { return dist < o.dist; }
    };
    std::priority_queue<ApproxCand> approx_heap;

    if (allowed_ids && allowed_ids->size() <= 256) {
        for (VectorId vid : *allowed_ids) {
            auto it = id_to_slot_.find(vid);
            if (it == id_to_slot_.end()) continue;
            uint32_t s = it->second;
            float d = shared_pq->adc_distance(
                pq_codes_.data() + static_cast<size_t>(s) * pq_subspaces_, shared_adc_table);
            if (approx_heap.size() < cand_limit) {
                approx_heap.push({d, s});
            } else if (d < approx_heap.top().dist) {
                approx_heap.pop();
                approx_heap.push({d, s});
            }
        }
    } else {
        for (uint32_t s = 0; s < total_slots_; ++s) {
            if (deleted_[s]) continue;
            if (allowed_ids && allowed_ids->find(slot_to_id_[s]) == allowed_ids->end()) continue;
            float d = shared_pq->adc_distance(
                pq_codes_.data() + static_cast<size_t>(s) * pq_subspaces_, shared_adc_table);
            if (approx_heap.size() < cand_limit) {
                approx_heap.push({d, s});
            } else if (d < approx_heap.top().dist) {
                approx_heap.pop();
                approx_heap.push({d, s});
            }
        }
    }

    std::vector<ApproxCand> cands;
    cands.reserve(approx_heap.size());
    while (!approx_heap.empty()) {
        cands.push_back(approx_heap.top());
        approx_heap.pop();
    }

    if (opts.bypass_cold_rerank || !disk_reader_.is_open()) {
        std::sort(cands.begin(), cands.end(), [](const ApproxCand& a, const ApproxCand& b) {
            return a.dist < b.dist;
        });
        std::vector<SearchResult> res;
        res.reserve(std::min<size_t>(opts.k, cands.size()));
        for (size_t i = 0; i < std::min<size_t>(opts.k, cands.size()); ++i) {
            res.push_back(SearchResult{
                slot_to_id_[cands[i].slot],
                cands[i].dist,
                distance_to_score(cands[i].dist, metric_),
                SegmentTier::COLD_DISK_PQ
            });
        }
        return res;
    }

    // Exact rerank of top-cand_limit candidates via BoundedVectorCache + pread()
    std::vector<float> vec_buf;
    float q_norm = vector_norm(query.data(), dim_);
    std::vector<SearchResult> reranked;
    reranked.reserve(cands.size());
    for (const auto& c : cands) {
        bool hit = false;
        uint64_t bytes = 0;
        if (!cache.get_or_read(id_, c.slot, disk_reader_, vec_buf, hit, bytes)) {
            continue;
        }
        if (hit) {
            ++diag.cache_hits;
        } else {
            ++diag.cache_misses;
            ++diag.disk_reads;
            diag.disk_bytes_read += bytes;
        }
        float exact_d = (metric_ == MetricType::COSINE)
            ? cosine_distance_with_norms(query.data(), q_norm, vec_buf.data(), vector_norm(vec_buf.data(), dim_), dim_)
            : compute_distance(query.data(), vec_buf.data(), dim_, metric_);
        reranked.push_back(SearchResult{
            slot_to_id_[c.slot],
            exact_d,
            distance_to_score(exact_d, metric_),
            SegmentTier::COLD_DISK_PQ
        });
    }
    std::sort(reranked.begin(), reranked.end(), [](const SearchResult& a, const SearchResult& b) {
        return a.distance < b.distance;
    });
    if (reranked.size() > opts.k) {
        reranked.resize(opts.k);
    }
    return reranked;
}

std::vector<std::pair<VectorId, std::vector<float>>> Segment::export_live_vectors() const {
    std::shared_lock lock(mutex_);
    std::vector<std::pair<VectorId, std::vector<float>>> out;
    out.reserve(live_count_);
    if (hnsw_) {
        return hnsw_->export_live_vectors();
    }
    std::vector<float> v(dim_);
    for (uint32_t s = 0; s < total_slots_; ++s) {
        if (!deleted_[s] && disk_reader_.is_open() && disk_reader_.read_slot(s, v.data())) {
            out.emplace_back(slot_to_id_[s], v);
        }
    }
    return out;
}

void Segment::record_scan(uint64_t tick) {
    std::unique_lock lock(mutex_);
    scan_freq_ewma_ += 1.0;
    last_access_tick_ = tick;
}

void Segment::record_useful_hit(uint64_t tick, double weight) {
    std::unique_lock lock(mutex_);
    useful_hits_ewma_ += weight;
    last_access_tick_ = tick;
    last_hit_tick_ = tick;
}

void Segment::record_write(uint64_t tick) {
    std::unique_lock lock(mutex_);
    write_freq_ewma_ += 1.0;
    last_access_tick_ = tick;
}

void Segment::decay_telemetry(double alpha) {
    std::unique_lock lock(mutex_);
    double keep = 1.0 - std::clamp(alpha, 0.01, 0.99);
    scan_freq_ewma_ *= keep;
    useful_hits_ewma_ *= keep;
    write_freq_ewma_ *= keep;
}

SegmentTier Segment::tier() const {
    std::shared_lock lock(mutex_);
    return tier_;
}

uint32_t Segment::total_slots() const {
    std::shared_lock lock(mutex_);
    return total_slots_;
}

uint32_t Segment::live_count() const {
    std::shared_lock lock(mutex_);
    return live_count_;
}

uint32_t Segment::tombstone_count() const {
    std::shared_lock lock(mutex_);
    return tombstone_count_;
}

double Segment::tombstone_ratio() const {
    std::shared_lock lock(mutex_);
    if (total_slots_ == 0) return 0.0;
    return static_cast<double>(tombstone_count_) / static_cast<double>(total_slots_);
}

bool Segment::is_sealed() const {
    std::shared_lock lock(mutex_);
    return sealed_;
}

uint64_t Segment::estimated_hot_ram_bytes() const {
    std::shared_lock lock(mutex_);
    uint64_t vec_bytes = static_cast<uint64_t>(total_slots_) * dim_ * sizeof(float);
    uint64_t gbytes = hnsw_ ? hnsw_->graph_ram_bytes()
                            : (cached_graph_bytes_ > 0 ? cached_graph_bytes_
                                                       : static_cast<uint64_t>(total_slots_) * (hnsw_m_ * 2 + 4) * sizeof(uint32_t));
    return vec_bytes + gbytes;
}

uint64_t Segment::estimated_warm_ram_bytes() const {
    std::shared_lock lock(mutex_);
    uint64_t gbytes = hnsw_ ? hnsw_->graph_ram_bytes()
                            : (cached_graph_bytes_ > 0 ? cached_graph_bytes_
                                                       : static_cast<uint64_t>(total_slots_) * (hnsw_m_ * 2 + 4) * sizeof(uint32_t));
    return gbytes;
}

uint64_t Segment::estimated_cold_ram_bytes() const {
    std::shared_lock lock(mutex_);
    uint64_t pq_b = static_cast<uint64_t>(total_slots_) * pq_subspaces_;
    uint64_t map_b = static_cast<uint64_t>(total_slots_) * (sizeof(VectorId) + sizeof(uint8_t) + 24);
    return pq_b + map_b;
}

SegmentTelemetry Segment::telemetry() const {
    std::shared_lock lock(mutex_);
    SegmentTelemetry t;
    t.segment_id = id_;
    t.tier = tier_;
    t.vector_count = total_slots_;
    t.live_count = live_count_;
    t.tombstone_count = tombstone_count_;
    t.scan_freq_ewma = scan_freq_ewma_;
    t.useful_hits_ewma = useful_hits_ewma_;
    t.write_freq_ewma = write_freq_ewma_;
    t.utility_score = utility_score_;
    t.last_access_tick = last_access_tick_;
    t.last_hit_tick = last_hit_tick_;
    t.last_transition_tick = last_transition_tick_;
    t.disk_bytes = disk_bytes_;
    if (tier_ == SegmentTier::HOT_MEMTABLE || tier_ == SegmentTier::HOT_RAM_HNSW) {
        t.ram_bytes_vectors = hnsw_ ? hnsw_->vector_ram_bytes() : 0;
        t.ram_bytes_hnsw = hnsw_ ? hnsw_->graph_ram_bytes() : 0;
    } else if (tier_ == SegmentTier::WARM_MMAP_HNSW) {
        t.ram_bytes_vectors = 0;
        t.ram_bytes_hnsw = hnsw_ ? hnsw_->graph_ram_bytes() : 0;
    } else {
        t.ram_bytes_pq = pq_codes_.capacity() + id_to_slot_.size() * 32 + slot_to_id_.capacity() * sizeof(VectorId);
    }
    return t;
}

void Segment::remove_disk_files() const {
    std::error_code ec;
    std::filesystem::remove(vec_path(), ec);
    std::filesystem::remove(hnsw_path(), ec);
    std::filesystem::remove(meta_path(), ec);
    std::filesystem::remove(pq_path(), ec);
}

} // namespace vectordb
