#include "vectordb/database.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>

namespace vectordb {

using json = nlohmann::json;

VectorDatabase::VectorDatabase(DatabaseConfig config)
    : config_(std::move(config)),
      disk_cache_(config_.disk_cache_bytes) {
    if (config_.dimension == 0) {
        throw std::invalid_argument("Database dimension must be > 0");
    }
    shared_pq_ = std::make_unique<ProductQuantizer>(
        config_.dimension, config_.metric, config_.pq_subspaces, config_.pq_centroids);
}

VectorDatabase::~VectorDatabase() {
    close();
}

bool VectorDatabase::open() {
    std::lock_guard wlock(write_mutex_);
    if (is_open_) return true;

    std::filesystem::create_directories(config_.data_dir);
    std::filesystem::create_directories(segments_dir());

    if (std::filesystem::exists(pq_codebook_path())) {
        auto loaded_pq = std::make_unique<ProductQuantizer>();
        if (loaded_pq->load(pq_codebook_path()) && loaded_pq->dim() == config_.dimension) {
            shared_pq_ = std::move(loaded_pq);
        }
    }

    if (!load_manifest_and_segments()) {
        return false;
    }

    if (!wal_.open(wal_path(), config_.wal_fsync)) {
        return false;
    }

    memtable_ = std::make_shared<Segment>(
        next_segment_id_++,
        segments_dir(),
        config_.dimension,
        config_.metric,
        config_.hnsw_m,
        config_.hnsw_ef_construction,
        config_.hnsw_ef_search,
        config_.pq_subspaces);

    Lsn max_lsn = 0;
    Lsn wal_cp_lsn = 0;
    const Lsn durable_cp = last_checkpoint_lsn_;
    wal_.replay(
        [&](const WalRecord& rec) {
            if (rec.lsn <= durable_cp) return;
            if (rec.op == WalOp::UPSERT) {
                if (rec.vec.size() == config_.dimension) {
                    apply_upsert_in_memory(rec.id, rec.vec, true);
                }
            } else if (rec.op == WalOp::DELETE) {
                apply_delete_in_memory(rec.id);
            }
        },
        max_lsn,
        wal_cp_lsn);
    last_checkpoint_lsn_ = std::max(last_checkpoint_lsn_, wal_cp_lsn);

    is_open_ = true;
    return true;
}

void VectorDatabase::close() {
    std::lock_guard wlock(write_mutex_);
    if (!is_open_) return;
    {
        std::lock_guard alock(adapt_mutex_);
        std::unique_lock dlock(db_mutex_);
        save_manifest_locked();
        for (auto& s : sealed_segments_) {
            s.reset();
        }
        sealed_segments_.clear();
        memtable_.reset();
        id_to_segment_.clear();
    }
    wal_.close();
    disk_cache_.clear();
    is_open_ = false;
}

void VectorDatabase::apply_upsert_in_memory(VectorId id, const std::vector<float>& vec, bool during_recovery) {
    uint64_t tick = access_tick_.load(std::memory_order_relaxed);
    std::unique_lock dlock(db_mutex_);
    auto it = id_to_segment_.find(id);
    if (it != id_to_segment_.end()) {
        it->second->remove(id);
        it->second->record_write(tick);
        if (!during_recovery) metrics_.record_update();
    } else {
        if (!during_recovery) metrics_.record_insert();
    }

    memtable_->insert_memtable(id, vec);
    memtable_->record_write(tick);
    id_to_segment_[id] = memtable_;
}

bool VectorDatabase::apply_delete_in_memory(VectorId id) {
    uint64_t tick = access_tick_.load(std::memory_order_relaxed);
    std::unique_lock dlock(db_mutex_);
    auto it = id_to_segment_.find(id);
    if (it == id_to_segment_.end()) {
        return false;
    }
    auto seg = it->second;
    bool removed = seg->remove(id);
    seg->record_write(tick);
    id_to_segment_.erase(it);
    return removed;
}

void VectorDatabase::ensure_pq_trained_locked() {
    if (shared_pq_->is_trained()) return;

    std::vector<std::shared_ptr<Segment>> segs;
    {
        std::shared_lock dlock(db_mutex_);
        segs = sealed_segments_;
        if (memtable_ && memtable_->live_count() > 0) {
            segs.push_back(memtable_);
        }
    }

    std::vector<float> train_samples;
    uint32_t sample_count = 0;
    for (const auto& s : segs) {
        auto vecs = s->export_live_vectors();
        for (const auto& kv : vecs) {
            train_samples.insert(train_samples.end(), kv.second.begin(), kv.second.end());
            ++sample_count;
            if (sample_count >= 4096) break;
        }
        if (sample_count >= 4096) break;
    }

    if (sample_count >= 16) {
        shared_pq_->train(train_samples.data(), sample_count, config_.pq_train_iters);
        shared_pq_->save(pq_codebook_path());
    }
}

bool VectorDatabase::seal_memtable_locked(Lsn current_lsn) {
    if (!memtable_ || memtable_->total_slots() == 0) return true;

    ensure_pq_trained_locked();
    if (!memtable_->seal(shared_pq_->is_trained() ? shared_pq_.get() : nullptr)) {
        return false;
    }

    {
        std::unique_lock dlock(db_mutex_);
        sealed_segments_.push_back(memtable_);
        memtable_ = std::make_shared<Segment>(
            next_segment_id_++,
            segments_dir(),
            config_.dimension,
            config_.metric,
            config_.hnsw_m,
            config_.hnsw_ef_construction,
            config_.hnsw_ef_search,
            config_.pq_subspaces);
        last_checkpoint_lsn_ = std::max(last_checkpoint_lsn_, current_lsn);
        save_manifest_locked();
    }

    if (wal_.bytes_written() >= config_.wal_checkpoint_bytes) {
        wal_.truncate_and_checkpoint(last_checkpoint_lsn_);
        metrics_.record_checkpoint();
    }
    return true;
}

void VectorDatabase::upsert(VectorId id, const std::vector<float>& vec) {
    if (vec.size() != config_.dimension) {
        throw std::invalid_argument("Vector dimension mismatch");
    }
    std::lock_guard wlock(write_mutex_);
    if (!is_open_) throw std::logic_error("Database is not open");

    Lsn lsn = wal_.append_upsert(id, vec);
    apply_upsert_in_memory(id, vec, false);

    if (memtable_->total_slots() >= config_.segment_capacity) {
        std::lock_guard alock(adapt_mutex_);
        seal_memtable_locked(lsn);
        if (config_.auto_background_adapt) {
            std::vector<std::shared_ptr<Segment>> sealed_copy;
            uint64_t mem_bytes = 0;
            {
                std::shared_lock dlock(db_mutex_);
                sealed_copy = sealed_segments_;
                if (memtable_) mem_bytes = memtable_->telemetry().total_ram_bytes();
            }
            ensure_pq_trained_locked();
            uint64_t tick = access_tick_.load(std::memory_order_relaxed);
            wahi_.reconcile(config_, sealed_copy, mem_bytes, shared_pq_.get(), disk_cache_, metrics_, tick);
        }
    }
}

void VectorDatabase::upsert_batch(const std::vector<std::pair<VectorId, std::vector<float>>>& items) {
    if (items.empty()) return;
    std::lock_guard wlock(write_mutex_);
    if (!is_open_) throw std::logic_error("Database is not open");

    Lsn last_lsn = 0;
    bool sealed_any = false;
    for (const auto& [id, vec] : items) {
        if (vec.size() != config_.dimension) {
            throw std::invalid_argument("Vector dimension mismatch in batch");
        }
        last_lsn = wal_.append_upsert(id, vec);
        apply_upsert_in_memory(id, vec, false);
        if (memtable_->total_slots() >= config_.segment_capacity) {
            std::lock_guard alock(adapt_mutex_);
            seal_memtable_locked(last_lsn);
            sealed_any = true;
        }
    }
    wal_.sync();

    if (sealed_any && config_.auto_background_adapt) {
        std::lock_guard alock(adapt_mutex_);
        std::vector<std::shared_ptr<Segment>> sealed_copy;
        uint64_t mem_bytes = 0;
        {
            std::shared_lock dlock(db_mutex_);
            sealed_copy = sealed_segments_;
            if (memtable_) mem_bytes = memtable_->telemetry().total_ram_bytes();
        }
        ensure_pq_trained_locked();
        uint64_t tick = access_tick_.load(std::memory_order_relaxed);
        wahi_.reconcile(config_, sealed_copy, mem_bytes, shared_pq_.get(), disk_cache_, metrics_, tick);
    }
}

bool VectorDatabase::remove(VectorId id) {
    std::lock_guard wlock(write_mutex_);
    if (!is_open_) throw std::logic_error("Database is not open");

    {
        std::shared_lock dlock(db_mutex_);
        if (id_to_segment_.find(id) == id_to_segment_.end()) {
            return false;
        }
    }
    wal_.append_delete(id);
    bool removed = apply_delete_in_memory(id);
    if (removed) {
        metrics_.record_delete();
    }
    return removed;
}

bool VectorDatabase::get(VectorId id, std::vector<float>& out) const {
    std::shared_ptr<Segment> seg;
    {
        std::shared_lock dlock(db_mutex_);
        auto it = id_to_segment_.find(id);
        if (it == id_to_segment_.end()) return false;
        seg = it->second;
    }
    return seg->get(id, out);
}

bool VectorDatabase::contains(VectorId id) const {
    std::shared_lock dlock(db_mutex_);
    return id_to_segment_.find(id) != id_to_segment_.end();
}

std::vector<SearchResult> VectorDatabase::search(
    const std::vector<float>& query,
    const SearchOptions& opts,
    SearchDiagnostics* out_diag,
    const std::unordered_set<VectorId>* allowed_ids) {
    if (query.size() != config_.dimension) {
        throw std::invalid_argument("Query dimension mismatch");
    }
    auto t0 = std::chrono::steady_clock::now();
    SearchDiagnostics diag;
    if (opts.k == 0 || (allowed_ids && allowed_ids->empty())) {
        if (out_diag) *out_diag = diag;
        return {};
    }

    uint64_t tick = access_tick_.fetch_add(1, std::memory_order_relaxed) + 1;

    std::vector<std::shared_ptr<Segment>> segs;
    bool pq_trained = false;
    {
        std::shared_lock dlock(db_mutex_);
        segs.reserve(sealed_segments_.size() + 1);
        if (memtable_ && memtable_->live_count() > 0) {
            segs.push_back(memtable_);
        }
        for (const auto& s : sealed_segments_) {
            if (s->live_count() > 0) segs.push_back(s);
        }
        pq_trained = shared_pq_ && shared_pq_->is_trained();
    }

    if (segs.empty()) {
        if (out_diag) *out_diag = diag;
        return {};
    }

    SearchOptions eff_opts = opts;
    if (eff_opts.cold_rerank_factor == 0) {
        eff_opts.cold_rerank_factor = config_.cold_rerank_factor;
    }

    // Build shared ADC lookup table once per query if any cold segments exist
    std::vector<float> adc_table;
    bool need_adc = false;
    if (pq_trained) {
        for (const auto& s : segs) {
            if (s->tier() == SegmentTier::COLD_DISK_PQ) {
                need_adc = true;
                break;
            }
        }
        if (need_adc) {
            shared_pq_->build_adc_table(query.data(), adc_table);
        }
    }

    struct TaggedResult {
        SearchResult res;
        Segment* seg;
    };
    std::vector<TaggedResult> merged;
    merged.reserve(segs.size() * eff_opts.k);

    for (const auto& s : segs) {
        uint32_t before_scanned = diag.segments_scanned;
        auto local = s->search(
            query,
            eff_opts,
            pq_trained ? shared_pq_.get() : nullptr,
            adc_table.empty() ? nullptr : adc_table.data(),
            disk_cache_,
            diag,
            allowed_ids);
        if (diag.segments_scanned > before_scanned) {
            s->record_scan(tick);
        }
        for (const auto& r : local) {
            merged.push_back(TaggedResult{r, s.get()});
        }
    }

    std::sort(merged.begin(), merged.end(), [](const TaggedResult& a, const TaggedResult& b) {
        if (a.res.distance != b.res.distance) return a.res.distance < b.res.distance;
        return a.res.id < b.res.id;
    });

    std::vector<SearchResult> final_results;
    final_results.reserve(std::min<size_t>(eff_opts.k, merged.size()));
    std::unordered_set<VectorId> seen_ids;
    seen_ids.reserve(eff_opts.k * 2);

    for (const auto& item : merged) {
        if (!seen_ids.insert(item.res.id).second) continue;
        size_t rank = final_results.size();
        // Weight higher ranks slightly more when attributing useful hits to segments
        double weight = 1.0 / (1.0 + 0.15 * static_cast<double>(rank));
        item.seg->record_useful_hit(tick, weight);
        final_results.push_back(item.res);
        if (final_results.size() >= eff_opts.k) break;
    }

    auto t1 = std::chrono::steady_clock::now();
    diag.latency_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    diag.candidates_scored = static_cast<uint32_t>(merged.size());
    metrics_.record_query(diag.latency_ms, allowed_ids != nullptr, diag);

    if (out_diag) *out_diag = diag;

    if (config_.auto_background_adapt && config_.wahi_reconcile_interval_queries > 0) {
        uint64_t qn = query_count_since_adapt_.fetch_add(1, std::memory_order_relaxed) + 1;
        if ((qn % config_.wahi_reconcile_interval_queries) == 0) {
            trigger_wahi_reconcile();
        }
    }

    return final_results;
}

bool VectorDatabase::flush_and_seal() {
    std::lock_guard wlock(write_mutex_);
    if (!is_open_) return false;
    wal_.sync();
    std::lock_guard alock(adapt_mutex_);
    return seal_memtable_locked(wal_.current_lsn());
}

size_t VectorDatabase::compact_tombstones() {
    std::lock_guard wlock(write_mutex_);
    std::lock_guard alock(adapt_mutex_);
    if (!is_open_) return 0;

    std::vector<std::shared_ptr<Segment>> candidates;
    {
        std::shared_lock dlock(db_mutex_);
        for (const auto& s : sealed_segments_) {
            if (s->tombstone_ratio() >= config_.compaction_tombstone_ratio) {
                candidates.push_back(s);
            }
        }
    }
    if (candidates.empty()) return 0;

    ensure_pq_trained_locked();
    size_t compacted = 0;

    for (const auto& old_seg : candidates) {
        auto live = old_seg->export_live_vectors();
        SegmentTier orig_tier = old_seg->tier();

        std::shared_ptr<Segment> new_seg;
        if (!live.empty()) {
            new_seg = std::make_shared<Segment>(
                next_segment_id_++,
                segments_dir(),
                config_.dimension,
                config_.metric,
                config_.hnsw_m,
                config_.hnsw_ef_construction,
                config_.hnsw_ef_search,
                config_.pq_subspaces);
            for (const auto& [vid, vec] : live) {
                new_seg->insert_memtable(vid, vec);
            }
            new_seg->seal(shared_pq_->is_trained() ? shared_pq_.get() : nullptr);
            if (orig_tier != SegmentTier::HOT_RAM_HNSW) {
                new_seg->transition_to(orig_tier, shared_pq_.get(), disk_cache_);
            }
        }

        {
            std::unique_lock dlock(db_mutex_);
            auto pos = std::find(sealed_segments_.begin(), sealed_segments_.end(), old_seg);
            if (pos != sealed_segments_.end()) {
                if (new_seg) {
                    *pos = new_seg;
                    for (const auto& [vid, _] : live) {
                        auto map_it = id_to_segment_.find(vid);
                        if (map_it != id_to_segment_.end() && map_it->second == old_seg) {
                            map_it->second = new_seg;
                        }
                    }
                } else {
                    sealed_segments_.erase(pos);
                }
            }
            save_manifest_locked();
        }
        disk_cache_.invalidate_segment(old_seg->id());
        old_seg->remove_disk_files();
        metrics_.record_compaction();
        ++compacted;
    }

    return compacted;
}

std::vector<WahiDecision> VectorDatabase::trigger_wahi_reconcile() {
    std::lock_guard alock(adapt_mutex_);
    if (!is_open_) return {};

    ensure_pq_trained_locked();

    std::vector<std::shared_ptr<Segment>> sealed_copy;
    uint64_t mem_ram = 0;
    {
        std::shared_lock dlock(db_mutex_);
        sealed_copy = sealed_segments_;
        if (memtable_) mem_ram = memtable_->telemetry().total_ram_bytes();
    }
    uint64_t tick = access_tick_.load(std::memory_order_relaxed);
    auto decisions = wahi_.reconcile(
        config_,
        sealed_copy,
        mem_ram,
        shared_pq_->is_trained() ? shared_pq_.get() : nullptr,
        disk_cache_,
        metrics_,
        tick);

    if (!decisions.empty()) {
        std::unique_lock dlock(db_mutex_);
        save_manifest_locked();
    }
    return decisions;
}

void VectorDatabase::set_wahi_policy(WahiPolicyMode mode, uint64_t ram_budget_bytes) {
    {
        std::lock_guard alock(adapt_mutex_);
        config_.wahi_mode = mode;
        if (ram_budget_bytes > 0) {
            config_.ram_budget_bytes = ram_budget_bytes;
        }
    }
    trigger_wahi_reconcile();
}

void VectorDatabase::set_ram_budget(uint64_t ram_budget_bytes) {
    {
        std::lock_guard alock(adapt_mutex_);
        config_.ram_budget_bytes = ram_budget_bytes;
    }
    trigger_wahi_reconcile();
}

EngineSnapshot VectorDatabase::snapshot() const {
    EngineSnapshot snap;
    snap.ram_budget_bytes = config_.ram_budget_bytes;
    snap.ram_bytes_cache = disk_cache_.used_bytes();
    snap.wal_bytes = wal_.bytes_written();
    snap.wal_records = wal_.records_written();
    snap.wahi_mode = config_.wahi_mode;

    std::vector<std::shared_ptr<Segment>> all_segs;
    {
        std::shared_lock dlock(db_mutex_);
        if (memtable_) all_segs.push_back(memtable_);
        for (const auto& s : sealed_segments_) all_segs.push_back(s);
    }

    for (const auto& s : all_segs) {
        auto tel = s->telemetry();
        snap.live_vectors += tel.live_count;
        snap.tombstones += tel.tombstone_count;
        snap.ram_bytes_vectors += tel.ram_bytes_vectors;
        snap.ram_bytes_hnsw += tel.ram_bytes_hnsw;
        snap.ram_bytes_pq += tel.ram_bytes_pq;
        snap.disk_bytes_total += tel.disk_bytes;
        switch (tel.tier) {
            case SegmentTier::HOT_MEMTABLE: ++snap.segments_memtable; break;
            case SegmentTier::HOT_RAM_HNSW: ++snap.segments_hot_ram; break;
            case SegmentTier::WARM_MMAP_HNSW: ++snap.segments_warm_mmap; break;
            case SegmentTier::COLD_DISK_PQ: ++snap.segments_cold_disk_pq; break;
        }
    }
    if (shared_pq_ && shared_pq_->is_trained()) {
        snap.ram_bytes_pq += shared_pq_->codebook_ram_bytes();
    }
    snap.ram_bytes_total = snap.ram_bytes_vectors + snap.ram_bytes_hnsw + snap.ram_bytes_pq + snap.ram_bytes_cache;
    return snap;
}

std::vector<SegmentTelemetry> VectorDatabase::segment_telemetries() const {
    std::vector<std::shared_ptr<Segment>> all_segs;
    {
        std::shared_lock dlock(db_mutex_);
        if (memtable_) all_segs.push_back(memtable_);
        for (const auto& s : sealed_segments_) all_segs.push_back(s);
    }
    std::vector<SegmentTelemetry> out;
    out.reserve(all_segs.size());
    for (const auto& s : all_segs) {
        out.push_back(s->telemetry());
    }
    return out;
}

std::string VectorDatabase::prometheus_metrics() const {
    return metrics_.export_prometheus(snapshot());
}

bool VectorDatabase::save_manifest_locked() const {
    json j;
    j["dimension"] = config_.dimension;
    j["metric"] = to_string(config_.metric);
    j["next_segment_id"] = next_segment_id_;
    j["checkpoint_lsn"] = last_checkpoint_lsn_;
    json segs = json::array();
    for (const auto& s : sealed_segments_) {
        segs.push_back({
            {"id", s->id()},
            {"tier", static_cast<int>(s->tier())}
        });
    }
    j["sealed_segments"] = segs;

    const std::string tmp = manifest_path() + ".tmp";
    std::ofstream os(tmp, std::ios::trunc);
    if (!os) return false;
    os << j.dump(2);
    os.flush();
    if (!os) return false;
    os.close();
    std::error_code ec;
    std::filesystem::rename(tmp, manifest_path(), ec);
    return !ec;
}

bool VectorDatabase::load_manifest_and_segments() {
    if (!std::filesystem::exists(manifest_path())) {
        return true;
    }
    std::ifstream is(manifest_path());
    if (!is) return false;
    json j;
    try {
        is >> j;
    } catch (...) {
        return false;
    }

    next_segment_id_ = j.value("next_segment_id", 1u);
    last_checkpoint_lsn_ = j.value("checkpoint_lsn", 0ULL);

    sealed_segments_.clear();
    id_to_segment_.clear();

    if (j.contains("sealed_segments") && j["sealed_segments"].is_array()) {
        for (const auto& item : j["sealed_segments"]) {
            SegmentId sid = item.value("id", 0u);
            SegmentTier tier = static_cast<SegmentTier>(item.value("tier", static_cast<int>(SegmentTier::HOT_RAM_HNSW)));
            if (sid == 0) continue;
            auto seg = Segment::load_from_disk(
                sid,
                segments_dir(),
                config_.hnsw_m,
                config_.hnsw_ef_construction,
                config_.hnsw_ef_search,
                config_.pq_subspaces,
                tier);
            if (!seg) continue;
            sealed_segments_.push_back(seg);
            next_segment_id_ = std::max(next_segment_id_, sid + 1);
            auto live = seg->export_live_vectors();
            for (const auto& [vid, _] : live) {
                id_to_segment_[vid] = seg;
            }
        }
    }
    return true;
}

} // namespace vectordb
