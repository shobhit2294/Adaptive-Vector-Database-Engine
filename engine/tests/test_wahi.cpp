#include <gtest/gtest.h>
#include "vectordb/brute_force.hpp"
#include "vectordb/database.hpp"
#include <filesystem>
#include <random>
#include <unordered_set>

using namespace vectordb;

namespace {
struct ScopedDir {
    std::string path;
    explicit ScopedDir(const std::string& name) {
        path = "/tmp/" + name;
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~ScopedDir() { std::filesystem::remove_all(path); }
};

double recall_at_k(const std::vector<SearchResult>& approx,
                   const std::vector<SearchResult>& truth,
                   uint32_t k) {
    std::unordered_set<VectorId> tset;
    for (size_t i = 0; i < std::min<size_t>(k, truth.size()); ++i) {
        tset.insert(truth[i].id);
    }
    if (tset.empty()) return 1.0;
    uint32_t hits = 0;
    for (size_t i = 0; i < std::min<size_t>(k, approx.size()); ++i) {
        if (tset.count(approx[i].id)) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(tset.size());
}
} // namespace

TEST(DatabaseWahiTest, PersistenceWalRecoveryAndCompaction) {
    ScopedDir dir("db_persist_test");
    DatabaseConfig cfg;
    cfg.data_dir = dir.path;
    cfg.dimension = 16;
    cfg.metric = MetricType::L2;
    cfg.segment_capacity = 200;
    cfg.pq_subspaces = 8;
    cfg.pq_centroids = 64;
    cfg.compaction_tombstone_ratio = 0.25;
    cfg.auto_background_adapt = false;

    std::mt19937 rng(101);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    BruteForceIndex truth(cfg.dimension, cfg.metric);

    {
        VectorDatabase db(cfg);
        ASSERT_TRUE(db.open());
        for (uint64_t id = 1; id <= 520; ++id) {
            std::vector<float> v(cfg.dimension);
            for (float& x : v) x = dist(rng);
            db.upsert(id, v);
            truth.upsert(id, v);
        }
        // Delete 90 vectors in first segment and update 20 vectors
        for (uint64_t id = 1; id <= 90; ++id) {
            EXPECT_TRUE(db.remove(id));
            truth.remove(id);
        }
        for (uint64_t id = 100; id < 120; ++id) {
            std::vector<float> v(cfg.dimension, 1.5f + 0.02f * static_cast<float>(id - 100));
            db.upsert(id, v);
            truth.upsert(id, v);
        }
        db.close();
    }

    // Reopen from disk + WAL replay (120 unsealed vectors + tombstones)
    VectorDatabase db2(cfg);
    ASSERT_TRUE(db2.open());
    EXPECT_EQ(db2.snapshot().live_vectors, truth.size());
    EXPECT_FALSE(db2.contains(1));
    EXPECT_TRUE(db2.contains(520));

    std::vector<float> q(cfg.dimension, 1.4f);
    auto gt = truth.search(q, 10);
    auto res = db2.search(q, SearchOptions{10, 64, 4, false});
    EXPECT_GT(recall_at_k(res, gt, 10), 0.90);

    // Compact tombstones in sealed segments
    size_t compacted = db2.compact_tombstones();
    EXPECT_GE(compacted, 1u);
    EXPECT_EQ(db2.snapshot().live_vectors, truth.size());

    auto res_after = db2.search(q, SearchOptions{10, 64, 4, false});
    EXPECT_GT(recall_at_k(res_after, gt, 10), 0.90);
}

TEST(DatabaseWahiTest, ShiftingHotspotAdaptationAndBudget) {
    ScopedDir dir("db_wahi_shift");
    constexpr uint32_t dim = 32;
    constexpr uint32_t seg_cap = 300;
    constexpr uint32_t num_clusters = 8; // 1 segment per cluster!

    std::mt19937 rng(42);
    std::normal_distribution<float> noise(0.0f, 0.20f);
    std::uniform_real_distribution<float> cdist(-3.0f, 3.0f);

    std::vector<std::vector<float>> centers(num_clusters, std::vector<float>(dim));
    for (uint32_t c = 0; c < num_clusters; ++c) {
        // Make clusters well-separated orthogonally + random
        for (uint32_t d = 0; d < dim; ++d) centers[c][d] = cdist(rng) * 0.2f;
        centers[c][c * 4] += 4.0f;
    }

    DatabaseConfig cfg;
    cfg.data_dir = dir.path;
    cfg.dimension = dim;
    cfg.metric = MetricType::COSINE;
    cfg.segment_capacity = seg_cap;
    cfg.hnsw_m = 12;
    cfg.hnsw_ef_construction = 80;
    cfg.hnsw_ef_search = 48;
    cfg.pq_subspaces = 16;
    cfg.pq_centroids = 128;
    cfg.disk_cache_bytes = 256 * 1024;
    // Set RAM budget so roughly 2-3 segments fit in HOT_RAM_HNSW while remaining stay WARM/COLD
    cfg.ram_budget_bytes = 900 * 1024;
    cfg.wahi_reconcile_interval_queries = 40;
    cfg.wahi_cooldown_ticks = 1;
    cfg.wahi_mode = WahiPolicyMode::WAHI_ADAPTIVE;

    VectorDatabase db(cfg);
    ASSERT_TRUE(db.open());
    BruteForceIndex bf(dim, MetricType::COSINE);

    // Insert 8 segments, one cluster per segment
    uint64_t next_id = 1;
    for (uint32_t c = 0; c < num_clusters; ++c) {
        std::vector<std::pair<VectorId, std::vector<float>>> batch;
        for (uint32_t i = 0; i < seg_cap; ++i) {
            std::vector<float> v(dim);
            for (uint32_t d = 0; d < dim; ++d) v[d] = centers[c][d] + noise(rng);
            batch.emplace_back(next_id, v);
            bf.upsert(next_id, v);
            ++next_id;
        }
        db.upsert_batch(batch);
    }

    // Phase A: Heavily query Cluster 0 (Segment 1)
    for (int i = 0; i < 120; ++i) {
        std::vector<float> q(dim);
        for (uint32_t d = 0; d < dim; ++d) q[d] = centers[0][d] + noise(rng);
        db.search(q, SearchOptions{10, 48, 4, false});
    }
    db.trigger_wahi_reconcile();

    auto tels_a = db.segment_telemetries();
    SegmentTier seg1_tier_a = SegmentTier::COLD_DISK_PQ;
    for (const auto& t : tels_a) {
        if (t.segment_id == 1) seg1_tier_a = t.tier;
    }
    EXPECT_TRUE(seg1_tier_a == SegmentTier::HOT_RAM_HNSW || seg1_tier_a == SegmentTier::WARM_MMAP_HNSW);

    auto snap_a = db.snapshot();
    EXPECT_GT(snap_a.segments_cold_disk_pq + snap_a.segments_warm_mmap, 0u);

    // Phase B: Shift workload exclusively to Cluster 7 (Segment 8)
    for (int i = 0; i < 200; ++i) {
        std::vector<float> q(dim);
        for (uint32_t d = 0; d < dim; ++d) q[d] = centers[7][d] + noise(rng);
        db.search(q, SearchOptions{10, 48, 4, false});
    }
    db.trigger_wahi_reconcile();

    auto tels_b = db.segment_telemetries();
    SegmentTier seg8_tier_b = SegmentTier::COLD_DISK_PQ;
    for (const auto& t : tels_b) {
        if (t.segment_id == 8) seg8_tier_b = t.tier;
    }
    EXPECT_EQ(seg8_tier_b, SegmentTier::HOT_RAM_HNSW);
    EXPECT_GT(db.metrics().wahi_promotions_total.load(), 0u);
    EXPECT_GT(db.metrics().wahi_demotions_total.load(), 0u);

    // Verify high recall on shifted hot cluster
    double recall_sum = 0.0;
    for (int i = 0; i < 20; ++i) {
        std::vector<float> q(dim);
        for (uint32_t d = 0; d < dim; ++d) q[d] = centers[7][d] + noise(rng);
        auto gt = bf.search(q, 10);
        auto res = db.search(q, SearchOptions{10, 48, 4, false});
        recall_sum += recall_at_k(res, gt, 10);
    }
    EXPECT_GT(recall_sum / 20.0, 0.90);

    // Verify Hybrid Allowlist Search across tiered segments
    std::unordered_set<VectorId> allowlist;
    for (VectorId id = 2101; id <= 2200; ++id) allowlist.insert(id); // inside cluster 7
    std::vector<float> q7(dim);
    for (uint32_t d = 0; d < dim; ++d) q7[d] = centers[7][d];
    auto gt_f = bf.search_with_allowlist(q7, 10, allowlist);
    auto res_f = db.search(q7, SearchOptions{10, 48, 4, false}, nullptr, &allowlist);
    EXPECT_EQ(res_f.size(), 10u);
    EXPECT_GT(recall_at_k(res_f, gt_f, 10), 0.95);
    for (const auto& r : res_f) {
        EXPECT_TRUE(allowlist.count(r.id) > 0);
    }
}
