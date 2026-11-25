#include <gtest/gtest.h>
#include "vectordb/database.hpp"
#include <atomic>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

using namespace vectordb;

TEST(DatabaseConcurrencyTest, ConcurrentReadsWritesDeletesAndWahiAdaptation) {
    const std::string dir = "/tmp/db_concurrency_test";
    std::filesystem::remove_all(dir);

    DatabaseConfig cfg;
    cfg.data_dir = dir;
    cfg.dimension = 16;
    cfg.metric = MetricType::COSINE;
    cfg.segment_capacity = 150;
    cfg.pq_subspaces = 8;
    cfg.pq_centroids = 64;
    cfg.ram_budget_bytes = 250 * 1024;
    cfg.wahi_reconcile_interval_queries = 25;
    cfg.wal_fsync = FsyncPolicy::NONE;

    VectorDatabase db(cfg);
    ASSERT_TRUE(db.open());

    // Preload 400 vectors
    for (uint64_t i = 1; i <= 400; ++i) {
        std::vector<float> v(cfg.dimension, static_cast<float>(i % 17) * 0.1f);
        v[i % cfg.dimension] += 1.0f;
        db.upsert(i, v);
    }

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> searches_done{0};
    std::atomic<uint64_t> writes_done{0};

    std::vector<std::thread> readers;
    for (int t = 0; t < 4; ++t) {
        readers.emplace_back([&, t]() {
            std::mt19937 rng(100 + t);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            while (!stop.load(std::memory_order_relaxed)) {
                std::vector<float> q(cfg.dimension);
                for (float& x : q) x = dist(rng);
                auto res = db.search(q, SearchOptions{5, 32, 2, false});
                EXPECT_LE(res.size(), 5u);
                searches_done.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    std::vector<std::thread> writers;
    for (int w = 0; w < 2; ++w) {
        writers.emplace_back([&, w]() {
            std::mt19937 rng(500 + w);
            std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
            for (int i = 0; i < 200; ++i) {
                uint64_t id = 401 + static_cast<uint64_t>(w * 200 + i);
                std::vector<float> v(cfg.dimension);
                for (float& x : v) x = dist(rng);
                db.upsert(id, v);
                if (i % 5 == 0 && id > 50) {
                    db.remove(id - 50);
                }
                writes_done.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    for (auto& w : writers) w.join();
    stop.store(true, std::memory_order_relaxed);
    for (auto& r : readers) r.join();

    EXPECT_EQ(writes_done.load(), 400u);
    EXPECT_GT(searches_done.load(), 20u);
    EXPECT_GT(db.snapshot().live_vectors, 600u);

    auto prom = db.prometheus_metrics();
    EXPECT_NE(prom.find("vectordb_vectors_total"), std::string::npos);

    db.close();
    std::filesystem::remove_all(dir);
}
