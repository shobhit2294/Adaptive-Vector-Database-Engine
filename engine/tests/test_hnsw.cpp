#include <gtest/gtest.h>
#include "vectordb/brute_force.hpp"
#include "vectordb/hnsw.hpp"
#include <cstdio>
#include <random>
#include <unordered_set>

using namespace vectordb;

namespace {
double compute_recall(const std::vector<SearchResult>& approx,
                      const std::vector<SearchResult>& truth,
                      uint32_t k) {
    std::unordered_set<VectorId> truth_ids;
    for (size_t i = 0; i < std::min<size_t>(k, truth.size()); ++i) {
        truth_ids.insert(truth[i].id);
    }
    if (truth_ids.empty()) return 1.0;
    uint32_t hits = 0;
    for (size_t i = 0; i < std::min<size_t>(k, approx.size()); ++i) {
        if (truth_ids.count(approx[i].id)) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(truth_ids.size());
}
} // namespace

TEST(HnswTest, RecallAt10OnClusteredData) {
    constexpr uint32_t dim = 32;
    constexpr uint32_t num_vecs = 2500;
    constexpr uint32_t num_queries = 50;
    constexpr uint32_t k = 10;

    std::mt19937 rng(123);
    std::normal_distribution<float> noise(0.0f, 0.25f);
    std::uniform_real_distribution<float> center_dist(-3.0f, 3.0f);

    std::vector<std::vector<float>> centers(10, std::vector<float>(dim));
    for (auto& c : centers) {
        for (float& v : c) v = center_dist(rng);
    }

    BruteForceIndex bf(dim, MetricType::COSINE);
    HnswIndex hnsw(dim, MetricType::COSINE, 16, 128, 64);

    for (uint32_t i = 0; i < num_vecs; ++i) {
        const auto& c = centers[i % centers.size()];
        std::vector<float> v(dim);
        for (uint32_t d = 0; d < dim; ++d) v[d] = c[d] + noise(rng);
        bf.upsert(i + 1, v);
        hnsw.upsert(i + 1, v);
    }

    double recall_sum = 0.0;
    for (uint32_t q = 0; q < num_queries; ++q) {
        const auto& c = centers[q % centers.size()];
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; ++d) query[d] = c[d] + noise(rng);
        auto truth = bf.search(query, k);
        auto approx = hnsw.search(query, k);
        recall_sum += compute_recall(approx, truth, k);
    }

    double avg_recall = recall_sum / num_queries;
    EXPECT_GT(avg_recall, 0.92);
}

TEST(HnswTest, DynamicUpdateDeleteCompactAndPersistence) {
    constexpr uint32_t dim = 16;
    std::mt19937 rng(77);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    BruteForceIndex bf(dim, MetricType::L2);
    HnswIndex hnsw(dim, MetricType::L2, 12, 96, 64);

    for (uint32_t i = 1; i <= 600; ++i) {
        std::vector<float> v(dim);
        for (float& x : v) x = dist(rng);
        bf.upsert(i, v);
        hnsw.upsert(i, v);
    }

    // Delete 200 vectors and update 100 vectors
    for (uint32_t i = 1; i <= 200; ++i) {
        EXPECT_TRUE(bf.remove(i));
        EXPECT_TRUE(hnsw.remove(i));
    }
    for (uint32_t i = 201; i <= 300; ++i) {
        std::vector<float> v(dim);
        for (float& x : v) x = dist(rng) + 2.0f;
        bf.upsert(i, v);
        hnsw.upsert(i, v);
    }

    EXPECT_EQ(hnsw.size(), 400u);
    std::vector<float> q(dim, 0.2f);
    auto truth = bf.search(q, 10);
    auto approx = hnsw.search(q, 10);
    EXPECT_GT(compute_recall(approx, truth, 10), 0.90);

    // Compact away tombstones
    hnsw.compact();
    EXPECT_EQ(hnsw.tombstone_count(), 0u);
    EXPECT_EQ(hnsw.size(), 400u);
    approx = hnsw.search(q, 10);
    EXPECT_GT(compute_recall(approx, truth, 10), 0.90);

    // Save & Reload
    const std::string path = "/tmp/test_hnsw.bin";
    ASSERT_TRUE(hnsw.save_to_file(path));
    auto reloaded = HnswIndex::load_from_file(path);
    ASSERT_NE(reloaded, nullptr);
    EXPECT_EQ(reloaded->size(), 400u);
    auto approx2 = reloaded->search(q, 10);
    EXPECT_EQ(approx2.size(), approx.size());
    EXPECT_EQ(approx2[0].id, approx[0].id);
    std::remove(path.c_str());
}
