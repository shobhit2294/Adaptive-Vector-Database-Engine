#include <gtest/gtest.h>
#include "vectordb/brute_force.hpp"
#include "vectordb/distance.hpp"
#include "vectordb/pq.hpp"
#include <algorithm>
#include <cstdio>
#include <random>
#include <unordered_set>

using namespace vectordb;

TEST(ProductQuantizerTest, AdcMatchesDecodedDistanceInvariant) {
    constexpr uint32_t dim = 16;
    constexpr uint32_t n = 400;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> flat(n * dim);
    for (float& x : flat) x = dist(rng);

    for (MetricType metric : {MetricType::L2, MetricType::DOT, MetricType::COSINE}) {
        ProductQuantizer pq(dim, metric, 4, 64);
        pq.train(flat.data(), n, 10, 7);

        std::vector<float> q(dim);
        for (float& x : q) x = dist(rng);

        std::vector<float> table;
        pq.build_adc_table(q.data(), table);

        std::vector<uint8_t> code(pq.subspaces());
        std::vector<float> decoded(dim);
        for (uint32_t i = 0; i < 25; ++i) {
            pq.encode(&flat[i * dim], code.data());
            pq.decode(code.data(), decoded.data());
            float adc = pq.adc_distance(code.data(), table.data());

            float expected = 0.0f;
            if (metric == MetricType::L2) {
                expected = l2_squared(q.data(), decoded.data(), dim);
            } else if (metric == MetricType::DOT) {
                expected = -dot_product(q.data(), decoded.data(), dim);
            } else {
                std::vector<float> qn = q;
                normalize_vector(qn.data(), dim);
                expected = 1.0f - dot_product(qn.data(), decoded.data(), dim);
            }
            EXPECT_NEAR(adc, expected, 1e-4f);
        }
    }
}

TEST(ProductQuantizerTest, TrainEncodeAdcRecallAndPersistence) {
    constexpr uint32_t dim = 32;
    constexpr uint32_t n = 2000;
    constexpr uint32_t k = 10;

    std::mt19937 rng(2026);
    std::normal_distribution<float> noise(0.0f, 0.25f);
    std::uniform_real_distribution<float> cdist(-2.0f, 2.0f);

    std::vector<std::vector<float>> centers(100, std::vector<float>(dim));
    for (auto& c : centers) {
        for (float& x : c) x = cdist(rng);
    }

    std::vector<float> flat(n * dim);
    BruteForceIndex bf(dim, MetricType::COSINE);
    for (uint32_t i = 0; i < n; ++i) {
        const auto& c = centers[i % centers.size()];
        std::vector<float> v(dim);
        for (uint32_t d = 0; d < dim; ++d) {
            v[d] = c[d] + noise(rng);
            flat[i * dim + d] = v[d];
        }
        bf.upsert(i, v);
    }

    ProductQuantizer pq(dim, MetricType::COSINE, 16, 256);
    pq.train(flat.data(), n, 15, 99);
    ASSERT_TRUE(pq.is_trained());

    auto codes = pq.encode_batch(flat.data(), n);
    ASSERT_EQ(codes.size(), static_cast<size_t>(n) * pq.subspaces());

    double adc_recall_sum = 0.0;
    double rerank_recall_sum = 0.0;
    constexpr uint32_t num_queries = 40;
    std::vector<float> table;

    for (uint32_t q = 0; q < num_queries; ++q) {
        const auto& c = centers[q % centers.size()];
        std::vector<float> query(dim);
        for (uint32_t d = 0; d < dim; ++d) query[d] = c[d] + noise(rng);

        auto truth = bf.search(query, k);
        std::unordered_set<VectorId> truth_ids;
        for (const auto& t : truth) truth_ids.insert(t.id);

        pq.build_adc_table(query.data(), table);
        std::vector<std::pair<float, uint32_t>> approx(n);
        for (uint32_t i = 0; i < n; ++i) {
            approx[i] = {pq.adc_distance(&codes[static_cast<size_t>(i) * pq.subspaces()], table.data()), i};
        }
        std::partial_sort(approx.begin(), approx.begin() + 60, approx.end());

        uint32_t adc_hits = 0;
        for (uint32_t i = 0; i < k; ++i) {
            if (truth_ids.count(approx[i].second)) ++adc_hits;
        }
        adc_recall_sum += static_cast<double>(adc_hits) / k;

        // Rerank top-60 with exact vectors
        std::vector<std::pair<float, uint32_t>> reranked;
        reranked.reserve(60);
        for (uint32_t i = 0; i < 60; ++i) {
            uint32_t idx = approx[i].second;
            reranked.push_back({compute_distance(query.data(), &flat[idx * dim], dim, MetricType::COSINE), idx});
        }
        std::sort(reranked.begin(), reranked.end());
        uint32_t rr_hits = 0;
        for (uint32_t i = 0; i < k; ++i) {
            if (truth_ids.count(reranked[i].second)) ++rr_hits;
        }
        rerank_recall_sum += static_cast<double>(rr_hits) / k;
    }

    EXPECT_GT(adc_recall_sum / num_queries, 0.70);
    EXPECT_GT(rerank_recall_sum / num_queries, 0.95);

    const std::string cb_path = "/tmp/test_pq_cb.bin";
    ASSERT_TRUE(pq.save(cb_path));
    ProductQuantizer pq2;
    ASSERT_TRUE(pq2.load(cb_path));
    EXPECT_EQ(pq2.dim(), pq.dim());
    EXPECT_EQ(pq2.subspaces(), pq.subspaces());
    std::vector<uint8_t> c1(pq.subspaces()), c2(pq2.subspaces());
    pq.encode(flat.data(), c1.data());
    pq2.encode(flat.data(), c2.data());
    EXPECT_EQ(c1, c2);
    std::remove(cb_path.c_str());
}
