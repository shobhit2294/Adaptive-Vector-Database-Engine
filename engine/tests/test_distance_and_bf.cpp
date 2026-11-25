#include <gtest/gtest.h>
#include "vectordb/brute_force.hpp"
#include "vectordb/distance.hpp"
#include <cmath>
#include <random>

using namespace vectordb;

TEST(DistanceTest, L2AndDotAndCosineExact) {
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {2.0f, 4.0f, 6.0f, 8.0f};
    float c[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
    float z[4] = {0.0f, 0.0f, 0.0f, 0.0f};

    EXPECT_NEAR(l2_squared(a, b, 4), 1.0f + 4.0f + 9.0f + 16.0f, 1e-5f);
    EXPECT_NEAR(dot_product(a, b, 4), 60.0f, 1e-5f);
    EXPECT_NEAR(cosine_distance(a, b, 4), 0.0f, 1e-5f);
    EXPECT_NEAR(cosine_distance(a, c, 4), 2.0f, 1e-5f);
    EXPECT_NEAR(cosine_distance(a, z, 4), 1.0f, 1e-5f);
}

TEST(DistanceTest, HighDimensionalUnrolled) {
    constexpr uint32_t dim = 131; // non-multiple of 8
    std::vector<float> a(dim), b(dim);
    for (uint32_t i = 0; i < dim; ++i) {
        a[i] = static_cast<float>(i % 7) * 0.25f;
        b[i] = static_cast<float>((i + 3) % 5) * 0.3f;
    }
    float expected_l2 = 0.0f, expected_dot = 0.0f;
    for (uint32_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        expected_l2 += d * d;
        expected_dot += a[i] * b[i];
    }
    EXPECT_NEAR(l2_squared(a.data(), b.data(), dim), expected_l2, 1e-3f);
    EXPECT_NEAR(dot_product(a.data(), b.data(), dim), expected_dot, 1e-3f);
}

TEST(BruteForceTest, UpsertDeleteAndSearch) {
    BruteForceIndex idx(4, MetricType::L2);
    idx.upsert(10, {0.0f, 0.0f, 0.0f, 0.0f});
    idx.upsert(20, {1.0f, 0.0f, 0.0f, 0.0f});
    idx.upsert(30, {2.0f, 0.0f, 0.0f, 0.0f});
    idx.upsert(40, {10.0f, 0.0f, 0.0f, 0.0f});

    auto res = idx.search({0.1f, 0.0f, 0.0f, 0.0f}, 2);
    ASSERT_EQ(res.size(), 2u);
    EXPECT_EQ(res[0].id, 10u);
    EXPECT_EQ(res[1].id, 20u);

    // Update id 40 to be nearest
    idx.upsert(40, {0.05f, 0.0f, 0.0f, 0.0f});
    res = idx.search({0.0f, 0.0f, 0.0f, 0.0f}, 2);
    EXPECT_EQ(res[0].id, 10u);
    EXPECT_EQ(res[1].id, 40u);

    // Delete id 10
    EXPECT_TRUE(idx.remove(10));
    EXPECT_FALSE(idx.remove(10));
    EXPECT_EQ(idx.size(), 3u);

    res = idx.search({0.0f, 0.0f, 0.0f, 0.0f}, 2);
    EXPECT_EQ(res[0].id, 40u);
    EXPECT_EQ(res[1].id, 20u);

    // Filtered search
    auto fres = idx.search({0.0f, 0.0f, 0.0f, 0.0f}, 2, [](VectorId id) { return id >= 25; });
    ASSERT_EQ(fres.size(), 2u);
    EXPECT_EQ(fres[0].id, 40u);
    EXPECT_EQ(fres[1].id, 30u);
}
