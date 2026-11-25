#include "vectordb/brute_force.hpp"
#include "vectordb/database.hpp"
#include "vectordb/distance.hpp"
#include "vectordb/hnsw.hpp"
#include "vectordb/pq.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

using namespace vectordb;
using json = nlohmann::json;

namespace {

double compute_recall(const std::vector<SearchResult>& approx,
                      const std::vector<SearchResult>& truth,
                      uint32_t k) {
    std::unordered_set<VectorId> gt;
    for (size_t i = 0; i < std::min<size_t>(k, truth.size()); ++i) {
        gt.insert(truth[i].id);
    }
    if (gt.empty()) return 1.0;
    uint32_t hits = 0;
    for (size_t i = 0; i < std::min<size_t>(k, approx.size()); ++i) {
        if (gt.count(approx[i].id)) ++hits;
    }
    return static_cast<double>(hits) / static_cast<double>(gt.size());
}

struct LatencyStats {
    double p50_ms = 0.0;
    double p95_ms = 0.0;
    double p99_ms = 0.0;
    double avg_ms = 0.0;
    double qps = 0.0;
};

LatencyStats summarize_latencies(std::vector<double>& lats_ms, double wall_ms) {
    LatencyStats s;
    if (lats_ms.empty()) return s;
    std::sort(lats_ms.begin(), lats_ms.end());
    auto pct = [&](double p) {
        size_t idx = static_cast<size_t>(p * static_cast<double>(lats_ms.size() - 1));
        return lats_ms[std::min(idx, lats_ms.size() - 1)];
    };
    s.p50_ms = pct(0.50);
    s.p95_ms = pct(0.95);
    s.p99_ms = pct(0.99);
    double sum = std::accumulate(lats_ms.begin(), lats_ms.end(), 0.0);
    s.avg_ms = sum / static_cast<double>(lats_ms.size());
    s.qps = (wall_ms > 0.0) ? (static_cast<double>(lats_ms.size()) / (wall_ms / 1000.0)) : 0.0;
    return s;
}

struct ClusteredDataset {
    uint32_t dim;
    uint32_t num_clusters;
    std::vector<std::vector<float>> centers;
    std::vector<std::vector<float>> vectors;
    std::vector<uint32_t> cluster_of;
};

ClusteredDataset make_dataset(uint32_t n_per_cluster, uint32_t num_clusters, uint32_t dim, uint64_t seed) {
    ClusteredDataset ds;
    ds.dim = dim;
    ds.num_clusters = num_clusters;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> u(-1.0f, 1.0f);
    std::normal_distribution<float> noise(0.0f, 0.20f);

    ds.centers.resize(num_clusters, std::vector<float>(dim));
    for (uint32_t c = 0; c < num_clusters; ++c) {
        for (uint32_t d = 0; d < dim; ++d) ds.centers[c][d] = u(rng) * 0.5f;
        uint32_t b1 = (c * 3) % dim;
        uint32_t b2 = (c * 7 + 13) % dim;
        for (uint32_t j = 0; j < 4; ++j) {
            ds.centers[c][(b1 + j) % dim] += (c % 2 == 0 ? 2.0f : -2.0f);
            ds.centers[c][(b2 + j) % dim] += ((c / 2) % 2 == 0 ? 1.5f : -1.5f);
        }
    }

    uint32_t total = n_per_cluster * num_clusters;
    ds.vectors.reserve(total);
    ds.cluster_of.reserve(total);
    for (uint32_t c = 0; c < num_clusters; ++c) {
        for (uint32_t i = 0; i < n_per_cluster; ++i) {
            std::vector<float> v(dim);
            for (uint32_t d = 0; d < dim; ++d) {
                v[d] = ds.centers[c][d] + noise(rng);
            }
            ds.vectors.push_back(std::move(v));
            ds.cluster_of.push_back(c);
        }
    }
    return ds;
}

std::vector<float> sample_query_from_cluster(const ClusteredDataset& ds, uint32_t cluster, std::mt19937& rng) {
    std::normal_distribution<float> noise(0.0f, 0.20f);
    std::vector<float> q(ds.dim);
    for (uint32_t d = 0; d < ds.dim; ++d) {
        q[d] = ds.centers[cluster][d] + noise(rng);
    }
    return q;
}

json run_experiment_1_algorithms(const ClusteredDataset& ds) {
    constexpr uint32_t k = 10;
    constexpr uint32_t num_queries = 150;
    const uint32_t n = static_cast<uint32_t>(ds.vectors.size());

    BruteForceIndex bf(ds.dim, MetricType::COSINE);
    HnswIndex hnsw(ds.dim, MetricType::COSINE, 16, 128, 64);
    std::vector<float> flat(static_cast<size_t>(n) * ds.dim);

    for (uint32_t i = 0; i < n; ++i) {
        bf.upsert(i + 1, ds.vectors[i]);
        hnsw.upsert(i + 1, ds.vectors[i]);
        std::copy(ds.vectors[i].begin(), ds.vectors[i].end(), flat.begin() + static_cast<size_t>(i) * ds.dim);
    }

    ProductQuantizer pq(ds.dim, MetricType::COSINE, 32, 256);
    pq.train(flat.data(), n, 12);
    auto pq_codes = pq.encode_batch(flat.data(), n);

    std::mt19937 qrng(999);
    std::vector<std::vector<float>> queries(num_queries);
    std::vector<std::vector<SearchResult>> truths(num_queries);

    std::vector<double> bf_lats;
    auto t0_bf = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < num_queries; ++i) {
        queries[i] = sample_query_from_cluster(ds, i % ds.num_clusters, qrng);
        auto qa = std::chrono::steady_clock::now();
        truths[i] = bf.search(queries[i], k);
        auto qb = std::chrono::steady_clock::now();
        bf_lats.push_back(std::chrono::duration<double, std::milli>(qb - qa).count());
    }
    auto t1_bf = std::chrono::steady_clock::now();
    auto bf_stat = summarize_latencies(bf_lats, std::chrono::duration<double, std::milli>(t1_bf - t0_bf).count());

    json results = json::array();
    results.push_back({
        {"name", "BruteForce"},
        {"recall_at_10", 1.0},
        {"p50_ms", bf_stat.p50_ms},
        {"p95_ms", bf_stat.p95_ms},
        {"qps", bf_stat.qps},
        {"ram_mb", static_cast<double>(bf.ram_bytes()) / (1024.0 * 1024.0)}
    });

    for (uint32_t ef : {32u, 64u, 128u}) {
        std::vector<double> lats;
        double rec_sum = 0.0;
        auto ta = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < num_queries; ++i) {
            auto qa = std::chrono::steady_clock::now();
            auto res = hnsw.search(queries[i], k, ef);
            auto qb = std::chrono::steady_clock::now();
            lats.push_back(std::chrono::duration<double, std::milli>(qb - qa).count());
            rec_sum += compute_recall(res, truths[i], k);
        }
        auto tb = std::chrono::steady_clock::now();
        auto st = summarize_latencies(lats, std::chrono::duration<double, std::milli>(tb - ta).count());
        results.push_back({
            {"name", "HNSW(ef=" + std::to_string(ef) + ")"},
            {"recall_at_10", rec_sum / num_queries},
            {"p50_ms", st.p50_ms},
            {"p95_ms", st.p95_ms},
            {"qps", st.qps},
            {"ram_mb", static_cast<double>(hnsw.total_ram_bytes()) / (1024.0 * 1024.0)}
        });
    }

    for (uint32_t rerank_mult : {0u, 8u}) {
        std::vector<double> lats;
        double rec_sum = 0.0;
        std::vector<float> adc_table;
        auto ta = std::chrono::steady_clock::now();
        for (uint32_t i = 0; i < num_queries; ++i) {
            auto qa = std::chrono::steady_clock::now();
            pq.build_adc_table(queries[i].data(), adc_table);
            uint32_t cand_k = rerank_mult > 0 ? k * rerank_mult : k;
            std::vector<std::pair<float, uint32_t>> approx(n);
            for (uint32_t j = 0; j < n; ++j) {
                approx[j] = {pq.adc_distance(&pq_codes[static_cast<size_t>(j) * pq.subspaces()], adc_table.data()), j};
            }
            std::partial_sort(approx.begin(), approx.begin() + cand_k, approx.end());

            std::vector<SearchResult> res;
            if (rerank_mult == 0) {
                for (uint32_t j = 0; j < k; ++j) {
                    res.push_back(SearchResult{approx[j].second + 1, approx[j].first, 0.0f, SegmentTier::COLD_DISK_PQ});
                }
            } else {
                std::vector<SearchResult> rr;
                rr.reserve(cand_k);
                for (uint32_t j = 0; j < cand_k; ++j) {
                    uint32_t idx = approx[j].second;
                    float d = cosine_distance(queries[i].data(), &flat[static_cast<size_t>(idx) * ds.dim], ds.dim);
                    rr.push_back(SearchResult{idx + 1, d, 0.0f, SegmentTier::COLD_DISK_PQ});
                }
                std::sort(rr.begin(), rr.end(), [](const SearchResult& a, const SearchResult& b) {
                    return a.distance < b.distance;
                });
                rr.resize(k);
                res = std::move(rr);
            }
            auto qb = std::chrono::steady_clock::now();
            lats.push_back(std::chrono::duration<double, std::milli>(qb - qa).count());
            rec_sum += compute_recall(res, truths[i], k);
        }
        auto tb = std::chrono::steady_clock::now();
        auto st = summarize_latencies(lats, std::chrono::duration<double, std::milli>(tb - ta).count());
        double pq_ram_mb = static_cast<double>(pq_codes.size() + pq.codebook_ram_bytes()) / (1024.0 * 1024.0);
        results.push_back({
            {"name", rerank_mult == 0 ? "PQ_ADC_Only" : "PQ_ExactRerank_8x"},
            {"recall_at_10", rec_sum / num_queries},
            {"p50_ms", st.p50_ms},
            {"p95_ms", st.p95_ms},
            {"qps", st.qps},
            {"ram_mb", pq_ram_mb}
        });
    }

    return results;
}

json run_experiment_2_skewed_policies(const ClusteredDataset& ds, const BruteForceIndex& gt) {
    constexpr uint32_t k = 10;
    constexpr uint32_t seg_cap = 1500;
    constexpr uint32_t num_queries = 900;

    // 60 clusters * 300 vecs = 18,000 vecs across 12 segments (clusters 0..14 reside in segments 1..3)
    std::mt19937 qrng(2026);
    std::vector<std::vector<float>> queries;
    std::vector<std::vector<SearchResult>> truths;
    queries.reserve(num_queries);
    truths.reserve(num_queries);

    // 85% of queries target hot clusters {0..14}; 15% target cold clusters {15..59}
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    for (uint32_t i = 0; i < num_queries; ++i) {
        uint32_t cl = (u01(qrng) < 0.85) ? (i % 15) : (15 + (i % (ds.num_clusters - 15)));
        auto q = sample_query_from_cluster(ds, cl, qrng);
        truths.push_back(gt.search(q, k));
        queries.push_back(std::move(q));
    }

    // At dim=128, full All-RAM HNSW is ~14.7 MB; constrain tight budget to ~5.2 MB (~35%!)
    constexpr uint64_t tight_budget_bytes = 5300ULL * 1024ULL;

    json out = json::array();
    std::vector<WahiPolicyMode> modes = {
        WahiPolicyMode::STATIC_ALL_RAM,
        WahiPolicyMode::STATIC_ALL_DISK_PQ,
        WahiPolicyMode::STATIC_LRU,
        WahiPolicyMode::WAHI_ADAPTIVE
    };

    for (auto mode : modes) {
        std::string dir = "/tmp/bench_exp2_" + std::string(to_string(mode));
        std::filesystem::remove_all(dir);

        DatabaseConfig cfg;
        cfg.data_dir = dir;
        cfg.dimension = ds.dim;
        cfg.metric = MetricType::COSINE;
        cfg.segment_capacity = seg_cap;
        cfg.hnsw_m = 16;
        cfg.hnsw_ef_construction = 96;
        cfg.hnsw_ef_search = 64;
        cfg.pq_subspaces = 32;
        cfg.pq_centroids = 256;
        cfg.cold_rerank_factor = 6;
        cfg.disk_cache_bytes = 512 * 1024;
        cfg.ram_budget_bytes = (mode == WahiPolicyMode::STATIC_ALL_RAM) ? (128ULL * 1024ULL * 1024ULL) : tight_budget_bytes;
        cfg.wahi_mode = mode;
        cfg.wahi_reconcile_interval_queries = 50;
        cfg.wahi_cooldown_ticks = 1;
        cfg.wal_fsync = FsyncPolicy::NONE;

        VectorDatabase db(cfg);
        db.open();
        std::vector<std::pair<VectorId, std::vector<float>>> batch;
        batch.reserve(ds.vectors.size());
        for (size_t i = 0; i < ds.vectors.size(); ++i) {
            batch.emplace_back(i + 1, ds.vectors[i]);
        }
        db.upsert_batch(batch);
        db.flush_and_seal();
        db.trigger_wahi_reconcile();

        for (uint32_t i = 0; i < 180; ++i) {
            db.search(queries[i], SearchOptions{k, 64, 6, false});
        }
        db.trigger_wahi_reconcile();
        db.metrics().reset();

        std::vector<double> lats;
        lats.reserve(num_queries - 180);
        double recall_sum = 0.0;
        auto ta = std::chrono::steady_clock::now();
        for (uint32_t i = 180; i < num_queries; ++i) {
            SearchDiagnostics diag;
            auto res = db.search(queries[i], SearchOptions{k, 64, 6, false}, &diag);
            lats.push_back(diag.latency_ms);
            recall_sum += compute_recall(res, truths[i], k);
        }
        auto tb = std::chrono::steady_clock::now();
        auto st = summarize_latencies(lats, std::chrono::duration<double, std::milli>(tb - ta).count());
        auto snap = db.snapshot();

        out.push_back({
            {"policy", to_string(mode)},
            {"recall_at_10", recall_sum / static_cast<double>(num_queries - 180)},
            {"p50_ms", st.p50_ms},
            {"p95_ms", st.p95_ms},
            {"p99_ms", st.p99_ms},
            {"qps", st.qps},
            {"ram_mb", static_cast<double>(snap.ram_bytes_total) / (1024.0 * 1024.0)},
            {"ram_budget_mb", static_cast<double>(tight_budget_bytes) / (1024.0 * 1024.0)},
            {"hot_segments", snap.segments_hot_ram},
            {"warm_segments", snap.segments_warm_mmap},
            {"cold_segments", snap.segments_cold_disk_pq},
            {"disk_reads", db.metrics().disk_reads_total.load()},
            {"disk_bytes_read_mb", static_cast<double>(db.metrics().disk_bytes_read_total.load()) / (1024.0 * 1024.0)},
            {"promotions", db.metrics().wahi_promotions_total.load()},
            {"demotions", db.metrics().wahi_demotions_total.load()}
        });

        db.close();
        std::filesystem::remove_all(dir);
    }
    return out;
}

json run_experiment_3_shifting_workload(const ClusteredDataset& ds, const BruteForceIndex& gt) {
    constexpr uint32_t k = 10;
    constexpr uint32_t seg_cap = 1500;
    constexpr uint64_t tight_budget_bytes = 5300ULL * 1024ULL;
    constexpr uint32_t epoch_queries = 450;

    std::mt19937 qrng(777);
    std::uniform_real_distribution<double> u01(0.0, 1.0);
    struct EpochSpec {
        std::string name;
        uint32_t c_start;
        uint32_t c_end;
    };
    // 5 clusters per segment: Seg 1-2 = 0..9, Seg 6-7 = 25..34, Seg 11-12 = 50..59
    std::vector<EpochSpec> epochs = {
        {"epoch_1_seg_1_2", 0, 10},
        {"epoch_2_seg_6_7", 25, 35},
        {"epoch_3_seg_11_12", 50, 60}
    };

    struct QueryItem {
        std::vector<float> q;
        std::vector<SearchResult> truth;
        uint32_t epoch_idx;
    };
    std::vector<QueryItem> stream;
    for (uint32_t e = 0; e < epochs.size(); ++e) {
        uint32_t span = epochs[e].c_end - epochs[e].c_start;
        for (uint32_t i = 0; i < epoch_queries; ++i) {
            uint32_t cl = (u01(qrng) < 0.85)
                ? (epochs[e].c_start + (i % span))
                : (i % ds.num_clusters);
            auto q = sample_query_from_cluster(ds, cl, qrng);
            auto t = gt.search(q, k);
            stream.push_back({std::move(q), std::move(t), e});
        }
    }

    json out = json::array();
    for (auto mode : {WahiPolicyMode::STATIC_ALL_DISK_PQ, WahiPolicyMode::STATIC_LRU, WahiPolicyMode::WAHI_ADAPTIVE}) {
        std::string dir = "/tmp/bench_exp3_" + std::string(to_string(mode));
        std::filesystem::remove_all(dir);

        DatabaseConfig cfg;
        cfg.data_dir = dir;
        cfg.dimension = ds.dim;
        cfg.metric = MetricType::COSINE;
        cfg.segment_capacity = seg_cap;
        cfg.hnsw_m = 16;
        cfg.hnsw_ef_construction = 96;
        cfg.hnsw_ef_search = 64;
        cfg.pq_subspaces = 32;
        cfg.pq_centroids = 256;
        cfg.cold_rerank_factor = 6;
        cfg.disk_cache_bytes = 512 * 1024;
        cfg.ram_budget_bytes = tight_budget_bytes;
        cfg.wahi_mode = mode;
        cfg.wahi_reconcile_interval_queries = 50;
        cfg.wahi_cooldown_ticks = 1;
        cfg.wal_fsync = FsyncPolicy::NONE;

        VectorDatabase db(cfg);
        db.open();
        std::vector<std::pair<VectorId, std::vector<float>>> batch;
        for (size_t i = 0; i < ds.vectors.size(); ++i) {
            batch.emplace_back(i + 1, ds.vectors[i]);
        }
        db.upsert_batch(batch);
        db.flush_and_seal();
        db.trigger_wahi_reconcile();

        json epoch_metrics = json::array();
        double total_recall = 0.0;
        std::vector<double> all_lats;
        auto t_start = std::chrono::steady_clock::now();

        for (uint32_t e = 0; e < epochs.size(); ++e) {
            std::vector<double> e_lats;
            double e_rec_early = 0.0, e_rec_late = 0.0;
            uint32_t half = epoch_queries / 2;
            auto e0 = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < epoch_queries; ++i) {
                const auto& qi = stream[e * epoch_queries + i];
                SearchDiagnostics diag;
                auto res = db.search(qi.q, SearchOptions{k, 64, 6, false}, &diag);
                double r = compute_recall(res, qi.truth, k);
                total_recall += r;
                if (i < half) e_rec_early += r;
                else e_rec_late += r;
                e_lats.push_back(diag.latency_ms);
                all_lats.push_back(diag.latency_ms);
            }
            auto e1 = std::chrono::steady_clock::now();
            auto est = summarize_latencies(e_lats, std::chrono::duration<double, std::milli>(e1 - e0).count());
            auto snap = db.snapshot();
            epoch_metrics.push_back({
                {"epoch", epochs[e].name},
                {"early_recall_at_10", e_rec_early / half},
                {"adapted_recall_at_10", e_rec_late / (epoch_queries - half)},
                {"p95_ms", est.p95_ms},
                {"ram_mb", static_cast<double>(snap.ram_bytes_total) / (1024.0 * 1024.0)},
                {"hot_segments", snap.segments_hot_ram},
                {"warm_segments", snap.segments_warm_mmap},
                {"cold_segments", snap.segments_cold_disk_pq}
            });
        }
        auto t_end = std::chrono::steady_clock::now();
        auto total_st = summarize_latencies(all_lats, std::chrono::duration<double, std::milli>(t_end - t_start).count());
        auto final_snap = db.snapshot();

        out.push_back({
            {"policy", to_string(mode)},
            {"overall_recall_at_10", total_recall / static_cast<double>(stream.size())},
            {"p50_ms", total_st.p50_ms},
            {"p95_ms", total_st.p95_ms},
            {"qps", total_st.qps},
            {"final_ram_mb", static_cast<double>(final_snap.ram_bytes_total) / (1024.0 * 1024.0)},
            {"disk_reads", db.metrics().disk_reads_total.load()},
            {"promotions", db.metrics().wahi_promotions_total.load()},
            {"demotions", db.metrics().wahi_demotions_total.load()},
            {"epochs", epoch_metrics}
        });

        db.close();
        std::filesystem::remove_all(dir);
    }
    return out;
}

} // namespace

int main(int argc, char** argv) {
    std::string out_path;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--out") {
            out_path = argv[i + 1];
        }
    }

    constexpr uint32_t dim = 128;
    constexpr uint32_t num_clusters = 60;
    constexpr uint32_t n_per_cluster = 300; // 18,000 vectors total across 12 segments
    auto ds = make_dataset(n_per_cluster, num_clusters, dim, 42);

    BruteForceIndex gt(dim, MetricType::COSINE);
    for (size_t i = 0; i < ds.vectors.size(); ++i) {
        gt.upsert(i + 1, ds.vectors[i]);
    }

    json report;
    report["dataset"] = {
        {"vectors", ds.vectors.size()},
        {"dimension", dim},
        {"clusters", num_clusters},
        {"metric", "cosine"}
    };

    std::cout << "=== Experiment 1: BruteForce vs HNSW vs Product Quantization (N=18000, dim=128) ===\n";
    report["experiment_1_algorithms"] = run_experiment_1_algorithms(ds);
    for (const auto& r : report["experiment_1_algorithms"]) {
        std::cout << std::left << std::setw(22) << r["name"].get<std::string>()
                  << " | Recall@10=" << std::fixed << std::setprecision(4) << r["recall_at_10"].get<double>()
                  << " | p50=" << std::setprecision(3) << r["p50_ms"].get<double>() << "ms"
                  << " | p95=" << r["p95_ms"].get<double>() << "ms"
                  << " | QPS=" << std::setprecision(1) << r["qps"].get<double>()
                  << " | RAM=" << std::setprecision(2) << r["ram_mb"].get<double>() << "MB\n";
    }

    std::cout << "\n=== Experiment 2: Constrained RAM (5.18 MB budget) Skewed 85/15 Workload ===\n";
    report["experiment_2_skewed_policies"] = run_experiment_2_skewed_policies(ds, gt);
    for (const auto& r : report["experiment_2_skewed_policies"]) {
        std::cout << std::left << std::setw(22) << r["policy"].get<std::string>()
                  << " | Recall@10=" << std::fixed << std::setprecision(4) << r["recall_at_10"].get<double>()
                  << " | p95=" << std::setprecision(3) << r["p95_ms"].get<double>() << "ms"
                  << " | QPS=" << std::setprecision(1) << r["qps"].get<double>()
                  << " | RAM=" << std::setprecision(2) << r["ram_mb"].get<double>() << "MB"
                  << " | Tiers(H/W/C)=" << r["hot_segments"] << "/" << r["warm_segments"] << "/" << r["cold_segments"]
                  << " | DiskReads=" << r["disk_reads"] << "\n";
    }

    std::cout << "\n=== Experiment 3: Non-Stationary Shifting Hotspot Adaptation ===\n";
    report["experiment_3_shifting_workload"] = run_experiment_3_shifting_workload(ds, gt);
    for (const auto& r : report["experiment_3_shifting_workload"]) {
        std::cout << std::left << std::setw(22) << r["policy"].get<std::string>()
                  << " | OverallRecall@10=" << std::fixed << std::setprecision(4) << r["overall_recall_at_10"].get<double>()
                  << " | p95=" << std::setprecision(3) << r["p95_ms"].get<double>() << "ms"
                  << " | RAM=" << std::setprecision(2) << r["final_ram_mb"].get<double>() << "MB"
                  << " | Prom/Dem=" << r["promotions"] << "/" << r["demotions"]
                  << " | DiskReads=" << r["disk_reads"] << "\n";
        for (const auto& e : r["epochs"]) {
            std::cout << "    " << std::left << std::setw(24) << e["epoch"].get<std::string>()
                      << " early=" << std::setprecision(4) << e["early_recall_at_10"].get<double>()
                      << " adapted=" << e["adapted_recall_at_10"].get<double>()
                      << " p95=" << std::setprecision(3) << e["p95_ms"].get<double>() << "ms"
                      << " H/W/C=" << e["hot_segments"] << "/" << e["warm_segments"] << "/" << e["cold_segments"] << "\n";
        }
    }

    if (!out_path.empty()) {
        std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
        std::ofstream os(out_path, std::ios::trunc);
        os << report.dump(2);
    }
    return 0;
}
