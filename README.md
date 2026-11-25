# Adaptive Vector Database Engine

A persistent vector database engine written from scratch in C++20, paired with a Node.js API server, PostgreSQL metadata filtering, Prometheus/Grafana monitoring, and Python evaluation scripts. Everything runs inside Docker.

## Why I built this

Most vector search prototypes just load vectors into RAM and run cosine similarity or HNSW. That breaks down quickly once you hit real constraints:

- Embeddings take a lot of memory (millions of 128d–1536d vectors quickly exceed RAM).
- Static indexing parameters assume the workload stays the same, whereas in practice some topics/clusters get queried thousands of times while older data goes cold.
- Continuous inserts, updates, and deletes degrade graph indexes if tombstones pile up or crash recovery isn't handled carefully.
- Combining vector similarity with SQL metadata filters (category, price range, tags, JSON fields) requires choosing between pre-filtering and post-filtering depending on how selective the filter is.

This project implements the storage engine, index structures, hybrid filtering pipeline, and an adaptive tiering strategy called **Workload-Adaptive Hybrid Indexing (WAHI)** to test whether shifting hot and cold segments dynamically beats static configurations under a tight RAM limit.

---

## How the system is structured

1. **C++20 Core Engine (`engine/`)**
   - **Distance kernels (`distance.cpp`)**: L2 squared, cosine distance, and inner product with loop-unrolled float accumulation.
   - **Brute-force baseline (`brute_force.cpp`)**: Exact search baseline used to measure Recall@K ground truth.
   - **HNSW (`hnsw.cpp`)**: Multi-layer navigable small world graph with heuristic neighbor selection, soft-delete tombstones (nodes stay traversable as routing hops until compacted), filtered traversal, and binary graph serialization.
   - **Product Quantization (`pq.cpp`)**: Subspace k-means++ codebook training, 8-bit encoding, Asymmetric Distance Computation (ADC) lookup tables, and optional exact reranking.
   - **Storage & Crash Recovery (`wal.cpp`, `mmap_storage.cpp`, `segment.cpp`)**:
     - Append-only Write-Ahead Log with CRC32 checksums on every record so incomplete tail writes from a crash or `SIGKILL` are detected and truncated cleanly on restart.
     - Segmented storage: writes land in a mutable memtable, get flushed/sealed to disk segments, and tombstoned segments are compacted once deleted slots cross the configured threshold.
     - POSIX `mmap` (`POSIX_MADV_RANDOM` / `POSIX_MADV_DONTNEED`) for warm segments, plus `pread()` backed by a bounded LRU vector cache for cold reranking.
   - **HTTP Server & Metrics (`server_main.cpp`, `metrics.cpp`)**: Multi-threaded POSIX socket HTTP/1.1 keep-alive server exposing vector CRUD, search, WAHI controls, and `/metrics` for Prometheus.

2. **Node.js API & Hybrid Query Optimizer (`api/`)**
   - Stores document metadata (`category`, `price`, `tags`, `metadata` JSONB) in **PostgreSQL 16** with B-tree and GIN indexes.
   - Estimates filter selectivity on incoming search queries:
     - **Tight filters** (low matching row count or selectivity <= 8%) fetch matching IDs from PostgreSQL and pass them as an allowlist straight into the C++ engine (`prefilter_allowlist`).
     - **Broad filters** expand the ANN candidate count `k` proportionally to `1 / selectivity`, intersect results against PostgreSQL (`adaptive_postfilter`), and fall back to pre-filtering if too few results survive.

3. **Observability (`observability/`)**
   - **Prometheus** scrapes both the C++ engine (`vectordb-engine:8080/metrics`) and the Node.js API (`api-server:3000/metrics`).
   - **Grafana** ships pre-configured with a dashboard tracking segment tier counts, RAM vs budget, p50/p95/p99 latency, WAHI promotions/demotions, disk reads, and query optimizer plans.

---

## Workload-Adaptive Hybrid Indexing (WAHI)

Systems like DiskANN/Vamana (SSD graph + in-RAM PQ) or SPANN (disk posting lists + RAM head index) pick one static layout upfront. In contrast, WAHI (`engine/src/wahi_controller.cpp`) moves sealed segments across three tiers at runtime while staying inside a fixed RAM budget:

- **Hot (`HOT_RAM_HNSW`)**: Full HNSW graph and raw float32 vectors in RAM. Fastest search, zero disk reads, highest memory cost.
- **Warm (`WARM_MMAP_HNSW`)**: HNSW graph stays in memory while raw vectors are read from disk via POSIX `mmap` instead of duplicating them in the heap.
- **Cold (`COLD_DISK_PQ`)**: HNSW graph and raw vectors are evicted from memory; only compact PQ codes stay in RAM, and top candidates are optionally reranked from disk via `pread()` and the LRU cache.

Instead of tracking plain scan counts (which look identical across segments whenever a query checks all segments), WAHI tracks exponentially decaying **useful hits** (segments that actually produce final top-K results) alongside scan and write frequency, scores each segment by utility per megabyte saved, and applies hysteresis + cooldowns so shifting workloads adapt without thrashing.

---

## Benchmarks and test results

Everything below was measured inside Docker (`Ubuntu 24.04`, `g++ 13.3.0 -O3 -std=c++20`, `Node 20`, `PostgreSQL 16`).

### Unit and concurrency tests
- **C++20 engine (`ctest`)**: 6/6 suites passed (`test_distance_and_bf`, `test_hnsw`, `test_wal_and_mmap`, `test_pq`, `test_wahi`, `test_concurrency`).
- **Node.js API (`node --test`)**: 3/3 tests passed covering SQL filter building, pre-filter selection, and post-filter selection.

### Core indexing comparison (`vectordb_bench`: 18,000 vectors, 128d, 60 clusters, Cosine)

| Index / Config | Recall@10 | p50 (ms) | p95 (ms) | QPS | RAM |
|---|---:|---:|---:|---:|---:|
| BruteForce | 1.0000 | 1.742 | 2.326 | 555.4 | 17.02 MB |
| HNSW (`ef=32`) | 0.9633 | 0.069 | 0.123 | 12,949.1 | 21.28 MB |
| HNSW (`ef=64`) | 0.9660 | 0.080 | 0.131 | 11,101.3 | 21.28 MB |
| HNSW (`ef=128`) | 0.9667 | 0.102 | 0.138 | 9,157.9 | 21.28 MB |
| PQ (`ADC only`) | 0.2273 | 0.386 | 0.537 | 2,430.0 | 0.67 MB |
| PQ (`ExactRerank 8x`) | 0.7560 | 0.446 | 0.715 | 2,080.2 | 0.67 MB |

### Tight RAM budget (`5.18 MB` cap) on an 85/15 skewed workload (12 segments)

| Policy | Recall@10 | p95 (ms) | QPS | RAM Used | Tiers (Hot / Warm / Cold) | Disk Reads |
|---|---:|---:|---:|---:|---:|---:|
| `static_all_ram` (unbounded) | 0.9999 | 1.258 | 1,110.9 | 16.27 MB | 12 / 0 / 0 | 0 |
| `static_all_disk_pq` | 0.6594 | 2.275 | 631.0 | 1.91 MB | 0 / 0 / 12 | 503,614 |
| `static_lru` (hit-recency) | 0.8232 | 1.653 | 730.2 | 4.53 MB | 3 / 0 / 9 | 377,520 |
| **`wahi_adaptive`** | **0.9544** | **1.194** | **985.9** | **5.12 MB** | **3 / 4 / 5** | **217,101** |

### Shifting hotspot workload (3 epochs under `5.18 MB` cap)

| Policy | Overall Recall@10 | Adapted Recall (Ep 1 / 2 / 3) | p95 (ms) | Promotions / Demotions | Disk Reads |
|---|---:|---:|---:|---:|---:|
| `static_all_disk_pq` | 0.6153 | 0.6924 / 0.5844 / 0.5560 | 2.301 | 0 / 12 | 943,544 |
| `static_lru` | 0.9161 | 0.9320 / 0.9476 / 0.9604 | 1.576 | 32 / 41 | 707,015 |
| **`wahi_adaptive`** | **0.9465** | **0.9551 / 0.9644 / 0.9764** | **1.820** | **26 / 34** | **410,993** |

### End-to-end Docker stack & crash recovery
- **Batch ingestion (4,000 vectors + PostgreSQL metadata)**: 2.556s (1,564.9 vec/s).
- **Hybrid optimizer over HTTP**: chose `prefilter_allowlist` at 0.75% selectivity (26.16 ms) and `adaptive_postfilter` at 48.18% selectivity (5.40 ms).
- **Crash recovery**: killed `vectordb-engine` with `SIGKILL` after writing unsealed records and a delete to the WAL; on restart, all surviving inserts and the delete replayed cleanly with zero data loss.

---

## Running and verifying locally

Start the stack (PostgreSQL, C++20 engine, Node.js API, Prometheus, Grafana):

```bash
docker compose up -d --build
```

Run the C++20 tests inside the running engine container:

```powershell
docker exec vectordb-engine sh -c 'for t in /app/tests/test_*; do "$t" || exit 1; done'
```

Run the Node.js API tests:

```powershell
docker exec vectordb-api node --test tests/api.test.js
```

Run the end-to-end Python benchmark and `SIGKILL` crash-recovery test:

```bash
docker compose --profile bench run --rm benchmark
python benchmarks/crash_recovery_test.py
```

### Local links once running
- **Grafana dashboard**: http://localhost:3001/d/wahi-vectordb-overview/adaptive-vectordb-wahi-telemetry
- **Prometheus targets**: http://localhost:9090/targets
- **Node.js API health / stats / metrics**: http://localhost:3000/health | http://localhost:3000/api/v1/stats | http://localhost:3000/metrics
- **C++20 engine health / stats / metrics**: http://localhost:8080/health | http://localhost:8080/v1/stats | http://localhost:8080/metrics
