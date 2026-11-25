# Adaptive Vector Database Engine (WAHI)

A production-grade, persistent, memory-bounded high-dimensional vector database engine built from scratch in **C++20**, paired with a **Node.js** REST API & selectivity-aware hybrid query optimizer, **PostgreSQL** metadata store, **Prometheus + Grafana** observability, and **Python** benchmarking/crash-recovery suites — containerized end-to-end with **Docker Compose**.

## System Architecture

```text
Client / Benchmark Harness
         │
         ▼
Node.js API & Hybrid Query Optimizer (port 3000)
  ├── Metadata Store: PostgreSQL 16 (port 5432, B-Tree + GIN JSONB/Array indexes)
  │     └── Selectivity estimation -> Pre-filter Allowlist vs Adaptive Post-filter
  └── Persistent HTTP Pool ──► C++20 Core Engine Server (port 8080, mem_limit 512m)
                                 ├── Write-Ahead Log (CRC32 checksums, torn-tail recovery, fsync)
                                 ├── Segments (Mutable Memtable + Sealed Disk Segments)
                                 ├── HNSW (Heuristic neighbor selection, tombstones, filtered traversal)
                                 ├── Product Quantization (k-means++ codebooks, ADC LUT, disk rerank)
                                 ├── POSIX mmap (MADV_RANDOM / MADV_DONTNEED) + Bounded LRU Vector Cache
                                 └── WAHI Adaptive Controller (3-tier utility-per-byte knapsack + hysteresis)
                                       ├── Tier 1: HOT_RAM_HNSW   (vectors + graph in RAM)
                                       ├── Tier 2: WARM_MMAP_HNSW (graph in RAM, vectors via POSIX mmap)
                                       └── Tier 3: COLD_DISK_PQ   (PQ codes in RAM, raw vectors via pread + LRU)

Observability: Prometheus (port 9090) -> Grafana Dashboard (port 3001)
```

## Workload-Adaptive Hybrid Indexing (WAHI)

Related prior systems such as **DiskANN/Vamana** (static SSD graph + in-RAM PQ) and **SPANN** (static disk posting lists + head index) rely on fixed index placements chosen ahead of time. **WAHI** (`engine/src/wahi_controller.cpp`) adapts segment index/storage placement online under a strict RAM budget:

1. **Telemetry Collection**: Tracks per-segment exponentially weighted moving averages (`useful_hits_ewma` from actual global top-$K$ contributions, `scan_freq_ewma`, and `write_freq_ewma`), distinguishing genuinely hot semantic segments from segments merely touched by full-fanout scans.
2. **Utility-per-Byte Scoring**: Ranks each sealed segment by workload benefit per saved megabyte of RAM (`HOT` vs `COLD`), adding hysteresis (`1.15x`) and cooldown ticks to prevent thrashing.
3. **3-Tier Budgeted Knapsack**: Allocates top-utility segments to `HOT_RAM_HNSW`, middle-tier segments to `WARM_MMAP_HNSW` (preserving HNSW navigability without pinning raw fp32 vectors in heap), and cold segments to `COLD_DISK_PQ` while scaling the bounded disk vector cache inside the budget.

## Verified Benchmark & Test Results

All tests and benchmarks were executed inside Docker (`Ubuntu 24.04`, `g++ 13.3.0 -O3 -std=c++20`, `Node 20`, `PostgreSQL 16`).

### 1. Unit & Concurrency Tests (`ctest` & `node --test`)
- C++20 Engine (`engine`): **6/6 suites passed (`100%`)** — `test_distance_and_bf`, `test_hnsw`, `test_wal_and_mmap`, `test_pq`, `test_wahi`, `test_concurrency`.
- Node.js API (`api`): **3/3 tests passed** — parameterized SQL filter builder, tight-selectivity pre-filter plan, broad-selectivity adaptive post-filter plan.

### 2. Core Algorithm & WAHI Benchmarks (`vectordb_bench`, `N = 18,000`, `dim = 128`, `60` clusters, Cosine)

| Strategy / Index | Recall@10 | p50 (ms) | p95 (ms) | QPS | Tracked RAM |
|---|---:|---:|---:|---:|---:|
| BruteForce | 1.0000 | 1.742 | 2.326 | 555.4 | 17.02 MB |
| HNSW (`ef=32`) | 0.9633 | 0.069 | 0.123 | 12,949.1 | 21.28 MB |
| HNSW (`ef=64`) | 0.9660 | 0.080 | 0.131 | 11,101.3 | 21.28 MB |
| HNSW (`ef=128`) | 0.9667 | 0.102 | 0.138 | 9,157.9 | 21.28 MB |
| PQ (`ADC only`) | 0.2273 | 0.386 | 0.537 | 2,430.0 | 0.67 MB |
| PQ (`ExactRerank 8x`) | 0.7560 | 0.446 | 0.715 | 2,080.2 | 0.67 MB |

#### Constrained RAM (`5.18 MB` budget) — 85/15 Skewed Workload (`12` segments)

| Policy | Recall@10 | p95 (ms) | QPS | RAM Used | Tiers (H/W/C) | Disk Reads |
|---|---:|---:|---:|---:|---:|---:|
| `static_all_ram` (unbounded) | 0.9999 | 1.258 | 1,110.9 | 16.27 MB | 12 / 0 / 0 | 0 |
| `static_all_disk_pq` | 0.6594 | 2.275 | 631.0 | 1.91 MB | 0 / 0 / 12 | 503,614 |
| `static_lru` (hit-recency) | 0.8232 | 1.653 | 730.2 | 4.53 MB | 3 / 0 / 9 | 377,520 |
| **`wahi_adaptive` (Ours)** | **0.9544** | **1.194** | **985.9** | **5.12 MB** | **3 / 4 / 5** | **217,101** |

#### Non-Stationary Shifting Hotspots (`3` epochs, `5.18 MB` budget)

| Policy | Overall Recall@10 | Adapted Recall (Ep 1 / 2 / 3) | p95 (ms) | Prom / Dem | Disk Reads |
|---|---:|---|---:|---:|---:|
| `static_all_disk_pq` | 0.6153 | 0.6924 / 0.5844 / 0.5560 | 2.301 | 0 / 12 | 943,544 |
| `static_lru` | 0.9161 | 0.9320 / 0.9476 / 0.9604 | 1.576 | 32 / 41 | 707,015 |
| **`wahi_adaptive`** | **0.9465** | **0.9551 / 0.9644 / 0.9764** | **1.820** | **26 / 34** | **410,993** |

### 3. End-to-End Docker Stack & Crash Recovery
- **Ingestion (`4,000` vectors + Postgres metadata via Node.js batch API)**: `2.556s` (`1,564.9 vec/s`)
- **HTTP WAHI hotspot shift**: `1.0000` adapted Recall@10 across Phase A and Phase B (`p50 ~5.1ms`, `p95 ~7.2ms`)
- **Hybrid Query Optimizer**: `prefilter_allowlist` selected at `0.75%` selectivity; `adaptive_postfilter` selected at `48.2%` selectivity (`5.40 ms`)
- **SIGKILL WAL Crash Recovery**: `docker kill -s KILL vectordb-engine` followed by restart recovered 100% of unsealed WAL upserts and deletes

## Quick Verification & Live Endpoints

Start the full stack:
```bash
docker compose up -d --build
```

Run the end-to-end benchmark & crash recovery tests:
```bash
docker compose --profile bench run --rm benchmark
python benchmarks/crash_recovery_test.py
```

### Live URLs
- **Grafana WAHI Dashboard**: http://localhost:3001/d/wahi-vectordb-overview/adaptive-vectordb-wahi-telemetry (anonymous view enabled; login `admin` / `admin`)
- **Prometheus UI / Targets**: http://localhost:9090/targets
- **Node.js API Health**: http://localhost:3000/health
- **Node.js API Live Stats & WAHI Tiers**: http://localhost:3000/api/v1/stats
- **Node.js API Metrics**: http://localhost:3000/metrics
- **C++20 Engine Health**: http://localhost:8080/health
- **C++20 Engine Live Stats**: http://localhost:8080/v1/stats
- **C++20 Engine Prometheus Metrics**: http://localhost:8080/metrics
