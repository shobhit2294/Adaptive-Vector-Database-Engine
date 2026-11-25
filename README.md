# Adaptive Vector Database Engine

I built this project to solve a real problem most basic vector search demos skip: what happens when your embedding dataset grows larger than RAM, keeps changing with inserts, updates, and deletes, and needs SQL metadata filtering at the same time?

Instead of wrapping an existing vector database, I built the storage and indexing engine from scratch in C++20, put a Node.js API and query optimizer in front of it, used PostgreSQL for metadata filtering, wired up Prometheus and Grafana for monitoring, and ran the whole stack inside Docker Compose.

## Why I Built This

Keeping 100M vectors pinned in memory with HNSW is fast, but RAM gets expensive fast. Compressing everything onto disk keeps memory low, but recall drops and disk reads slow queries down.

Real workloads are uneven:
- A small fraction of vectors gets searched heavily right now while older data sits cold.
- Hot topics shift over time.
- Records get inserted, updated, and deleted continuously.
- Users combine similarity search with structured filters like category, price range, and tags.

To handle this, I designed an adaptive tiering controller called Workload-Adaptive Hybrid Indexing (WAHI). It watches which data segments actually answer search queries, keeps hot segments in RAM with HNSW, moves warm segments to POSIX mmap HNSW, and stores cold segments on disk compressed with Product Quantization (PQ), adjusting automatically as traffic shifts.

## System Flow (How Everything Connects)

    Client / Benchmark Script
            |
            v
    Node.js API & Hybrid Query Optimizer (port 3000)
            |
            +---> Metadata SQL & Filters ---> PostgreSQL 16 (port 5432)
            |
            +---> Vector CRUD & Search -----> C++20 Vector Engine (port 8080)
                                                    |
                                                    +--> Write-Ahead Log (CRC32 + fsync)
                                                    +--> Mutable Memtable (HNSW)
                                                    +--> WAHI 3-Tier Sealed Segments
                                                    +--> /metrics -> Prometheus (9090) -> Grafana (3001)

## Step-by-Step Flows

### 1. Write and Crash-Recovery Flow

Every insert, update, or delete is written to the Write-Ahead Log before touching memory, so even an abrupt SIGKILL crash recovers without losing committed writes.

    Insert / Update / Delete Request
            |
            v
    Append record to wal.log with CRC32 checksum
            |
            +--> If ID already exists: mark old slot as tombstone
            |
            v
    Insert into active HNSW Memtable
            |
            +--> Memtable not full -> Return OK
            |
            +--> Memtable full -> Flush segment (.vec, .hnsw, .meta, .pq) to disk
                                      |
                                      v
                                Update db_manifest.json and checkpoint WAL

When the engine restarts after a crash:
1. It reads db_manifest.json and reloads all sealed disk segments.
2. It replays wal.log for any records written after the last checkpoint.
3. If a crash left partial bytes at the end of wal.log, the CRC32 check catches it and truncates cleanly back to the last valid record.

### 2. Hybrid Search Flow (Vector + PostgreSQL Metadata)

Always pre-filtering or always post-filtering performs poorly across different filter sizes. The Node.js query optimizer in api/src/optimizer/queryOptimizer.js estimates filter selectivity in PostgreSQL first and picks the better plan:

    POST /api/v1/search (vector + optional filter)
            |
            +--> No filter ------> Search C++ Engine directly -> Fetch metadata from Postgres
            |
            +--> Has filter -----> Count matching rows in PostgreSQL
                                        |
                                        +--> 0 matches -> Return empty list immediately
                                        |
                                        +--> Tight filter (<= 500 rows OR <= 8% selectivity)
                                        |       Fetch allowed_ids from Postgres -> Allowlist search in C++ Engine
                                        |
                                        +--> Broad filter (> 8% selectivity)
                                                Expand k in C++ Engine -> Intersect in Postgres
                                                (Falls back to allowlist search if fewer than k survive)

### 3. WAHI Adaptive Tiering Flow

Static engines like DiskANN/Vamana or SPANN choose one layout up front. WAHI moves sealed segments across three tiers at runtime while staying inside a fixed RAM budget:

    Incoming Search & Write Traffic
            |
            v
    Track useful top-K hits, scans, and writes per segment (EWMA)
            |
            v
    Score each segment by utility per MB of RAM saved
            |
            v
    Allocate tiers greedily inside the RAM budget (with hysteresis & cooldown)
            |
            +--> Tier 1: HOT_RAM_HNSW   (vectors + HNSW graph pinned in RAM)
            +--> Tier 2: WARM_MMAP_HNSW (HNSW graph in RAM, vectors accessed via POSIX mmap)
            +--> Tier 3: COLD_DISK_PQ   (PQ codes in RAM, raw vectors on disk via pread + LRU cache)

One key lesson from benchmarking: tracking plain scan counts fails when pure vector searches check every segment, because every segment gets the same scan count. Tracking useful hits (which segments actually produced the final top-K results) cleanly identifies where the real hotspot is.

## Project Structure

- engine/ : C++20 core engine (distance metrics, brute force, HNSW, Product Quantization, WAL, mmap + LRU cache, segments, WAHI controller, HTTP/metrics server, GoogleTests, benchmarks).
- api/ : Node.js Express server, PostgreSQL schema and batch UNNEST queries, hybrid query optimizer, and Node unit tests.
- observability/ : Prometheus scrape configuration and pre-built Grafana dashboard.
- benchmarks/ : Python end-to-end benchmark, SIGKILL crash-recovery and concurrency stress test, and saved JSON reports.
- docker-compose.yml : Runs PostgreSQL, C++20 engine (512m memory cap), Node.js API, Prometheus, and Grafana together.

## Benchmark and Test Results

All tests and benchmarks ran inside Docker (Ubuntu 24.04, g++ 13.3.0 -O3 -std=c++20, Node 20, PostgreSQL 16).

### Unit and Concurrency Tests
- C++20 engine (ctest): 6/6 suites passed (test_distance_and_bf, test_hnsw, test_wal_and_mmap, test_pq, test_wahi, test_concurrency).
- Node.js API (node --test): 3/3 tests passed (SQL filter builder, tight pre-filter plan, broad post-filter plan).

### Core Algorithm Comparison (18,000 vectors, 128d, 60 clusters, Cosine)

    Index / Config        Recall@10    p50 (ms)    p95 (ms)       QPS        RAM
    ----------------------------------------------------------------------------
    BruteForce               1.0000       1.742       2.326     555.4   17.02 MB
    HNSW (ef=32)             0.9633       0.069       0.123   12949.1   21.28 MB
    HNSW (ef=64)             0.9660       0.080       0.131   11101.3   21.28 MB
    HNSW (ef=128)            0.9667       0.102       0.138    9157.9   21.28 MB
    PQ (ADC only)            0.2273       0.386       0.537    2430.0    0.67 MB
    PQ (ExactRerank 8x)      0.7560       0.446       0.715    2080.2    0.67 MB

### Skewed 85/15 Workload Under Tight RAM (5.18 MB cap across 12 segments)

    Policy                  Recall@10    p95 (ms)      QPS    RAM Used   Tiers (H/W/C)   Disk Reads
    -----------------------------------------------------------------------------------------------
    static_all_ram (no cap)    0.9999       1.258   1110.9   16.27 MB      12 / 0 / 0             0
    static_all_disk_pq         0.6594       2.275    631.0    1.91 MB      0 / 0 / 12       503,614
    static_lru (hit-recency)   0.8232       1.653    730.2    4.53 MB       3 / 0 / 9       377,520
    wahi_adaptive (mine)       0.9544       1.194    985.9    5.12 MB       3 / 4 / 5       217,101

Using about 68% less RAM than static_all_ram, WAHI kept 0.9544 Recall@10 and cut disk reads by 42.5% compared to LRU.

### Shifting Hotspot Workload (3 Epochs Under 5.18 MB cap)

    Policy                  Overall Recall   Adapted (Ep 1 / 2 / 3)    p95 (ms)   Prom/Dem   Disk Reads
    ---------------------------------------------------------------------------------------------------
    static_all_disk_pq              0.6153   0.6924 / 0.5844 / 0.5560     2.301     0 / 12      943,544
    static_lru                      0.9161   0.9320 / 0.9476 / 0.9604     1.576    32 / 41      707,015
    wahi_adaptive (mine)            0.9465   0.9551 / 0.9644 / 0.9764     1.820    26 / 34      410,993

### End-to-End Docker Stack and Crash Recovery
- Batch ingestion (4,000 vectors + PostgreSQL metadata via Node.js): 2.556s (1,564.9 vec/s).
- Hybrid optimizer over HTTP: picked prefilter_allowlist at 0.75% selectivity (26.16 ms) and adaptive_postfilter at 48.18% selectivity (5.40 ms).
- SIGKILL recovery: killed vectordb-engine with SIGKILL after writing unsealed inserts and a delete to the WAL; on restart, all surviving writes and the delete replayed cleanly with zero data loss.

## How to Run and Verify

1. Start the full stack:

    docker compose up -d --build

2. Run all 6 C++20 unit test suites and Node.js API tests:

    docker exec vectordb-engine sh -c 'for t in /app/tests/test_*; do "$t" || exit 1; done'
    docker exec vectordb-api node --test tests/api.test.js

3. Run the end-to-end Python benchmark and SIGKILL crash-recovery test:

    docker compose --profile bench run --rm benchmark
    python benchmarks/crash_recovery_test.py

4. Open live dashboards and endpoints:

- Grafana dashboard: http://localhost:3001/d/wahi-vectordb-overview/adaptive-vectordb-wahi-telemetry
- Prometheus targets: http://localhost:9090/targets
- Node.js API: http://localhost:3000/health and http://localhost:3000/api/v1/stats
- C++20 engine: http://localhost:8080/health and http://localhost:8080/v1/stats
