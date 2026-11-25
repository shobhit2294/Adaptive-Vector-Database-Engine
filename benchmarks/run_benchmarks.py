import json
import os
import time
import numpy as np
import requests


API_URL = os.environ.get("API_URL", "http://api-server:3000")
ENGINE_URL = os.environ.get("ENGINE_URL", "http://vectordb-engine:8080")
DIM = int(os.environ.get("VECTORDB_DIM", "64"))
OUT_PATH = os.environ.get("BENCH_OUT", "/results/e2e_benchmark.json")


def wait_for_healthy(session: requests.Session, url: str, timeout_s: int = 60):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = session.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"Service {url} did not become healthy within {timeout_s}s")


def make_clustered_vectors(num_clusters: int, per_cluster: int, dim: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0.0, 0.35, size=(num_clusters, dim)).astype(np.float32)
    for c in range(num_clusters):
        block = (c * 4) % dim
        centers[c, block : min(dim, block + 4)] += 2.5 if (c % 2 == 0) else -2.5

    vectors = []
    categories = ["electronics", "books", "home", "sports", "fashion", "kitchen"]
    for c in range(num_clusters):
        pts = centers[c] + rng.normal(0.0, 0.22, size=(per_cluster, dim)).astype(np.float32)
        for i in range(per_cluster):
            idx = c * per_cluster + i + 1
            cat = categories[c % len(categories)]
            price = float(round(10.0 + ((idx * 17) % 490), 2))
            tags = [f"cluster_{c}", "featured" if (idx % 10 == 0) else "standard"]
            vectors.append({
                "id": idx,
                "external_key": f"doc_{idx}",
                "vector": pts[i].tolist(),
                "category": cat,
                "price": price,
                "tags": tags,
                "metadata": {"cluster": c, "tier_hint": "sample"},
            })
    return centers, vectors


def exact_cosine_topk(vectors_mat: np.ndarray, ids: np.ndarray, query: np.ndarray, k: int):
    qn = query / (np.linalg.norm(query) + 1e-12)
    vn = vectors_mat / (np.linalg.norm(vectors_mat, axis=1, keepdims=True) + 1e-12)
    sims = vn @ qn
    top_idx = np.argsort(-sims)[:k]
    return set(int(ids[i]) for i in top_idx)


def main():
    s = requests.Session()
    print(f"Waiting for API ({API_URL}) and Engine ({ENGINE_URL})...")
    wait_for_healthy(s, ENGINE_URL)
    wait_for_healthy(s, API_URL)

    num_clusters = 40
    per_cluster = 100  # 4,000 vectors across 8 segments (5 clusters per segment)
    centers, items = make_clustered_vectors(num_clusters, per_cluster, DIM)

    print(f"Ingesting {len(items)} vectors + metadata via Node.js API -> PostgreSQL + C++20 Engine...")
    t0 = time.perf_counter()
    batch_size = 500
    for i in range(0, len(items), batch_size):
        r = s.post(f"{API_URL}/api/v1/vectors/batch", json={"items": items[i : i + batch_size]}, timeout=30)
        r.raise_for_status()
    ingest_s = time.perf_counter() - t0
    print(f"Ingestion complete in {ingest_s:.2f}s ({len(items) / ingest_s:.1f} vec/s)")

    s.post(f"{API_URL}/api/v1/admin/flush", timeout=15).raise_for_status()

    # Constrain engine RAM budget to ~55% of full RAM to test 3-tier WAHI over HTTP end-to-end
    stats_before = s.get(f"{API_URL}/api/v1/stats", timeout=10).json()
    full_ram = max(1, stats_before["engine"]["ram_bytes"]["total"])
    tight_budget = max(int(full_ram * 0.55), 1100000)
    s.post(f"{API_URL}/api/v1/wahi/policy", json={"mode": "wahi_adaptive", "ram_budget_bytes": tight_budget}, timeout=15).raise_for_status()

    vec_mat = np.array([x["vector"] for x in items], dtype=np.float32)
    id_arr = np.array([x["id"] for x in items], dtype=np.int64)
    rng = np.random.default_rng(123)

    # Pure search evaluation under shifting hotspot
    print("Running Phase A hotspot queries (clusters 0..4, Segment 1)...")
    phase_a_lats = []
    phase_a_recalls = []
    for i in range(120):
        cl = i % 5
        q = (centers[cl] + rng.normal(0.0, 0.22, size=DIM).astype(np.float32))
        gt = exact_cosine_topk(vec_mat, id_arr, q, 10)
        ta = time.perf_counter()
        resp = s.post(f"{API_URL}/api/v1/search", json={"vector": q.tolist(), "k": 10}, timeout=10).json()
        phase_a_lats.append((time.perf_counter() - ta) * 1000.0)
        got = set(r["id"] for r in resp["results"])
        phase_a_recalls.append(len(gt & got) / 10.0)

    s.post(f"{API_URL}/api/v1/wahi/reconcile", timeout=10).raise_for_status()
    stats_a = s.get(f"{API_URL}/api/v1/stats", timeout=10).json()

    print("Shifting workload to Phase B hotspot queries (clusters 35..39, Segment 8)...")
    phase_b_lats = []
    phase_b_recalls = []
    for i in range(120):
        cl = 35 + (i % 5)
        q = (centers[cl] + rng.normal(0.0, 0.22, size=DIM).astype(np.float32))
        gt = exact_cosine_topk(vec_mat, id_arr, q, 10)
        ta = time.perf_counter()
        resp = s.post(f"{API_URL}/api/v1/search", json={"vector": q.tolist(), "k": 10}, timeout=10).json()
        phase_b_lats.append((time.perf_counter() - ta) * 1000.0)
        got = set(r["id"] for r in resp["results"])
        phase_b_recalls.append(len(gt & got) / 10.0)

    s.post(f"{API_URL}/api/v1/wahi/reconcile", timeout=10).raise_for_status()
    stats_b = s.get(f"{API_URL}/api/v1/stats", timeout=10).json()

    # Hybrid metadata filter experiments (tight prefilter vs broad postfilter)
    print("Evaluating Hybrid Query Optimizer (tight prefilter vs broad postfilter)...")
    q_hyb = centers[0].tolist()
    tight_resp = s.post(
        f"{API_URL}/api/v1/search",
        json={
            "vector": q_hyb,
            "k": 10,
            "filter": {"category": "electronics", "tags_all": ["featured"], "max_price": 200},
        },
        timeout=10,
    ).json()

    broad_resp = s.post(
        f"{API_URL}/api/v1/search",
        json={
            "vector": q_hyb,
            "k": 10,
            "filter": {"categories": ["electronics", "books", "home"], "min_price": 50},
        },
        timeout=10,
    ).json()

    report = {
        "vectors_ingested": len(items),
        "ingest_seconds": round(ingest_s, 3),
        "full_ram_bytes": full_ram,
        "tight_budget_bytes": tight_budget,
        "phase_a": {
            "mean_recall_at_10": float(np.mean(phase_a_recalls)),
            "adapted_recall_at_10": float(np.mean(phase_a_recalls[60:])),
            "p50_ms": float(np.percentile(phase_a_lats, 50)),
            "p95_ms": float(np.percentile(phase_a_lats, 95)),
            "tiers": stats_a["engine"]["tiers"],
            "ram_total_bytes": stats_a["engine"]["ram_bytes"]["total"],
        },
        "phase_b_after_shift": {
            "mean_recall_at_10": float(np.mean(phase_b_recalls)),
            "adapted_recall_at_10": float(np.mean(phase_b_recalls[60:])),
            "p50_ms": float(np.percentile(phase_b_lats, 50)),
            "p95_ms": float(np.percentile(phase_b_lats, 95)),
            "tiers": stats_b["engine"]["tiers"],
            "ram_total_bytes": stats_b["engine"]["ram_bytes"]["total"],
            "promotions": stats_b["engine"]["counters"]["wahi_promotions_total"],
            "demotions": stats_b["engine"]["counters"]["wahi_demotions_total"],
        },
        "hybrid_tight": {
            "plan": tight_resp["query_plan"]["plan"],
            "selectivity": tight_resp["query_plan"]["selectivity"],
            "results_count": len(tight_resp["results"]),
            "total_ms": tight_resp["query_plan"]["total_ms"],
        },
        "hybrid_broad": {
            "plan": broad_resp["query_plan"]["plan"],
            "selectivity": broad_resp["query_plan"]["selectivity"],
            "results_count": len(broad_resp["results"]),
            "total_ms": broad_resp["query_plan"]["total_ms"],
        },
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== End-to-End Docker Stack Benchmark Summary ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
