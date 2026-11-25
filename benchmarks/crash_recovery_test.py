import concurrent.futures
import json
import random
import subprocess
import sys
import time
import urllib.request
import urllib.error


API_URL = "http://localhost:3000"
ENGINE_URL = "http://localhost:8080"


def http_json(method: str, url: str, payload=None, timeout=10):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def http_text(url: str, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return resp.status, resp.read().decode("utf-8")


def wait_healthy(url: str, timeout_s=60):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            st, body = http_json("GET", f"{url}/health", timeout=2)
            if st == 200 and body.get("status") == "ok":
                return body
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"{url} did not become healthy")


def main():
    print("1. Checking initial stack health...")
    wait_healthy(ENGINE_URL)
    wait_healthy(API_URL)

    # Insert distinctive unsealed vectors (high IDs > 900000) without flushing so they reside in WAL/memtable
    dim = int(http_json("GET", f"{ENGINE_URL}/health")[1]["dimension"])
    sentinel_ids = [900001, 900002, 900003, 900004, 900005]
    print(f"2. Inserting {len(sentinel_ids)} sentinel vectors (dim={dim}) into API + Postgres + Engine WAL...")
    for idx, vid in enumerate(sentinel_ids):
        vec = [0.05] * dim
        vec[idx % dim] = 5.0
        st, _ = http_json("POST", f"{API_URL}/api/v1/vectors", {
            "id": vid,
            "external_key": f"crash_sentinel_{vid}",
            "vector": vec,
            "category": "crash_test",
            "price": 99.5 + idx,
            "tags": ["wal", "durable"],
            "metadata": {"phase": "pre_sigkill"},
        })
        assert st == 200

    # Delete sentinel 900005 to verify delete WAL replay across crash too
    st, del_res = http_json("DELETE", f"{API_URL}/api/v1/vectors/900005")
    assert st == 200 and del_res["deleted"] is True

    print("3. Sending SIGKILL to vectordb-engine container (simulating abrupt power loss)...")
    subprocess.run(["docker", "kill", "-s", "KILL", "vectordb-engine"], check=True)
    time.sleep(1.0)

    print("4. Restarting vectordb-engine container and verifying WAL crash recovery...")
    subprocess.run(["docker", "start", "vectordb-engine"], check=True)
    eng_health = wait_healthy(ENGINE_URL, timeout_s=60)
    wait_healthy(API_URL, timeout_s=60)
    print(f"   Engine recovered: {eng_health}")

    # Verify 900001..900004 exist and 900005 remains deleted
    for vid in [900001, 900002, 900003, 900004]:
        st, got = http_json("GET", f"{API_URL}/api/v1/vectors/{vid}")
        assert st == 200 and got["id"] == vid and got["record"]["category"] == "crash_test"

    try:
        http_json("GET", f"{API_URL}/api/v1/vectors/900005")
        raise AssertionError("Expected 900005 to remain deleted after crash recovery!")
    except urllib.error.HTTPError as e:
        assert e.code in (404, 400)

    # Verify hybrid search finds sentinel 900001
    q = [0.05] * dim
    q[0] = 5.0
    st, sres = http_json("POST", f"{API_URL}/api/v1/search", {
        "vector": q,
        "k": 3,
        "filter": {"category": "crash_test"},
    })
    assert st == 200
    assert any(r["id"] == 900001 for r in sres["results"]), f"Expected 900001 in {sres}"
    print("   Crash recovery verified: 100% WAL survival + delete durability + hybrid search intact.")

    print("5. Running concurrent HTTP stress test (8 workers x 25 requests = 200 concurrent ops)...")
    def worker(wid: int):
        rng = random.Random(1000 + wid)
        ok = 0
        for i in range(25):
            if i % 5 == 0:
                vid = 910000 + wid * 100 + i
                vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
                st, _ = http_json("POST", f"{API_URL}/api/v1/vectors", {
                    "id": vid,
                    "vector": vec,
                    "category": "stress",
                    "price": 42.0,
                    "tags": ["concurrent"],
                })
                assert st == 200
            else:
                vec = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
                st, res = http_json("POST", f"{API_URL}/api/v1/search", {
                    "vector": vec,
                    "k": 5,
                })
                assert st == 200 and len(res["results"]) > 0
            ok += 1
        return ok

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        counts = list(ex.map(worker, range(8)))
    elapsed = time.perf_counter() - t0
    print(f"   Completed {sum(counts)} concurrent API operations in {elapsed:.2f}s ({sum(counts)/elapsed:.1f} ops/s).")

    print("6. Checking Prometheus metrics endpoints on Engine, API, Prometheus, and Grafana...")
    st_em, em = http_text(f"{ENGINE_URL}/metrics")
    assert st_em == 200 and "vectordb_vectors_total" in em
    st_am, am = http_text(f"{API_URL}/metrics")
    assert st_am == 200 and "api_optimizer_plan_total" in am
    st_prom, _ = http_text("http://localhost:9090/-/healthy")
    assert st_prom == 200
    st_graf, _ = http_text("http://localhost:3001/api/health")
    assert st_graf == 200
    print("   All observability endpoints healthy!")


if __name__ == "__main__":
    main()
