#include "vectordb/database.hpp"
#include <algorithm>
#include <arpa/inet.h>
#include <atomic>
#include <cctype>
#include <csignal>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <nlohmann/json.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace vectordb;
using json = nlohmann::json;

namespace {

std::atomic<bool> g_running{true};
int g_listen_fd = -1;

void handle_signal(int) {
    g_running.store(false, std::memory_order_relaxed);
    if (g_listen_fd >= 0) {
        ::shutdown(g_listen_fd, SHUT_RDWR);
        ::close(g_listen_fd);
        g_listen_fd = -1;
    }
}

std::string to_lower(std::string s) {
    for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

std::string trim(const std::string& s) {
    size_t a = 0;
    while (a < s.size() && (s[a] == ' ' || s[a] == '\t' || s[a] == '\r' || s[a] == '\n')) ++a;
    size_t b = s.size();
    while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r' || s[b - 1] == '\n')) --b;
    return s.substr(a, b - a);
}

struct HttpRequest {
    std::string method;
    std::string path;
    std::unordered_map<std::string, std::string> headers;
    std::string body;
    bool keep_alive = true;
};

bool read_http_request(int fd, std::string& buffer, HttpRequest& req) {
    while (true) {
        size_t hdr_end = buffer.find("\r\n\r\n");
        if (hdr_end != std::string::npos) {
            std::string head = buffer.substr(0, hdr_end);
            std::istringstream iss(head);
            std::string req_line;
            if (!std::getline(iss, req_line)) return false;
            if (!req_line.empty() && req_line.back() == '\r') req_line.pop_back();
            std::istringstream rl(req_line);
            std::string version;
            rl >> req.method >> req.path >> version;
            if (req.method.empty() || req.path.empty()) return false;
            auto qpos = req.path.find('?');
            if (qpos != std::string::npos) req.path = req.path.substr(0, qpos);

            req.headers.clear();
            std::string line;
            size_t content_len = 0;
            req.keep_alive = (to_lower(version) != "http/1.0");
            while (std::getline(iss, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                auto colon = line.find(':');
                if (colon != std::string::npos) {
                    std::string k = to_lower(trim(line.substr(0, colon)));
                    std::string v = trim(line.substr(colon + 1));
                    req.headers[k] = v;
                    if (k == "content-length") {
                        content_len = static_cast<size_t>(std::stoull(v));
                    } else if (k == "connection") {
                        std::string lv = to_lower(v);
                        if (lv == "close") req.keep_alive = false;
                        else if (lv == "keep-alive") req.keep_alive = true;
                    }
                }
            }

            size_t total_needed = hdr_end + 4 + content_len;
            while (buffer.size() < total_needed) {
                char chunk[8192];
                ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
                if (n <= 0) return false;
                buffer.append(chunk, static_cast<size_t>(n));
            }
            req.body = buffer.substr(hdr_end + 4, content_len);
            buffer.erase(0, total_needed);
            return true;
        }

        char chunk[8192];
        ssize_t n = ::recv(fd, chunk, sizeof(chunk), 0);
        if (n <= 0) return false;
        buffer.append(chunk, static_cast<size_t>(n));
        if (buffer.size() > 64ULL * 1024ULL * 1024ULL) return false;
    }
}

bool send_http_response(int fd,
                        int status,
                        const std::string& status_text,
                        const std::string& content_type,
                        const std::string& body,
                        bool keep_alive) {
    std::ostringstream oss;
    oss << "HTTP/1.1 " << status << " " << status_text << "\r\n"
        << "Content-Type: " << content_type << "\r\n"
        << "Content-Length: " << body.size() << "\r\n"
        << "Connection: " << (keep_alive ? "keep-alive" : "close") << "\r\n\r\n"
        << body;
    std::string out = oss.str();
    size_t sent = 0;
    while (sent < out.size()) {
        ssize_t n = ::send(fd, out.data() + sent, out.size() - sent, MSG_NOSIGNAL);
        if (n <= 0) return false;
        sent += static_cast<size_t>(n);
    }
    return true;
}

uint32_t env_u32(const char* name, uint32_t def) {
    const char* v = std::getenv(name);
    return (v && *v) ? static_cast<uint32_t>(std::stoul(v)) : def;
}

uint64_t env_u64(const char* name, uint64_t def) {
    const char* v = std::getenv(name);
    return (v && *v) ? static_cast<uint64_t>(std::stoull(v)) : def;
}

std::string env_str(const char* name, const std::string& def) {
    const char* v = std::getenv(name);
    return (v && *v) ? std::string(v) : def;
}

void handle_request(VectorDatabase& db, const HttpRequest& req, int client_fd) {
    auto reply_json = [&](int status, const std::string& st, const json& j) {
        send_http_response(client_fd, status, st, "application/json", j.dump(), req.keep_alive);
    };

    try {
        if (req.method == "GET" && req.path == "/health") {
            auto snap = db.snapshot();
            reply_json(200, "OK", {
                {"status", "ok"},
                {"live_vectors", snap.live_vectors},
                {"dimension", db.config().dimension},
                {"metric", to_string(db.config().metric)},
                {"wahi_mode", to_string(snap.wahi_mode)}
            });
            return;
        }

        if (req.method == "GET" && req.path == "/metrics") {
            std::string body = db.prometheus_metrics();
            send_http_response(client_fd, 200, "OK", "text/plain; version=0.0.4", body, req.keep_alive);
            return;
        }

        if (req.method == "GET" && req.path == "/v1/stats") {
            auto snap = db.snapshot();
            auto lat = db.metrics().latency_summary();
            auto tels = db.segment_telemetries();
            json segs = json::array();
            for (const auto& t : tels) {
                segs.push_back({
                    {"segment_id", t.segment_id},
                    {"tier", to_string(t.tier)},
                    {"vector_count", t.vector_count},
                    {"live_count", t.live_count},
                    {"tombstone_count", t.tombstone_count},
                    {"scan_freq_ewma", t.scan_freq_ewma},
                    {"useful_hits_ewma", t.useful_hits_ewma},
                    {"write_freq_ewma", t.write_freq_ewma},
                    {"utility_score", t.utility_score},
                    {"ram_bytes_vectors", t.ram_bytes_vectors},
                    {"ram_bytes_hnsw", t.ram_bytes_hnsw},
                    {"ram_bytes_pq", t.ram_bytes_pq},
                    {"ram_bytes_total", t.total_ram_bytes()},
                    {"disk_bytes", t.disk_bytes}
                });
            }
            reply_json(200, "OK", {
                {"live_vectors", snap.live_vectors},
                {"tombstones", snap.tombstones},
                {"wahi_mode", to_string(snap.wahi_mode)},
                {"ram_budget_bytes", snap.ram_budget_bytes},
                {"ram_bytes", {
                    {"vectors", snap.ram_bytes_vectors},
                    {"hnsw", snap.ram_bytes_hnsw},
                    {"pq", snap.ram_bytes_pq},
                    {"cache", snap.ram_bytes_cache},
                    {"total", snap.ram_bytes_total}
                }},
                {"disk_bytes_total", snap.disk_bytes_total},
                {"wal_bytes", snap.wal_bytes},
                {"wal_records", snap.wal_records},
                {"tiers", {
                    {"memtable", snap.segments_memtable},
                    {"hot_ram", snap.segments_hot_ram},
                    {"warm_mmap", snap.segments_warm_mmap},
                    {"cold_disk_pq", snap.segments_cold_disk_pq}
                }},
                {"counters", {
                    {"queries_total", db.metrics().queries_total.load()},
                    {"filtered_queries_total", db.metrics().filtered_queries_total.load()},
                    {"inserts_total", db.metrics().inserts_total.load()},
                    {"updates_total", db.metrics().updates_total.load()},
                    {"deletes_total", db.metrics().deletes_total.load()},
                    {"disk_reads_total", db.metrics().disk_reads_total.load()},
                    {"disk_bytes_read_total", db.metrics().disk_bytes_read_total.load()},
                    {"cache_hits_total", db.metrics().cache_hits_total.load()},
                    {"cache_misses_total", db.metrics().cache_misses_total.load()},
                    {"wahi_promotions_total", db.metrics().wahi_promotions_total.load()},
                    {"wahi_demotions_total", db.metrics().wahi_demotions_total.load()},
                    {"compactions_total", db.metrics().compactions_total.load()}
                }},
                {"latency", {
                    {"p50_ms", lat.p50_ms},
                    {"p95_ms", lat.p95_ms},
                    {"p99_ms", lat.p99_ms},
                    {"avg_ms", lat.avg_ms}
                }},
                {"segments", segs}
            });
            return;
        }

        if (req.method == "POST" && req.path == "/v1/vectors") {
            auto j = json::parse(req.body);
            VectorId id = j.at("id").get<VectorId>();
            std::vector<float> vec = j.at("vector").get<std::vector<float>>();
            db.upsert(id, vec);
            reply_json(200, "OK", {{"status", "upserted"}, {"id", id}});
            return;
        }

        if (req.method == "POST" && req.path == "/v1/vectors/batch") {
            auto j = json::parse(req.body);
            const auto& arr = j.at("items");
            std::vector<std::pair<VectorId, std::vector<float>>> items;
            items.reserve(arr.size());
            for (const auto& item : arr) {
                items.emplace_back(item.at("id").get<VectorId>(), item.at("vector").get<std::vector<float>>());
            }
            db.upsert_batch(items);
            reply_json(200, "OK", {{"status", "upserted_batch"}, {"count", items.size()}});
            return;
        }

        if (req.path.rfind("/v1/vectors/", 0) == 0) {
            std::string id_str = req.path.substr(std::strlen("/v1/vectors/"));
            VectorId id = std::stoull(id_str);
            if (req.method == "GET") {
                std::vector<float> vec;
                if (!db.get(id, vec)) {
                    reply_json(404, "Not Found", {{"error", "vector not found"}});
                    return;
                }
                reply_json(200, "OK", {{"id", id}, {"vector", vec}});
                return;
            }
            if (req.method == "DELETE") {
                bool removed = db.remove(id);
                reply_json(200, "OK", {{"deleted", removed}, {"id", id}});
                return;
            }
        }

        if (req.method == "POST" && req.path == "/v1/search") {
            auto j = json::parse(req.body);
            std::vector<float> query = j.at("vector").get<std::vector<float>>();
            SearchOptions opts;
            opts.k = j.value("k", 10u);
            opts.ef_search = j.value("ef_search", 0u);
            opts.cold_rerank_factor = j.value("cold_rerank_factor", 0u);
            opts.bypass_cold_rerank = j.value("bypass_cold_rerank", false);

            std::unordered_set<VectorId> allowed;
            const std::unordered_set<VectorId>* allowed_ptr = nullptr;
            if (j.contains("allowed_ids") && j["allowed_ids"].is_array()) {
                for (const auto& id_val : j["allowed_ids"]) {
                    allowed.insert(id_val.get<VectorId>());
                }
                allowed_ptr = &allowed;
            }

            SearchDiagnostics diag;
            auto results = db.search(query, opts, &diag, allowed_ptr);
            json res_arr = json::array();
            for (const auto& r : results) {
                res_arr.push_back({
                    {"id", r.id},
                    {"distance", r.distance},
                    {"score", r.score},
                    {"tier", to_string(r.source_tier)}
                });
            }
            reply_json(200, "OK", {
                {"results", res_arr},
                {"diagnostics", {
                    {"latency_ms", diag.latency_ms},
                    {"segments_scanned", diag.segments_scanned},
                    {"hot_segments_scanned", diag.hot_segments_scanned},
                    {"warm_segments_scanned", diag.warm_segments_scanned},
                    {"cold_segments_scanned", diag.cold_segments_scanned},
                    {"disk_reads", diag.disk_reads},
                    {"disk_bytes_read", diag.disk_bytes_read},
                    {"cache_hits", diag.cache_hits},
                    {"cache_misses", diag.cache_misses}
                }}
            });
            return;
        }

        if (req.method == "POST" && req.path == "/v1/wahi/reconcile") {
            auto decisions = db.trigger_wahi_reconcile();
            json arr = json::array();
            for (const auto& d : decisions) {
                arr.push_back({
                    {"segment_id", d.segment_id},
                    {"from_tier", to_string(d.from_tier)},
                    {"to_tier", to_string(d.to_tier)},
                    {"utility_score", d.utility_score}
                });
            }
            reply_json(200, "OK", {{"transitions", arr}});
            return;
        }

        if (req.method == "POST" && req.path == "/v1/wahi/policy") {
            auto j = json::parse(req.body);
            std::string mode_str = j.value("mode", std::string("wahi_adaptive"));
            uint64_t budget = j.value("ram_budget_bytes", 0ULL);
            db.set_wahi_policy(policy_from_string(mode_str), budget);
            auto snap = db.snapshot();
            reply_json(200, "OK", {
                {"wahi_mode", to_string(snap.wahi_mode)},
                {"ram_budget_bytes", snap.ram_budget_bytes},
                {"ram_bytes_total", snap.ram_bytes_total}
            });
            return;
        }

        if (req.method == "POST" && req.path == "/v1/admin/flush") {
            bool ok = db.flush_and_seal();
            reply_json(200, "OK", {{"flushed", ok}});
            return;
        }

        if (req.method == "POST" && req.path == "/v1/admin/compact") {
            size_t compacted = db.compact_tombstones();
            reply_json(200, "OK", {{"compacted_segments", compacted}});
            return;
        }

        reply_json(404, "Not Found", {{"error", "endpoint not found"}});
    } catch (const std::exception& e) {
        reply_json(400, "Bad Request", {{"error", e.what()}});
    }
}

} // namespace

int main() {
    std::signal(SIGPIPE, SIG_IGN);
    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    DatabaseConfig cfg;
    cfg.data_dir = env_str("VECTORDB_DATA_DIR", "/data/vectordb");
    cfg.dimension = env_u32("VECTORDB_DIM", 128);
    cfg.metric = metric_from_string(env_str("VECTORDB_METRIC", "cosine"));
    cfg.segment_capacity = env_u32("VECTORDB_SEG_CAP", 2048);
    cfg.hnsw_m = env_u32("VECTORDB_HNSW_M", 16);
    cfg.hnsw_ef_construction = env_u32("VECTORDB_HNSW_EF_CONSTRUCT", 128);
    cfg.hnsw_ef_search = env_u32("VECTORDB_HNSW_EF_SEARCH", 64);
    cfg.pq_subspaces = env_u32("VECTORDB_PQ_SUBSPACES", 16);
    cfg.ram_budget_bytes = env_u64("VECTORDB_RAM_BUDGET_MB", 64) * 1024ULL * 1024ULL;
    cfg.disk_cache_bytes = env_u64("VECTORDB_DISK_CACHE_MB", 16) * 1024ULL * 1024ULL;
    cfg.wahi_mode = policy_from_string(env_str("VECTORDB_WAHI_MODE", "wahi_adaptive"));
    cfg.wahi_reconcile_interval_queries = env_u32("VECTORDB_WAHI_INTERVAL", 100);

    std::string fsync_str = env_str("VECTORDB_WAL_FSYNC", "batched");
    if (fsync_str == "always") cfg.wal_fsync = FsyncPolicy::ALWAYS;
    else if (fsync_str == "none") cfg.wal_fsync = FsyncPolicy::NONE;
    else cfg.wal_fsync = FsyncPolicy::BATCHED;

    VectorDatabase db(cfg);
    if (!db.open()) {
        std::cerr << "Failed to open VectorDatabase at " << cfg.data_dir << "\n";
        return 1;
    }

    uint32_t port = env_u32("VECTORDB_PORT", 8080);
    uint32_t num_workers = env_u32("VECTORDB_THREADS", std::max(4u, std::thread::hardware_concurrency()));

    g_listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (g_listen_fd < 0) {
        std::cerr << "Failed to create socket\n";
        return 1;
    }

    int opt = 1;
    ::setsockopt(g_listen_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    if (::bind(g_listen_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        std::cerr << "Failed to bind port " << port << "\n";
        return 1;
    }
    if (::listen(g_listen_fd, 128) < 0) {
        std::cerr << "Failed to listen\n";
        return 1;
    }

    std::cout << "Adaptive VectorDB Engine listening on 0.0.0.0:" << port
              << " (dim=" << cfg.dimension
              << ", metric=" << to_string(cfg.metric)
              << ", wahi_mode=" << to_string(cfg.wahi_mode)
              << ", ram_budget_mb=" << (cfg.ram_budget_bytes / (1024 * 1024))
              << ", recovered_vectors=" << db.snapshot().live_vectors << ")\n"
              << std::flush;

    std::mutex q_mtx;
    std::condition_variable q_cv;
    std::queue<int> conn_queue;

    std::vector<std::thread> workers;
    workers.reserve(num_workers);
    for (uint32_t i = 0; i < num_workers; ++i) {
        workers.emplace_back([&]() {
            while (true) {
                int cfd = -1;
                {
                    std::unique_lock lock(q_mtx);
                    q_cv.wait(lock, [&]() { return !g_running.load() || !conn_queue.empty(); });
                    if (!g_running.load() && conn_queue.empty()) return;
                    cfd = conn_queue.front();
                    conn_queue.pop();
                }

                int nodelay = 1;
                ::setsockopt(cfd, IPPROTO_TCP, TCP_NODELAY, &nodelay, sizeof(nodelay));

                std::string buf;
                HttpRequest req;
                while (g_running.load() && read_http_request(cfd, buf, req)) {
                    handle_request(db, req, cfd);
                    if (!req.keep_alive) break;
                }
                ::close(cfd);
            }
        });
    }

    while (g_running.load()) {
        sockaddr_in cli{};
        socklen_t clen = sizeof(cli);
        int cfd = ::accept(g_listen_fd, reinterpret_cast<sockaddr*>(&cli), &clen);
        if (cfd < 0) {
            if (!g_running.load()) break;
            continue;
        }
        {
            std::lock_guard lock(q_mtx);
            conn_queue.push(cfd);
        }
        q_cv.notify_one();
    }

    q_cv.notify_all();
    for (auto& w : workers) w.join();
    db.close();
    return 0;
}
