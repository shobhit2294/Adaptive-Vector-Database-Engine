#pragma once

#include "vectordb/types.hpp"
#include <cstdint>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace vectordb {

// 64-byte aligned header at the start of every packed .vec segment file
struct alignas(64) VectorFileHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t dim;
    uint32_t count;
    uint8_t reserved[48];
};
static_assert(sizeof(VectorFileHeader) == 64, "VectorFileHeader must be 64 bytes");

// Writes packed fp32 vector arena to disk (64-byte header + count * dim floats).
bool write_vector_file(const std::string& path, const float* data, uint32_t count, uint32_t dim);

// POSIX mmap read-only wrapper for WARM_MMAP_HNSW tier.
class MmapVectorFile {
public:
    MmapVectorFile() = default;
    ~MmapVectorFile();

    MmapVectorFile(const MmapVectorFile&) = delete;
    MmapVectorFile& operator=(const MmapVectorFile&) = delete;

    bool open(const std::string& path);
    void close();
    void advise_random();
    void advise_dontneed();

    bool is_open() const { return mapped_ != nullptr; }
    const float* vectors() const { return vectors_; }
    uint32_t dim() const { return dim_; }
    uint32_t count() const { return count_; }
    uint64_t file_bytes() const { return file_size_; }

private:
    int fd_ = -1;
    void* mapped_ = nullptr;
    size_t file_size_ = 0;
    const float* vectors_ = nullptr;
    uint32_t dim_ = 0;
    uint32_t count_ = 0;
};

// Direct pread() reader for COLD_DISK_PQ reranking.
class DiskVectorReader {
public:
    DiskVectorReader() = default;
    ~DiskVectorReader();

    DiskVectorReader(const DiskVectorReader&) = delete;
    DiskVectorReader& operator=(const DiskVectorReader&) = delete;

    bool open(const std::string& path);
    void close();
    bool read_slot(uint32_t slot, float* out_vec) const;
    bool is_open() const { return fd_ >= 0; }
    uint32_t dim() const { return dim_; }
    uint32_t count() const { return count_; }
    uint64_t file_bytes() const { return file_bytes_; }

private:
    int fd_ = -1;
    uint32_t dim_ = 0;
    uint32_t count_ = 0;
    uint64_t file_bytes_ = 0;
};

// Thread-safe bounded LRU cache for cold-segment disk vector reads.
class BoundedVectorCache {
public:
    explicit BoundedVectorCache(uint64_t max_bytes);

    void set_capacity(uint64_t max_bytes);
    uint64_t capacity() const;
    uint64_t used_bytes() const;
    void invalidate_segment(SegmentId seg_id);
    void clear();

    bool get_or_read(SegmentId seg_id,
                     uint32_t slot,
                     const DiskVectorReader& reader,
                     std::vector<float>& out,
                     bool& out_cache_hit,
                     uint64_t& out_bytes_read);

private:
    struct Key {
        SegmentId seg_id;
        uint32_t slot;
        bool operator==(const Key& o) const {
            return seg_id == o.seg_id && slot == o.slot;
        }
    };
    struct KeyHash {
        size_t operator()(const Key& k) const {
            return (static_cast<uint64_t>(k.seg_id) << 32) ^ k.slot;
        }
    };
    struct Entry {
        Key key;
        std::vector<float> vec;
    };

    uint64_t max_bytes_;
    uint64_t used_bytes_ = 0;
    mutable std::mutex mutex_;
    std::list<Entry> lru_;
    std::unordered_map<Key, std::list<Entry>::iterator, KeyHash> index_;
};

} // namespace vectordb
