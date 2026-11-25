#include "vectordb/mmap_storage.hpp"
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vectordb {

namespace {
constexpr uint32_t kVecFileMagic = 0x56454331; // "VEC1"
constexpr uint32_t kVecFileVersion = 1;
} // namespace

bool write_vector_file(const std::string& path, const float* data, uint32_t count, uint32_t dim) {
    auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    const std::string tmp_path = path + ".tmp";
    int fd = ::open(tmp_path.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
    if (fd < 0) return false;

    VectorFileHeader hdr{};
    hdr.magic = kVecFileMagic;
    hdr.version = kVecFileVersion;
    hdr.dim = dim;
    hdr.count = count;

    auto write_buf = [fd](const void* buf, size_t len) -> bool {
        const uint8_t* p = static_cast<const uint8_t*>(buf);
        size_t off = 0;
        while (off < len) {
            ssize_t n = ::write(fd, p + off, len - off);
            if (n <= 0) {
                if (n < 0 && errno == EINTR) continue;
                return false;
            }
            off += static_cast<size_t>(n);
        }
        return true;
    };

    bool ok = write_buf(&hdr, sizeof(hdr));
    if (ok && count > 0 && data != nullptr) {
        ok = write_buf(data, static_cast<size_t>(count) * dim * sizeof(float));
    }
    if (ok) {
        ::fsync(fd);
    }
    ::close(fd);
    if (!ok) {
        std::filesystem::remove(tmp_path);
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(tmp_path, path, ec);
    return !ec;
}

MmapVectorFile::~MmapVectorFile() {
    close();
}

bool MmapVectorFile::open(const std::string& path) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;

    struct stat st{};
    if (::fstat(fd_, &st) != 0 || st.st_size < static_cast<off_t>(sizeof(VectorFileHeader))) {
        close();
        return false;
    }
    file_size_ = static_cast<size_t>(st.st_size);
    mapped_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (mapped_ == MAP_FAILED) {
        mapped_ = nullptr;
        close();
        return false;
    }
    const auto* hdr = reinterpret_cast<const VectorFileHeader*>(mapped_);
    if (hdr->magic != kVecFileMagic || hdr->version != kVecFileVersion) {
        close();
        return false;
    }
    const size_t expected = sizeof(VectorFileHeader) + static_cast<size_t>(hdr->count) * hdr->dim * sizeof(float);
    if (file_size_ < expected) {
        close();
        return false;
    }
    dim_ = hdr->dim;
    count_ = hdr->count;
    vectors_ = reinterpret_cast<const float*>(static_cast<const uint8_t*>(mapped_) + sizeof(VectorFileHeader));
    advise_random();
    return true;
}

void MmapVectorFile::close() {
    if (mapped_ != nullptr) {
        ::munmap(mapped_, file_size_);
        mapped_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    vectors_ = nullptr;
    file_size_ = 0;
    dim_ = 0;
    count_ = 0;
}

void MmapVectorFile::advise_random() {
    if (mapped_ != nullptr && file_size_ > 0) {
        ::posix_madvise(mapped_, file_size_, POSIX_MADV_RANDOM);
    }
}

void MmapVectorFile::advise_dontneed() {
    if (mapped_ != nullptr && file_size_ > 0) {
        ::posix_madvise(mapped_, file_size_, POSIX_MADV_DONTNEED);
    }
}

DiskVectorReader::~DiskVectorReader() {
    close();
}

bool DiskVectorReader::open(const std::string& path) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) return false;

    VectorFileHeader hdr{};
    ssize_t n = ::pread(fd_, &hdr, sizeof(hdr), 0);
    if (n != static_cast<ssize_t>(sizeof(hdr)) || hdr.magic != kVecFileMagic || hdr.version != kVecFileVersion) {
        close();
        return false;
    }
    struct stat st{};
    if (::fstat(fd_, &st) == 0) {
        file_bytes_ = static_cast<uint64_t>(st.st_size);
    }
    dim_ = hdr.dim;
    count_ = hdr.count;
    return true;
}

void DiskVectorReader::close() {
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    dim_ = 0;
    count_ = 0;
    file_bytes_ = 0;
}

bool DiskVectorReader::read_slot(uint32_t slot, float* out_vec) const {
    if (fd_ < 0 || slot >= count_ || out_vec == nullptr) return false;
    const size_t bytes = static_cast<size_t>(dim_) * sizeof(float);
    const off_t offset = static_cast<off_t>(sizeof(VectorFileHeader) + static_cast<size_t>(slot) * bytes);
    size_t done = 0;
    uint8_t* dst = reinterpret_cast<uint8_t*>(out_vec);
    while (done < bytes) {
        ssize_t n = ::pread(fd_, dst + done, bytes - done, offset + static_cast<off_t>(done));
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        done += static_cast<size_t>(n);
    }
    return true;
}

BoundedVectorCache::BoundedVectorCache(uint64_t max_bytes) : max_bytes_(max_bytes) {}

void BoundedVectorCache::set_capacity(uint64_t max_bytes) {
    std::lock_guard lock(mutex_);
    max_bytes_ = max_bytes;
    while (used_bytes_ > max_bytes_ && !lru_.empty()) {
        auto& back = lru_.back();
        used_bytes_ -= back.vec.size() * sizeof(float) + 48;
        index_.erase(back.key);
        lru_.pop_back();
    }
}

uint64_t BoundedVectorCache::capacity() const {
    std::lock_guard lock(mutex_);
    return max_bytes_;
}

uint64_t BoundedVectorCache::used_bytes() const {
    std::lock_guard lock(mutex_);
    return used_bytes_;
}

void BoundedVectorCache::invalidate_segment(SegmentId seg_id) {
    std::lock_guard lock(mutex_);
    for (auto it = lru_.begin(); it != lru_.end();) {
        if (it->key.seg_id == seg_id) {
            used_bytes_ -= it->vec.size() * sizeof(float) + 48;
            index_.erase(it->key);
            it = lru_.erase(it);
        } else {
            ++it;
        }
    }
}

void BoundedVectorCache::clear() {
    std::lock_guard lock(mutex_);
    lru_.clear();
    index_.clear();
    used_bytes_ = 0;
}

bool BoundedVectorCache::get_or_read(SegmentId seg_id,
                                     uint32_t slot,
                                     const DiskVectorReader& reader,
                                     std::vector<float>& out,
                                     bool& out_cache_hit,
                                     uint64_t& out_bytes_read) {
    Key key{seg_id, slot};
    {
        std::lock_guard lock(mutex_);
        auto it = index_.find(key);
        if (it != index_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second);
            out = it->second->vec;
            out_cache_hit = true;
            out_bytes_read = 0;
            return true;
        }
    }

    out_cache_hit = false;
    out.resize(reader.dim());
    if (!reader.read_slot(slot, out.data())) {
        return false;
    }
    out_bytes_read = static_cast<uint64_t>(reader.dim()) * sizeof(float);

    if (max_bytes_ == 0) {
        return true;
    }

    std::lock_guard lock(mutex_);
    if (index_.find(key) == index_.end()) {
        uint64_t entry_bytes = out.size() * sizeof(float) + 48;
        if (entry_bytes <= max_bytes_) {
            while (used_bytes_ + entry_bytes > max_bytes_ && !lru_.empty()) {
                auto& back = lru_.back();
                used_bytes_ -= back.vec.size() * sizeof(float) + 48;
                index_.erase(back.key);
                lru_.pop_back();
            }
            lru_.push_front(Entry{key, out});
            index_[key] = lru_.begin();
            used_bytes_ += entry_bytes;
        }
    }
    return true;
}

} // namespace vectordb
