#include "vectordb/wal.hpp"
#include <algorithm>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace vectordb {

namespace {
constexpr uint32_t kWalMagic = 0x57414C31; // "WAL1"

#pragma pack(push, 1)
struct WalHeader {
    uint32_t magic;
    uint64_t lsn;
    uint8_t op;
    uint64_t id;
    uint32_t dim;
};
#pragma pack(pop)

bool write_all(int fd, const uint8_t* buf, size_t len) {
    size_t offset = 0;
    while (offset < len) {
        ssize_t n = ::write(fd, buf + offset, len - offset);
        if (n <= 0) {
            if (n < 0 && errno == EINTR) continue;
            return false;
        }
        offset += static_cast<size_t>(n);
    }
    return true;
}

bool read_exact(int fd, uint8_t* buf, size_t len) {
    size_t offset = 0;
    while (offset < len) {
        ssize_t n = ::read(fd, buf + offset, len - offset);
        if (n == 0) return false;
        if (n < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        offset += static_cast<size_t>(n);
    }
    return true;
}
} // namespace

uint32_t crc32(const uint8_t* data, size_t len) noexcept {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= data[i];
        for (int j = 0; j < 8; ++j) {
            uint32_t mask = -(crc & 1u);
            crc = (crc >> 1) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

WriteAheadLog::~WriteAheadLog() {
    close();
}

bool WriteAheadLog::open(const std::string& path, FsyncPolicy fsync_policy) {
    std::lock_guard lock(mutex_);
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    path_ = path;
    fsync_policy_ = fsync_policy;
    auto parent = std::filesystem::path(path_).parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
    fd_ = ::open(path_.c_str(), O_CREAT | O_RDWR | O_APPEND, 0644);
    if (fd_ < 0) return false;

    struct stat st{};
    if (::fstat(fd_, &st) == 0) {
        bytes_written_ = static_cast<uint64_t>(st.st_size);
    }
    return true;
}

void WriteAheadLog::close() {
    std::lock_guard lock(mutex_);
    if (fd_ >= 0) {
        ::fsync(fd_);
        ::close(fd_);
        fd_ = -1;
    }
}

Lsn WriteAheadLog::write_record(WalOp op, VectorId id, const float* vec, uint32_t dim, Lsn checkpoint_target) {
    std::lock_guard lock(mutex_);
    if (fd_ < 0) {
        throw std::runtime_error("WAL is not open");
    }
    Lsn lsn = next_lsn_++;
    WalHeader hdr{
        kWalMagic,
        lsn,
        static_cast<uint8_t>(op),
        (op == WalOp::CHECKPOINT) ? checkpoint_target : id,
        dim
    };

    const size_t payload_bytes = static_cast<size_t>(dim) * sizeof(float);
    const size_t body_bytes = sizeof(WalHeader) + payload_bytes;
    std::vector<uint8_t> frame(body_bytes + sizeof(uint32_t));
    std::memcpy(frame.data(), &hdr, sizeof(WalHeader));
    if (payload_bytes > 0 && vec != nullptr) {
        std::memcpy(frame.data() + sizeof(WalHeader), vec, payload_bytes);
    }
    uint32_t checksum = crc32(frame.data(), body_bytes);
    std::memcpy(frame.data() + body_bytes, &checksum, sizeof(uint32_t));

    if (!write_all(fd_, frame.data(), frame.size())) {
        throw std::runtime_error("Failed to write WAL record");
    }
    bytes_written_ += frame.size();
    ++records_written_;
    ++pending_since_sync_;

    if (fsync_policy_ == FsyncPolicy::ALWAYS ||
        (fsync_policy_ == FsyncPolicy::BATCHED && pending_since_sync_ >= 64)) {
        ::fsync(fd_);
        pending_since_sync_ = 0;
    }
    return lsn;
}

Lsn WriteAheadLog::append_upsert(VectorId id, const std::vector<float>& vec) {
    return write_record(WalOp::UPSERT, id, vec.data(), static_cast<uint32_t>(vec.size()));
}

Lsn WriteAheadLog::append_delete(VectorId id) {
    return write_record(WalOp::DELETE, id, nullptr, 0);
}

Lsn WriteAheadLog::append_checkpoint(Lsn flushed_lsn) {
    Lsn lsn = write_record(WalOp::CHECKPOINT, 0, nullptr, 0, flushed_lsn);
    sync();
    return lsn;
}

void WriteAheadLog::sync() {
    std::lock_guard lock(mutex_);
    if (fd_ >= 0) {
        ::fsync(fd_);
        pending_since_sync_ = 0;
    }
}

size_t WriteAheadLog::replay(const std::function<void(const WalRecord&)>& visitor,
                             Lsn& out_max_lsn,
                             Lsn& out_checkpoint_lsn) {
    std::lock_guard lock(mutex_);
    out_max_lsn = 0;
    out_checkpoint_lsn = 0;
    if (fd_ < 0) return 0;

    int rfd = ::open(path_.c_str(), O_RDONLY);
    if (rfd < 0) return 0;

    struct stat st{};
    if (::fstat(rfd, &st) != 0) {
        ::close(rfd);
        return 0;
    }
    const off_t file_size = st.st_size;
    off_t valid_end = 0;
    size_t count = 0;

    while (valid_end + static_cast<off_t>(sizeof(WalHeader) + sizeof(uint32_t)) <= file_size) {
        WalHeader hdr{};
        if (!read_exact(rfd, reinterpret_cast<uint8_t*>(&hdr), sizeof(WalHeader))) {
            break;
        }
        if (hdr.magic != kWalMagic || hdr.dim > (1u << 20)) {
            break;
        }
        const size_t payload_bytes = static_cast<size_t>(hdr.dim) * sizeof(float);
        const off_t record_total = static_cast<off_t>(sizeof(WalHeader) + payload_bytes + sizeof(uint32_t));
        if (valid_end + record_total > file_size) {
            break;
        }

        std::vector<uint8_t> body(sizeof(WalHeader) + payload_bytes);
        std::memcpy(body.data(), &hdr, sizeof(WalHeader));
        if (payload_bytes > 0) {
            if (!read_exact(rfd, body.data() + sizeof(WalHeader), payload_bytes)) {
                break;
            }
        }
        uint32_t stored_crc = 0;
        if (!read_exact(rfd, reinterpret_cast<uint8_t*>(&stored_crc), sizeof(uint32_t))) {
            break;
        }
        uint32_t computed_crc = crc32(body.data(), body.size());
        if (stored_crc != computed_crc) {
            break;
        }

        WalRecord rec;
        rec.lsn = hdr.lsn;
        rec.op = static_cast<WalOp>(hdr.op);
        rec.id = hdr.id;
        if (hdr.dim > 0) {
            rec.vec.resize(hdr.dim);
            std::memcpy(rec.vec.data(), body.data() + sizeof(WalHeader), payload_bytes);
        }

        out_max_lsn = std::max(out_max_lsn, rec.lsn);
        if (rec.op == WalOp::CHECKPOINT) {
            out_checkpoint_lsn = std::max(out_checkpoint_lsn, rec.id);
        }
        if (visitor) {
            visitor(rec);
        }

        valid_end += record_total;
        ++count;
    }

    ::close(rfd);

    // Truncate torn/corrupted tail if detected
    if (valid_end < file_size) {
        if (::ftruncate(fd_, valid_end) == 0) {
            ::fsync(fd_);
            bytes_written_ = static_cast<uint64_t>(valid_end);
        }
    }
    next_lsn_ = out_max_lsn + 1;
    return count;
}

bool WriteAheadLog::truncate_and_checkpoint(Lsn checkpoint_lsn) {
    std::lock_guard lock(mutex_);
    if (fd_ < 0) return false;
    ::close(fd_);
    fd_ = ::open(path_.c_str(), O_CREAT | O_RDWR | O_TRUNC | O_APPEND, 0644);
    if (fd_ < 0) return false;
    bytes_written_ = 0;
    pending_since_sync_ = 0;

    // Write fresh checkpoint marker as initial record
    Lsn lsn = std::max(next_lsn_, checkpoint_lsn + 1);
    next_lsn_ = lsn + 1;
    WalHeader hdr{kWalMagic, lsn, static_cast<uint8_t>(WalOp::CHECKPOINT), checkpoint_lsn, 0};
    std::vector<uint8_t> frame(sizeof(WalHeader) + sizeof(uint32_t));
    std::memcpy(frame.data(), &hdr, sizeof(WalHeader));
    uint32_t checksum = crc32(frame.data(), sizeof(WalHeader));
    std::memcpy(frame.data() + sizeof(WalHeader), &checksum, sizeof(uint32_t));
    if (!write_all(fd_, frame.data(), frame.size())) return false;
    ::fsync(fd_);
    bytes_written_ = frame.size();
    return true;
}

uint64_t WriteAheadLog::bytes_written() const {
    std::lock_guard lock(mutex_);
    return bytes_written_;
}

uint64_t WriteAheadLog::records_written() const {
    std::lock_guard lock(mutex_);
    return records_written_;
}

Lsn WriteAheadLog::current_lsn() const {
    std::lock_guard lock(mutex_);
    return next_lsn_ > 0 ? next_lsn_ - 1 : 0;
}

} // namespace vectordb
