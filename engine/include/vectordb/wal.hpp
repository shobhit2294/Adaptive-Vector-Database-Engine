#pragma once

#include "vectordb/types.hpp"
#include <cstdint>
#include <fstream>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace vectordb {

enum class WalOp : uint8_t {
    UPSERT = 1,
    DELETE = 2,
    CHECKPOINT = 3
};

struct WalRecord {
    Lsn lsn = 0;
    WalOp op = WalOp::UPSERT;
    VectorId id = 0;
    std::vector<float> vec;
};

uint32_t crc32(const uint8_t* data, size_t len) noexcept;

class WriteAheadLog {
public:
    WriteAheadLog() = default;
    ~WriteAheadLog();

    WriteAheadLog(const WriteAheadLog&) = delete;
    WriteAheadLog& operator=(const WriteAheadLog&) = delete;

    bool open(const std::string& path, FsyncPolicy fsync_policy = FsyncPolicy::BATCHED);
    void close();

    Lsn append_upsert(VectorId id, const std::vector<float>& vec);
    Lsn append_delete(VectorId id);
    Lsn append_checkpoint(Lsn flushed_lsn);

    void sync();

    // Replays valid records from WAL; truncates any torn/corrupted tail automatically.
    // Returns number of records replayed and sets max_lsn / last_checkpoint_lsn.
    size_t replay(const std::function<void(const WalRecord&)>& visitor,
                  Lsn& out_max_lsn,
                  Lsn& out_checkpoint_lsn);

    // Resets WAL after all state has been durably checkpointed to segments.
    bool truncate_and_checkpoint(Lsn checkpoint_lsn);

    uint64_t bytes_written() const;
    uint64_t records_written() const;
    Lsn current_lsn() const;
    const std::string& path() const { return path_; }

private:
    std::string path_;
    FsyncPolicy fsync_policy_ = FsyncPolicy::BATCHED;
    int fd_ = -1;
    mutable std::mutex mutex_;
    Lsn next_lsn_ = 1;
    uint64_t bytes_written_ = 0;
    uint64_t records_written_ = 0;
    uint32_t pending_since_sync_ = 0;

    Lsn write_record(WalOp op, VectorId id, const float* vec, uint32_t dim, Lsn checkpoint_target = 0);
};

} // namespace vectordb
