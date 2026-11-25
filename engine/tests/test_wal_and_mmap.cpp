#include <gtest/gtest.h>
#include "vectordb/mmap_storage.hpp"
#include "vectordb/wal.hpp"
#include <filesystem>
#include <fstream>
#include <vector>

using namespace vectordb;

namespace {
struct TempDir {
    std::string path;
    explicit TempDir(const std::string& name) {
        path = "/tmp/" + name;
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path);
    }
    ~TempDir() {
        std::filesystem::remove_all(path);
    }
};
} // namespace

TEST(WalTest, AppendReplayAndCrashRecoveryTruncatesTornTail) {
    TempDir dir("wal_test_1");
    const std::string wal_path = dir.path + "/wal.log";

    {
        WriteAheadLog wal;
        ASSERT_TRUE(wal.open(wal_path, FsyncPolicy::ALWAYS));
        EXPECT_EQ(wal.append_upsert(1, {1.0f, 2.0f, 3.0f, 4.0f}), 1u);
        EXPECT_EQ(wal.append_upsert(2, {5.0f, 6.0f, 7.0f, 8.0f}), 2u);
        EXPECT_EQ(wal.append_delete(1), 3u);
        wal.sync();
    }

    // Simulate crash leaving partial garbage bytes at the end of WAL
    auto size_before = std::filesystem::file_size(wal_path);
    {
        std::ofstream os(wal_path, std::ios::binary | std::ios::app);
        const char garbage[13] = {'X', 'Y', 'Z', 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
        os.write(garbage, sizeof(garbage));
    }
    ASSERT_GT(std::filesystem::file_size(wal_path), size_before);

    std::vector<WalRecord> replayed;
    Lsn max_lsn = 0, cp_lsn = 0;
    {
        WriteAheadLog wal;
        ASSERT_TRUE(wal.open(wal_path, FsyncPolicy::BATCHED));
        size_t n = wal.replay([&](const WalRecord& rec) { replayed.push_back(rec); }, max_lsn, cp_lsn);
        EXPECT_EQ(n, 3u);
        EXPECT_EQ(max_lsn, 3u);
        EXPECT_EQ(cp_lsn, 0u);
        // Torn tail must have been cleanly truncated back to valid boundary
        EXPECT_EQ(std::filesystem::file_size(wal_path), size_before);

        // Appending after recovery continues with LSN 4
        EXPECT_EQ(wal.append_upsert(3, {9.0f, 10.0f, 11.0f, 12.0f}), 4u);
    }

    ASSERT_EQ(replayed.size(), 3u);
    EXPECT_EQ(replayed[0].op, WalOp::UPSERT);
    EXPECT_EQ(replayed[0].id, 1u);
    EXPECT_EQ(replayed[2].op, WalOp::DELETE);
    EXPECT_EQ(replayed[2].id, 1u);

    // Checkpoint truncation test
    {
        WriteAheadLog wal;
        ASSERT_TRUE(wal.open(wal_path, FsyncPolicy::ALWAYS));
        ASSERT_TRUE(wal.truncate_and_checkpoint(4));
        replayed.clear();
        size_t n = wal.replay([&](const WalRecord& rec) { replayed.push_back(rec); }, max_lsn, cp_lsn);
        EXPECT_EQ(n, 1u);
        EXPECT_EQ(cp_lsn, 4u);
        EXPECT_EQ(replayed[0].op, WalOp::CHECKPOINT);
    }
}

TEST(MmapAndCacheTest, WriteMmapPreadAndLruCache) {
    TempDir dir("mmap_test_1");
    const std::string vec_path = dir.path + "/seg_0001.vec";

    constexpr uint32_t count = 32;
    constexpr uint32_t dim = 8;
    std::vector<float> vecs(count * dim);
    for (uint32_t i = 0; i < count; ++i) {
        for (uint32_t d = 0; d < dim; ++d) {
            vecs[i * dim + d] = static_cast<float>(i * 10 + d);
        }
    }
    ASSERT_TRUE(write_vector_file(vec_path, vecs.data(), count, dim));

    MmapVectorFile mmap_file;
    ASSERT_TRUE(mmap_file.open(vec_path));
    EXPECT_EQ(mmap_file.count(), count);
    EXPECT_EQ(mmap_file.dim(), dim);
    for (uint32_t i = 0; i < count; ++i) {
        EXPECT_FLOAT_EQ(mmap_file.vectors()[i * dim + 3], static_cast<float>(i * 10 + 3));
    }
    mmap_file.advise_dontneed();
    mmap_file.close();

    DiskVectorReader reader;
    ASSERT_TRUE(reader.open(vec_path));
    // Cache capacity allows ~4 vectors (each 8*4 + 48 = 80 bytes)
    BoundedVectorCache cache(320);
    std::vector<float> out;
    bool hit = false;
    uint64_t bytes = 0;

    ASSERT_TRUE(cache.get_or_read(1, 5, reader, out, hit, bytes));
    EXPECT_FALSE(hit);
    EXPECT_EQ(bytes, dim * sizeof(float));
    EXPECT_FLOAT_EQ(out[0], 50.0f);

    // Second read of slot 5 should hit cache with 0 bytes disk I/O
    ASSERT_TRUE(cache.get_or_read(1, 5, reader, out, hit, bytes));
    EXPECT_TRUE(hit);
    EXPECT_EQ(bytes, 0u);

    // Fill cache past capacity to verify LRU eviction
    for (uint32_t s = 10; s < 20; ++s) {
        ASSERT_TRUE(cache.get_or_read(1, s, reader, out, hit, bytes));
    }
    EXPECT_LE(cache.used_bytes(), 320u);

    // Slot 5 should now be evicted
    ASSERT_TRUE(cache.get_or_read(1, 5, reader, out, hit, bytes));
    EXPECT_FALSE(hit);
}
