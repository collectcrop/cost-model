#pragma once
#include <cstdint>
#include <cstddef>
#include <memory>
namespace pgm {
typedef enum CacheType {
    SEGMENT,        // cache for segments
    DATA            // cache for data
} CacheType;

typedef enum IOInterface {
    PSYNC,
    LIBAIO,
    IO_URING
} IOInterface;

// typedef enum CacheStrategy {
//     NONE,
//     LRU,
//     LFU,
//     FIFO
// } CacheStrategy;
enum class CachePolicy { NONE, FIFO, LRU, LFU };

typedef enum RangeSearchStrategy {
    LO,
    MID,
    HI
} RangeSearchStrategy;

typedef enum SearchStrategy {
    ALL_IN_ONCE,
    ONE_BY_ONE
} SearchStrategy;

typedef struct Record {
    uint64_t key;
    bool operator<(const Record& other) const {
        return key < other.key;
    }
} Record;

struct IOResult {
    long long ns = 0;           // elapsed nanoseconds
    int64_t bytes = 0;          // total bytes
    size_t logical_ios = 0;     
    size_t physical_ios = 0;   
    int64_t res;
};

struct Page{
    std::shared_ptr<char[]> data;
    size_t valid_len=0;
};

struct RangeQ { uint64_t lo, hi; };

constexpr size_t RECORD_SIZE   = sizeof(Record);
constexpr size_t PAGE_SIZE     = 4096;
constexpr const char* SEGMENT_FILE = "pgm_test_file_seg.bin";
constexpr size_t ITEM_PER_PAGE = PAGE_SIZE / RECORD_SIZE;
constexpr size_t BATCH_SIZE    = 64;   // number of pages to read in one batch
}