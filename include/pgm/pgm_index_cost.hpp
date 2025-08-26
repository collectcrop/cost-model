// This file is part of PGM-index <https://github.com/gvinciguerra/PGM-index>.
// Copyright (c) 2018 Giorgio Vinciguerra.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "piecewise_linear_model.hpp"
#include "cache/FIFOCache.hpp"
#include "cache/LRUCache.hpp"
#include "cache/LFUCache.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>  
#include <unordered_map>
#include <memory>
#include <list>
#include <iostream>
#include <chrono>

using timer = std::chrono::high_resolution_clock;

namespace pgm {
using ::Page;
#define PGM_SUB_EPS(x, epsilon) ((x) <= (epsilon) ? 0 : ((x) - (epsilon)))
#define PGM_ADD_EPS(x, epsilon, size) ((x) + (epsilon) + 2 >= (size) ? (size) : (x) + (epsilon) + 2)
#define PAGE_SIZE 4096
#define SEGMENT_FILE "pgm_test_file_seg.bin"
#define ITEM_PER_PAGE (PAGE_SIZE / RECORD_SIZE)
#define DEBUG true
/**
 * A struct that stores the result of a query to a @ref PGMIndex, that is, a range [@ref lo, @ref hi)
 * centered around an approximate position @ref pos of the sought key.
 */
struct ApproxPos {
    size_t pos; ///< The approximate position of the key.
    size_t lo;  ///< The lower bound of the range.
    size_t hi;  ///< The upper bound of the range.
};

typedef enum CacheType {
    SEGMENT,        // cache for segments
    DATA            // cache for data
} CacheType;

typedef enum CacheStrategy {
    LRU,
    LFU,
    FIFO
} CacheStrategy;

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

struct ApproxPosExt {
    std::vector<Record> records;  // record buffer
    size_t lo;      // minimal pos in buffer
    size_t hi;      // maximal pos in buffer
};
#define RECORD_SIZE sizeof(Record)

/**
 * A space-efficient index that enables fast search operations on a sorted sequence of numbers.
 *
 * A search returns a struct @ref ApproxPos containing an approximate position of the sought key in the sequence and
 * the bounds of a range where the sought key is guaranteed to be found if present.
 * If the key is not present, a @ref std::lower_bound search on the range finds a key that is greater or equal to the
 * sought key, if any.
 * In the case of repeated keys, the index finds the position of the first occurrence of a key.
 *
 * The @p Epsilon template parameter should be set according to the desired space-time trade-off. A smaller value
 * makes the estimation more precise and the range smaller but at the cost of increased space usage.
 *
 * Internally the index uses a succinct piecewise linear mapping from keys to their position in the sorted order.
 * This mapping is represented as a sequence of linear models (segments) which, if @p EpsilonRecursive is not zero, are
 * themselves recursively indexed by other piecewise linear mappings.
 *
 * @tparam K the type of the indexed keys
 * @tparam Epsilon controls the size of the returned search range
 * @tparam EpsilonRecursive controls the size of the search range in the internal structure
 * @tparam Floating the floating-point type to use for slopes
 */
template<typename K, size_t Epsilon = 64, size_t MemoryBudget=1<<23, CacheType type = DATA,
    size_t EpsilonRecursive = 4, typename Floating = float>
class PGMIndex {
protected:
    template<typename, size_t, size_t, uint8_t, typename>
    friend class BucketingPGMIndex;

    template<typename, size_t, typename>
    friend class EliasFanoPGMIndex;

    static_assert(Epsilon > 0);
    struct Segment;

    size_t n;                           ///< The number of elements this index was built on.
    K first_key;                        ///< The smallest element.
    std::vector<Segment> segments;      ///< The segments composing the index.
    std::vector<size_t> levels_offsets; ///< The starting position of each level in segments[], in reverse order.
    int data_fd = -1;
    int seg_fd = -1;
    
    /// Sentinel value to avoid bounds checking.
    static constexpr K sentinel = std::numeric_limits<K>::has_infinity ? std::numeric_limits<K>::infinity()
                                                                       : std::numeric_limits<K>::max();

    template<typename RandomIt>
    static void build(RandomIt first, RandomIt last,
                      size_t epsilon, size_t epsilon_recursive,
                      std::vector<Segment> &segments,
                      std::vector<size_t> &levels_offsets) {
        auto n = (size_t) std::distance(first, last);
        if (n == 0)
            return;

        levels_offsets.push_back(0);
        segments.reserve(n / (epsilon * epsilon));

        if (*std::prev(last) == sentinel)
            throw std::invalid_argument("The value " + std::to_string(sentinel) + " is reserved as a sentinel.");

        auto build_level = [&](auto epsilon, auto in_fun, auto out_fun, size_t last_n) {
            auto n_segments = internal::make_segmentation_par(last_n, epsilon, in_fun, out_fun);
            if (segments.back() == sentinel)
                --n_segments;
            else {
                if (segments.back()(sentinel - 1) < last_n)
                    segments.emplace_back(*std::prev(last) + 1, 0, last_n); // Ensure keys > last are mapped to last_n
                segments.emplace_back(sentinel, 0, last_n);
            }
            return n_segments;
        };

        // Build first level
        auto in_fun = [&](auto i) { return K(first[i]); };
        auto out_fun = [&](auto cs) { segments.emplace_back(cs); };
        auto last_n = build_level(epsilon, in_fun, out_fun, n);
        levels_offsets.push_back(segments.size());

        // Build upper levels
        while (epsilon_recursive && last_n > 1) {
            auto offset = levels_offsets[levels_offsets.size() - 2];
            auto in_fun_rec = [&](auto i) { return segments[offset + i].key; };
            last_n = build_level(epsilon_recursive, in_fun_rec, out_fun, last_n);
            levels_offsets.push_back(segments.size());
        }

        //  Write segments to file
        if (std::ofstream ofs{SEGMENT_FILE, std::ios::binary | std::ios::trunc}) {
            for (auto segment : segments) {
                ofs.write(reinterpret_cast<const char*>(&segment), sizeof(segment));
            }
        } else {
            throw std::runtime_error("Failed to write segment file.");
        }
    }

    Segment* get_segment_ptr(size_t segment_index) const {
        size_t page_index = segment_index * sizeof(Segment) / PAGE_SIZE;
        size_t offset = (segment_index * sizeof(Segment)) % PAGE_SIZE;

        Page& page = seg_cache->get(page_index);
        return reinterpret_cast<Segment*>(page.data.get() + offset);
    }
    /**
     * Returns the segment responsible for a given key, that is, the rightmost segment having key <= the sought key.
     * @param key the value of the element to search for
     * @return an iterator to the segment responsible for the given key
     */
    auto segment_for_key(const K &key) const {
        if constexpr (EpsilonRecursive == 0) {
            return std::prev(std::upper_bound(segments.begin(), segments.begin() + segments_count(), key));
        }

        auto it = segments.begin() + *(levels_offsets.end() - 2);
        for (auto l = int(height()) - 2; l >= 0; --l) {
            auto level_begin = segments.begin() + levels_offsets[l];
            auto pos = std::min<size_t>((*it)(key), std::next(it)->intercept);
            auto lo = level_begin + PGM_SUB_EPS(pos, EpsilonRecursive + 1);

            static constexpr size_t linear_search_threshold = 8 * 64 / sizeof(Segment);
            if constexpr (EpsilonRecursive <= linear_search_threshold) {
                for (; std::next(lo)->key <= key; ++lo)
                    continue;
                it = lo;
            } else {
                auto level_size = levels_offsets[l + 1] - levels_offsets[l] - 1;
                auto hi = level_begin + PGM_ADD_EPS(pos, EpsilonRecursive, level_size);
                it = std::prev(std::upper_bound(lo, hi, key));
            }
        }
        return it;    
    }

    auto segment_for_key_disk(const K &key) const {
        if constexpr (EpsilonRecursive == 0) {
            // Non-recursive PGM
            size_t begin = 0, end = segments_count();  
            size_t lo = begin, hi = end;

            while (lo < hi) {
                size_t mid = (lo + hi) / 2;
                Segment* s = get_segment_ptr(mid);
                if (s->key <= key) lo = mid + 1;
                else hi = mid;
            }
            return get_segment_ptr(lo - 1);
        }
        // Recursive PGM
        size_t level = height() - 2;
        size_t seg_id = *(levels_offsets.end() - 2);  // root segment index

        Segment* seg = get_segment_ptr(seg_id);
        Segment* next;
        while (true) {
            next = get_segment_ptr(seg_id + 1);
            size_t pos = std::min<size_t>((*seg)(key), next->intercept);

            size_t level_start = levels_offsets[level];
            size_t lo_index = level_start + PGM_SUB_EPS(pos, EpsilonRecursive + 1);

            constexpr size_t linear_threshold = 8 * 64 / sizeof(Segment);

            if constexpr (EpsilonRecursive <= linear_threshold) {
                while (get_segment_ptr(lo_index+1)->key <= key) {
                    ++lo_index;
                }
                seg_id = lo_index;
            } else {
                size_t level_size = levels_offsets[level + 1] - level_start - 1;
                size_t hi_index = level_start + PGM_ADD_EPS(pos, EpsilonRecursive, level_size);

                // Binary search in [lo_index, hi_index)
                size_t lo = lo_index, hi = hi_index;
                while (lo < hi) {
                    size_t mid = (lo + hi) / 2;
                    if (get_segment_ptr(mid)->key <= key)
                        lo = mid + 1;
                    else
                        hi = mid;
                }
                seg_id = lo - 1;
            }

                if (level == 0) break;
                seg = get_segment_ptr(seg_id);
                level--;
            }

            return seg_id;
    }

public:

    static constexpr size_t epsilon_value = Epsilon;
    // mutable PageCache cache;
    // mutable std::shared_ptr<PageCache> cache,seg_cache;
    mutable std::shared_ptr<ICache<size_t, pgm::Page>> cache,seg_cache;

    /**
     * Constructs an empty index.
     */
    PGMIndex() = default;

    /**
     * Constructs the index on the given sorted vector.
     * @param data the vector of keys to be indexed, must be sorted
     */
    explicit PGMIndex(const std::vector<K> &data, std::string filename, CacheStrategy strategy=LFU, double factor=0.01) 
    : PGMIndex(data.begin(), data.end(), filename, strategy, factor) {}

    /**
     * Constructs the index on the sorted keys in the range [first, last).
     * @param first, last the range containing the sorted keys to be indexed
     */
    template<typename RandomIt>
    PGMIndex(RandomIt first, RandomIt last, std::string filename, CacheStrategy strategy=LFU, double factor = 0.01)
        : n(std::distance(first, last)),
          first_key(n ? *first : K(0)),
          segments(),
          levels_offsets() {
        data_fd = open(filename.c_str(),O_RDONLY|O_DIRECT);      // test for real IO time
        // data_fd = open(filename.c_str(),O_RDONLY);                  // test for IOs
        if (data_fd < 0)
            throw std::runtime_error("Cannot open data file");
        build(first, last, Epsilon, EpsilonRecursive, segments, levels_offsets);
        seg_fd = open(SEGMENT_FILE,O_RDONLY|O_DIRECT);
        if (seg_fd < 0)
            throw std::runtime_error("Cannot open segment file");
        
        if (type==SEGMENT){     // ||(ssize_t)(MemoryBudget-n*sizeof(Segment)/(2*Epsilon))<=0
            std::cout<<"Segment cache"<<std::endl;
            if (strategy == LRU) {
                cache = std::make_unique<LRUCache>((1-factor)*MemoryBudget/PAGE_SIZE, data_fd);
                seg_cache = std::make_unique<LRUCache>(factor*MemoryBudget/PAGE_SIZE, seg_fd);
            }else if (strategy == FIFO){
                cache = std::make_unique<FIFOCache>((1-factor)*MemoryBudget/PAGE_SIZE, data_fd);
                seg_cache = std::make_unique<FIFOCache>(factor*MemoryBudget/PAGE_SIZE, seg_fd);
            }else if (strategy == LFU){
                cache = std::make_unique<LFUCache>((1-factor)*MemoryBudget/PAGE_SIZE, data_fd);
                seg_cache = std::make_unique<LFUCache>(factor*MemoryBudget/PAGE_SIZE, seg_fd);
            }
        }
        else{
            if ((ssize_t)(MemoryBudget-n*sizeof(Segment)/(2*Epsilon))<=0){
                std::cout << "memory overflow!" << std::endl;
            }
            if (strategy == LRU) {
                cache = std::make_unique<LRUCache>((MemoryBudget-n*sizeof(Segment)/(2*Epsilon))/PAGE_SIZE, data_fd);
                seg_cache = std::make_unique<LRUCache>(factor*MemoryBudget/PAGE_SIZE, seg_fd);
            }else if (strategy == FIFO) { 
                cache = std::make_unique<FIFOCache>((MemoryBudget-n*sizeof(Segment)/(2*Epsilon))/PAGE_SIZE, data_fd);
                seg_cache = std::make_unique<FIFOCache>(factor*MemoryBudget/PAGE_SIZE, seg_fd);
            }else if (strategy == LFU){
                cache = std::make_unique<LFUCache>((MemoryBudget-n*sizeof(Segment)/(2*Epsilon))/PAGE_SIZE, data_fd);
                seg_cache = std::make_unique<LFUCache>(factor*MemoryBudget/PAGE_SIZE, seg_fd);
            }

        } 

    }

    ~PGMIndex() {
        if (data_fd >= 0) close(data_fd);
        if (seg_fd >= 0) close(seg_fd);
    }

    /**
     * Returns the approximate position and the range where @p key can be found.
     * @param key the value of the element to search for
     * @return a record vector with the approximate position and bounds of the range
     */
    ApproxPosExt search(const K &key, SearchStrategy s=ONE_BY_ONE)const {
        auto k = std::max(first_key, key);\
        size_t pos;
        if (type==DATA){
            auto it = segment_for_key(k);
            pos = std::min<size_t>((*it)(k), std::next(it)->intercept);
        }else {
            auto idx = segment_for_key_disk(k);
            pos = std::min<size_t>((*get_segment_ptr(idx))(k),get_segment_ptr(idx+1)->intercept);
        }
        auto lo = PGM_SUB_EPS(pos, Epsilon);
        auto hi = PGM_ADD_EPS(pos, Epsilon, n);

        // all-in-once
        if (s == ALL_IN_ONCE) {
            std::vector<Record> buffer;
            std::vector<Page> pages = cache->get(lo / ITEM_PER_PAGE, hi / ITEM_PER_PAGE);

            for (auto &page : pages) {
                size_t num_records = page.valid_len / sizeof(Record);   // valid record num
                Record* records = reinterpret_cast<Record*>(page.data.get());
                buffer.insert(buffer.end(), records, records + num_records);
            }

            size_t page_lo = lo / ITEM_PER_PAGE;
            size_t rel_lo = lo % ITEM_PER_PAGE;
            size_t rel_hi = std::min(hi - page_lo * ITEM_PER_PAGE, buffer.size());

            return {std::move(buffer), rel_lo, rel_hi};
        }
        else{       // one-by-one
            Page page;
            Record* records = nullptr;
            for (size_t pageIndex=lo/ITEM_PER_PAGE;pageIndex<=hi/ITEM_PER_PAGE;pageIndex++){
                page = cache->get(pageIndex);
                records = reinterpret_cast<Record*>(page.data.get());
                if (records[page.valid_len/sizeof(Record) - 1].key>=key)    break;
            }
            size_t num_records = page.valid_len / sizeof(Record);
            std::vector<Record> buffer(records, records + num_records);
            size_t size = buffer.size(); 
            return {std::move(buffer), 0, size};
        }
    }

    /**
     * Returns all keys in provided range.
     * @param lo the target keys lower bound
     * @param hi the target keys upper bound
     * @return a vector contains all keys in provided range.
     */
    std::vector<K> range_search(const K &lo, const K &hi, RangeSearchStrategy s=LO){
        std::vector<K> res;
        if (hi<first_key) return res;
        // one-by-one strategy
        K tar;
        switch(s){
            case MID:
                tar = (lo+hi)/2;
                break;
            case LO:
                tar = lo;
                break;
            case HI:
                tar = hi;
                break;
        }
        auto it = segment_for_key(tar);
        size_t pos = std::min<size_t>((*it)(tar), std::next(it)->intercept);
        size_t pageIndex = pos/ITEM_PER_PAGE;
        Page* page = &cache->get(pageIndex);
        K l,r;
        std::pair<K,K> lr;
        lr = inner_search(*page,res,lo,hi);l=lr.first;r=lr.second;
        size_t p = pageIndex;
        while (l==0&&p>0){
            page = &cache->get(--p);
            lr = inner_search(*page,res,lo,hi);
            l = lr.first;
        }
        p = pageIndex;
        while (r==ITEM_PER_PAGE-1){
            page = &cache->get(++p);
            lr = inner_search(*page,res,lo,hi);
            r = lr.second;
        }

        // all-in-once strategy
        // auto it_lo = segment_for_key(lo);
        // auto it_hi = segment_for_key(hi);
        // size_t pos_lo = std::min<size_t>((*it_lo)(lo), std::next(it_lo)->intercept);
        // size_t pos_hi = std::min<size_t>((*it_hi)(hi), std::next(it_hi)->intercept);
        // size_t page_lo = std::max((size_t)0,pos_lo-Epsilon)/ITEM_PER_PAGE;
        // size_t page_hi = std::min(n,pos_hi+Epsilon)/ITEM_PER_PAGE;
        // std::vector<Page> pages = cache->get(page_lo,page_hi);

        // for (auto &page:pages){
        //     if (reinterpret_cast<Record*>(page.data.get())[page.valid_len / sizeof(Record) - 1].key<lo) continue;
        //     else if (reinterpret_cast<Record*>(page.data.get())[0].key>hi) break;
        //     for (size_t i = 0; i < page.valid_len / sizeof(Record); i++) {
        //         K key = reinterpret_cast<Record*>(page.data.get())[i].key;
        //         if (key >= lo && key <= hi) {
        //             res.push_back(key);
        //         }
        //     }
        // }
        return res;
    }

    std::pair<K,K> inner_search(Page &p, std::vector<K>& res, K lo, K hi){
        Record* recs = reinterpret_cast<Record*>(p.data.get());
        Record* begin = recs;
        Record* end   = recs + p.valid_len / sizeof(Record);

        Record* lb = std::lower_bound(begin, end, lo, 
            [](const Record& r, K key) { return r.key < key; });
        Record* ub = std::upper_bound(begin, end, hi, 
            [](K key, const Record& r) { return key < r.key; });

        for (auto it=lb;it<ub;it++){
            res.push_back(it->key);
        }
        return {lb-begin,ub-begin-1};
    }
    /**
     * Returns the number of segments in the last level of the index.
     * @return the number of segments
     */
    size_t segments_count() const { return segments.empty() ? 0 : levels_offsets[1] - 1; }

    /**
     * Returns the number of levels of the index.
     * @return the number of levels of the index
     */
    size_t height() const { return levels_offsets.size() - 1; }

    /**
     * Returns the size of the index in bytes.
     * @return the size of the index in bytes
     */
    size_t size_in_bytes() const { return segments.size() * sizeof(Segment); }  // + levels_offsets.size() * sizeof(size_t)

    /**
     * Returns the level offsets.
     * @return the level offsets
     */
    const std::vector<size_t> &get_levels_offsets() const { return levels_offsets; }
    const size_t get_segment_size() const { return sizeof(Segment); }
    const auto get_data_cache() const { return cache.get(); }
    const auto get_index_cache() const { return seg_cache.get(); }
    // const size_t get_IO_time() const { return IO_time; }
};


#pragma pack(push, 1)

template<typename K, size_t Epsilon, size_t MemoryBudget, CacheType type,
    size_t EpsilonRecursive, typename Floating>
struct PGMIndex<K, Epsilon, MemoryBudget, type, EpsilonRecursive, Floating>::Segment {
    K key;              ///< The first key that the segment indexes.
    Floating slope;     ///< The slope of the segment.
    uint32_t intercept; ///< The intercept of the segment.

    Segment() = default;

    Segment(K key, Floating slope, uint32_t intercept) : key(key), slope(slope), intercept(intercept) {};

    explicit Segment(const typename internal::OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment &cs)
        : key(cs.get_first_x()) {
        auto[cs_slope, cs_intercept] = cs.get_floating_point_segment(key);
        if (cs_intercept > std::numeric_limits<decltype(intercept)>::max())
            throw std::overflow_error("Change the type of Segment::intercept to uint64");
        if (cs_intercept < 0)
            throw std::overflow_error("Unexpected intercept < 0");
        slope = cs_slope;
        intercept = cs_intercept;
    }

    friend inline bool operator<(const Segment &s, const K &k) { return s.key < k; }
    friend inline bool operator<(const K &k, const Segment &s) { return k < s.key; }
    friend inline bool operator<(const Segment &s, const Segment &t) { return s.key < t.key; }

    operator K() { return key; };

    /**
     * Returns the approximate position of the specified key.
     * @param k the key whose position must be approximated
     * @return the approximate position of the specified key
     */
    inline size_t operator()(const K &k) const {
        size_t pos;
        if constexpr (std::is_same_v<K, int64_t> || std::is_same_v<K, int32_t>)
            pos = size_t(slope * double(std::make_unsigned_t<K>(k) - key));
        else
            pos = size_t(slope * double(k - key));
        return pos + intercept;
    }
};

#pragma pack(pop)

}