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
#include "utils/include.hpp"
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
#define PGM_SUB_EPS(x, epsilon) ((x) <= (epsilon) ? 0 : ((x) - (epsilon)))
#define PGM_ADD_EPS(x, epsilon, size) ((x) + (epsilon) + 2 >= (size) ? (size) : (x) + (epsilon) + 2)


/**
 * A struct that stores the result of a query to a @ref PGMIndexCost, that is, a range [@ref lo, @ref hi)
 * centered around an approximate position @ref pos of the sought key.
 */
struct ApproxPos {
    size_t pos; ///< The approximate position of the key.
    size_t lo;  ///< The lower bound of the range.
    size_t hi;  ///< The upper bound of the range.
};

struct ApproxPosExt {
    std::vector<Record> records;  // record buffer
    size_t lo;      // minimal pos in buffer
    size_t hi;      // maximal pos in buffer
};


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
template<typename K, size_t Epsilon = 64, size_t MemoryBudget=1<<23,
    size_t EpsilonRecursive = 4, typename Floating = float>
class PGMIndexCost {
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

        pgm::Page& page = seg_cache->get(page_index);
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


public:

    static constexpr size_t epsilon_value = Epsilon;
    // mutable PageCache cache;
    // mutable std::shared_ptr<PageCache> cache,seg_cache;
    mutable std::shared_ptr<ICache<size_t, pgm::Page>> cache,seg_cache;

    /**
     * Constructs an empty index.
     */
    PGMIndexCost() = default;

    /**
     * Constructs the index on the given sorted vector.
     * @param data the vector of keys to be indexed, must be sorted
     */
    explicit PGMIndexCost(const std::vector<K> &data, std::string filename, CacheStrategy strategy=LRU, IOInterface interface=pgm::PSYNC, threads=1) 
    : PGMIndexCost(data.begin(), data.end(), filename, strategy, interface, threads) {}

    /**
     * Constructs the index on the sorted keys in the range [first, last).
     * @param first, last the range containing the sorted keys to be indexed
     */
    template<typename RandomIt>
    PGMIndexCost(RandomIt first, RandomIt last, std::string filename, CacheStrategy strategy=LRU, IOInterface interface=pgm::PSYNC, threads=1)
        : n(std::distance(first, last)),
          first_key(n ? *first : K(0)),
          segments(),
          levels_offsets() {
        data_fd = open(filename.c_str(),O_RDONLY|O_DIRECT);      // test for real IO time
        // data_fd = open(filename.c_str(),O_RDONLY);                  // test for IOs
        if (data_fd < 0)
            throw std::runtime_error("Cannot open data file");
        build(first, last, Epsilon, EpsilonRecursive, segments, levels_offsets);

        if ((ssize_t)(MemoryBudget-n*sizeof(Segment)/(2*Epsilon))<=0){
            std::cout << "memory overflow!" << std::endl;
        }
        if (strategy == LRU) {
            cache = std::make_unique<LRUCache>((MemoryBudget-size_in_bytes())/PAGE_SIZE, data_fd, interface);
        }else if (strategy == FIFO) { 
            cache = std::make_unique<FIFOCache>((MemoryBudget-size_in_bytes())/PAGE_SIZE, data_fd, interface);
        }else if (strategy == LFU){
            cache = std::make_unique<LFUCache>((MemoryBudget-size_in_bytes())/PAGE_SIZE, data_fd, interface);
        }
    }

    ~PGMIndexCost() {
        if (data_fd >= 0) close(data_fd);
    }

    /**
     * Returns the approximate position and the range where @p key can be found.
     * @param key the value of the element to search for
     * @return a record vector with the approximate position and bounds of the range
     */
    ApproxPosExt search(const K &key, SearchStrategy s=ONE_BY_ONE)const {
        auto k = std::max(first_key, key);
        size_t pos;
        // auto t0 = timer::now();
        
        auto it = segment_for_key(k);
        pos = std::min<size_t>((*it)(k), std::next(it)->intercept);
        
        // auto t1 = timer::now();
        // std::cout << "Index time:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << std::endl;
        auto lo = PGM_SUB_EPS(pos, Epsilon);
        auto hi = PGM_ADD_EPS(pos, Epsilon, n);

        // all-at-once
        if (s == ALL_IN_ONCE) {
            std::vector<Record> buffer;
            // auto t0 = timer::now();
            std::vector<pgm::Page> pages = cache->get(lo / ITEM_PER_PAGE, hi / ITEM_PER_PAGE);
            // auto t1 = timer::now();
            // std::cout << "Cache time:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << std::endl;
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
            pgm::Page page;
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
     * Aggragate search for multiple keys.
     */
    // std::vector<ApproxPosExt> search(const std::vector<K> &keys, SearchStrategy s=ONE_BY_ONE)const {
    //     std::vector<ApproxPosExt> res;
    //     if (keys.size()==0) return res;

    //     std::vector<std::pair<size_t,size_t>> batch;
    //     for (auto k : keys){
    //         auto k = std::max(first_key, key);
    //         size_t pos;
    //         // auto t0 = timer::now();
            
    //         auto it = segment_for_key(k);
    //         pos = std::min<size_t>((*it)(k), std::next(it)->intercept);
            
    //         // auto t1 = timer::now();
    //         // std::cout << "Index time:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << std::endl;
    //         auto lo = PGM_SUB_EPS(pos, Epsilon);
    //         auto hi = PGM_ADD_EPS(pos, Epsilon, n);
    //         batch.push_back({lo,hi});
    //     }
        
    //     // all-at-once
    //     if (s == ALL_IN_ONCE) {
    //         std::vector<Record> buffer;
    //         // auto t0 = timer::now();
    //         std::vector<pgm::Page> pages = cache->get(lo / ITEM_PER_PAGE, hi / ITEM_PER_PAGE);
    //         // auto t1 = timer::now();
    //         // std::cout << "Cache time:" << std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() << std::endl;
    //         for (auto &page : pages) {
    //             size_t num_records = page.valid_len / sizeof(Record);   // valid record num
    //             Record* records = reinterpret_cast<Record*>(page.data.get());
    //             buffer.insert(buffer.end(), records, records + num_records);
    //         }

    //         size_t page_lo = lo / ITEM_PER_PAGE;
    //         size_t rel_lo = lo % ITEM_PER_PAGE;
    //         size_t rel_hi = std::min(hi - page_lo * ITEM_PER_PAGE, buffer.size());

    //         return {std::move(buffer), rel_lo, rel_hi};
    //     }
    //     else{       // one-by-one
    //         pgm::Page page;
    //         Record* records = nullptr;
    //         for (size_t pageIndex=lo/ITEM_PER_PAGE;pageIndex<=hi/ITEM_PER_PAGE;pageIndex++){
    //             page = cache->get(pageIndex);
    //             records = reinterpret_cast<Record*>(page.data.get());
    //             if (records[page.valid_len/sizeof(Record) - 1].key>=key)    break;
    //         }
    //         size_t num_records = page.valid_len / sizeof(Record);
    //         std::vector<Record> buffer(records, records + num_records);
    //         size_t size = buffer.size(); 
    //         return {std::move(buffer), 0, size};
    //     }
    // }

    /**
     * Returns all keys in provided range.
     * @param lo the target keys lower bound
     * @param hi the target keys upper bound
     * @return a vector contains all keys in provided range.
     */
    std::vector<K> range_search(const K &lo, const K &hi, std::vector<K> target={}, SearchStrategy s=ONE_BY_ONE){
        std::vector<K> res;
        if (hi<first_key) return res;
        size_t p = 0;   // current target index
        // one-by-one strategy
        if (s==ONE_BY_ONE){
            auto it = segment_for_key(lo);
            size_t pos = std::min<size_t>((*it)(lo), std::next(it)->intercept);
            size_t pageIndex = pos/ITEM_PER_PAGE;
            pgm::Page* page = &cache->get(pageIndex);
            K l,r;
            std::pair<K,K> lr;
            lr = inner_search(*page,res,lo,hi,p,target);l=lr.first;r=lr.second;
            size_t p = pageIndex;
            while (l==0&&p>0){
                page = &cache->get(--p);
                lr = inner_search(*page,res,lo,hi,p,target);
                l = lr.first;
            }
            p = pageIndex;
            while (r==ITEM_PER_PAGE-1){
                page = &cache->get(++p);
                lr = inner_search(*page,res,lo,hi,p,target);
                r = lr.second;
            }
        }else{      // all-in-once strategy
            auto it_lo = segment_for_key(lo);
            auto it_hi = segment_for_key(hi);
            int64_t pos_lo = std::min<int64_t>((*it_lo)(lo), std::next(it_lo)->intercept);
            int64_t pos_hi = std::min<int64_t>((*it_hi)(hi), std::next(it_hi)->intercept);
            int64_t page_lo = std::max<int64_t>((int64_t)0,pos_lo-(int64_t)Epsilon)/ITEM_PER_PAGE;
            int64_t page_hi = std::min<int64_t>(n,pos_hi+(int64_t)Epsilon)/ITEM_PER_PAGE;
            std::vector<pgm::Page> pages = cache->get(page_lo,page_hi);

            for (auto &page:pages){
                if (reinterpret_cast<Record*>(page.data.get())[page.valid_len / sizeof(Record) - 1].key<lo) continue;
                else if (reinterpret_cast<Record*>(page.data.get())[0].key>hi) break;
                for (size_t i = 0; i < page.valid_len / sizeof(Record); i++) {
                    K key = reinterpret_cast<Record*>(page.data.get())[i].key;
                    if (target.size()!=0){
                        while (p<target.size()&&target[p]<key) p++;
                        if (p >= target.size()) return res;
                        while (key >= lo && key <= hi && key == target[p]) {
                            res.push_back(key);
                            p++;
                        }
                        if(key>hi) {
                            return res;
                        }
                    }else{
                        if (key >= lo && key <= hi) {
                            res.push_back(key);
                        }
                    }
                }
            }
        }    
        return res;
    }

    std::pair<K,K> inner_search(pgm::Page &p, std::vector<K>& res, K lo, K hi, size_t &idx, std::vector<K> target={}){
        Record* recs = reinterpret_cast<Record*>(p.data.get());
        Record* begin = recs;
        Record* end   = recs + p.valid_len / sizeof(Record);

        Record* lb = std::lower_bound(begin, end, lo, 
            [](const Record& r, K key) { return r.key < key; });
        Record* ub = std::upper_bound(begin, end, hi, 
            [](K key, const Record& r) { return key < r.key; });

        for (auto it=lb;it<ub;it++){
            if (target.size()!=0){
                if (it->key == target[idx]) {
                    res.push_back(it->key);
                    idx++;
                }
                while (idx<target.size()&&target[idx]<it->key) idx++;
                if (idx >= target.size()) return {lb-begin,ub-begin-1};
            }else{
                res.push_back(it->key);
            }
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
    // const size_t get_IO_time() const { return IO_time; }
};


#pragma pack(push, 1)

template<typename K, size_t Epsilon, size_t MemoryBudget,
    size_t EpsilonRecursive, typename Floating>
struct PGMIndexCost<K, Epsilon, MemoryBudget, EpsilonRecursive, Floating>::Segment {
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