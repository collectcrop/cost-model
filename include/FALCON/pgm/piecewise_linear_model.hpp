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

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <map>
#ifdef _OPENMP
#include <omp.h>
#else
#pragma message ("Compilation with -fopenmp is optional but recommended")
#define omp_get_num_procs() 1
#define omp_get_max_threads() 1
#endif

#define OPT_OFFSET_MASK 0x0F
#define OPT_SLICES_PER_SIDE 4        
#define ACCURATE 0XF
#define UNSET 0x0
#define ABOVE 0x1
#define BELOW 0x2
#define OPT_RANGE_START 0x3
#define PGM_SUB_EPS(x, epsilon) ((x) <= (epsilon) ? 0 : ((x) - (epsilon)))
#define PGM_ADD_EPS(x, epsilon, size) ((x) + (epsilon) + 2 >= (size) ? (size) : (x) + (epsilon) + 2)

namespace pgm::internal {

typedef struct Record {
    uint64_t key;
    uint64_t pointer;
} Record;

template<typename T>
T* encode_ptr_with_offset(T* ptr, uint8_t offset_info) {
    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    p = (p & ~uintptr_t(OPT_OFFSET_MASK)) | offset_info;  
    return reinterpret_cast<T*>(p);
}

template<typename T>
uint8_t inline get_offset_from_ptr(T* ptr) {
    uintptr_t p = reinterpret_cast<uintptr_t>(ptr);
    return static_cast<uint8_t>(p & OPT_OFFSET_MASK);
}

uint8_t generate_encoding(int offset, size_t epsilon) {
    // check validity of offset
    if (offset < -(int)epsilon || offset > (int)epsilon) {
        throw std::out_of_range("offset is out of [-epsilon, epsilon]");
    }
    if (!offset){
        return ACCURATE;  // accurate prediction
    }
    double step = static_cast<double>(epsilon) / 4.0;
    int index = static_cast<int>((offset + static_cast<int>(epsilon)) / step);
    if (index >= 8) index = 7;  // cap at 7 to avoid overflow
    return static_cast<uint8_t>(OPT_RANGE_START + index);
}

std::pair<int, int> decode_encoding(uint8_t code, size_t epsilon , size_t pos, size_t size) {
    // if (code < OPT_RANGE_START || code >= OPT_RANGE_START + 8) {
    //     throw std::invalid_argument("Invalid encoding value");
    // }
    int low,high;
    if (code>=OPT_RANGE_START) {
        int index = code - OPT_RANGE_START;
        double step = static_cast<double>(epsilon) / 4.0;

        // offset ∈ [low, high], both inclusive
        low = static_cast<int>(pos+std::ceil(-static_cast<double>(epsilon) + index * step));
        high = static_cast<int>(pos+std::floor(-static_cast<double>(epsilon) + (index + 1) * step));
    }else if(code!=UNSET){
        low = (code == ABOVE) ? pos+1 : PGM_SUB_EPS(pos, epsilon);
        high = (code == BELOW) ? pos-1 : PGM_ADD_EPS(pos, epsilon, size);
    }else{
        low = PGM_SUB_EPS(pos, epsilon);
        high = PGM_ADD_EPS(pos, epsilon, size);
    }

    
    return {low, high};
}

template<typename RandomIt, typename MapT, typename SegmentT>
void analyze_segment_conflict(const SegmentT& s, RandomIt begin, RandomIt end, RandomIt data_begin, 
                                MapT& conflict_map, size_t epsilon) {
    std::map<size_t, std::vector<uint64_t>> buckets;

    for (auto it = begin; it != end; ++it) {
        // const Record &rec = *it; 
        const auto key = *it;
        size_t pos = s(key);
        buckets[pos].push_back(key);
    }

    auto it = buckets.begin();
    auto last = std::prev(buckets.end());
    for (; it != buckets.end(); ++it) {
        const auto& [pos, keys] = *it;

        if (it == last) {
            // skip last bucket — avoid cross-segment conflict
            break;
        }
        uint8_t encoded;
        if (keys.size() == 1) {     // No conflict
            // const Record &rec = records[0];
            // size_t actual_pos = std::distance(data_begin, std::lower_bound(begin, end, rec.key));
            // int offset = static_cast<int>(actual_pos) - static_cast<int>(pos);
            // encoded = generate_encoding(offset, epsilon);

            // // write the pointer with offset
            // pgm::internal::encode_ptr_with_offset<uint64_t>(rec.pointer, encoded);
            continue;
        }

        auto mid_key = keys[0];
        bool has_exact_match = false;
        bool isAbove = false;
        for (const auto& key : keys) {
            size_t actual_pos = std::distance(data_begin, std::lower_bound(begin, end, key));
            if (pos == actual_pos) {
                mid_key = key; 
                has_exact_match = true;
                break;
            }else if (actual_pos > pos) {
                isAbove = true;
                // maybe could break here
            }
        }

        conflict_map[pos] = mid_key;
        // if (has_exact_match) {
        //     conflict_map[pos] = mid_key;  // 中位 key 作代表
        // }else{
        //     encoded = isAbove ? ABOVE : BELOW;  // 1: above, 2: below
        //     const Record &rec = records.front();
        //     pgm::internal::encode_ptr_with_offset(rec.pointer, encoded);
        // }
    }
}

template<typename T>
using LargeSigned = typename std::conditional_t<std::is_floating_point_v<T>,
                                                long double,
                                                std::conditional_t<(sizeof(T) < 8), int64_t, __int128>>;

template<typename X, typename Y>
class OptimalPiecewiseLinearModel {
private:
    using SX = LargeSigned<X>;
    using SY = LargeSigned<Y>;

    struct Slope {
        SX dx{};
        SY dy{};

        bool operator<(const Slope &p) const { return dy * p.dx < dx * p.dy; }
        bool operator>(const Slope &p) const { return dy * p.dx > dx * p.dy; }
        bool operator==(const Slope &p) const { return dy * p.dx == dx * p.dy; }
        bool operator!=(const Slope &p) const { return dy * p.dx != dx * p.dy; }
        explicit operator long double() const { return dy / (long double) dx; }
    };

    struct Point {
        X x{};
        Y y{};

        Slope operator-(const Point &p) const { return {SX(x) - p.x, SY(y) - p.y}; }
    };

    const Y epsilon;
    std::vector<Point> lower;
    std::vector<Point> upper;
    X first_x = 0;
    X last_x = 0;
    size_t lower_start = 0;
    size_t upper_start = 0;
    size_t points_in_hull = 0;
    Point rectangle[4];

    auto cross(const Point &O, const Point &A, const Point &B) const {
        auto OA = A - O;
        auto OB = B - O;
        return OA.dx * OB.dy - OA.dy * OB.dx;
    }

public:

    class CanonicalSegment;

    explicit OptimalPiecewiseLinearModel(Y epsilon) : epsilon(epsilon), lower(), upper() {
        if (epsilon < 0)
            throw std::invalid_argument("epsilon cannot be negative");

        upper.reserve(1u << 16);
        lower.reserve(1u << 16);
    }

    bool add_point(const X &x, const Y &y) {
        if (points_in_hull > 0 && x <= last_x)
            throw std::logic_error("Points must be increasing by x.");

        last_x = x;
        auto max_y = std::numeric_limits<Y>::max();
        auto min_y = std::numeric_limits<Y>::lowest();
        Point p1{x, y >= max_y - epsilon ? max_y : y + epsilon};
        Point p2{x, y <= min_y + epsilon ? min_y : y - epsilon};

        if (points_in_hull == 0) {
            first_x = x;
            rectangle[0] = p1;
            rectangle[1] = p2;
            upper.clear();
            lower.clear();
            upper.push_back(p1);
            lower.push_back(p2);
            upper_start = lower_start = 0;
            ++points_in_hull;
            return true;
        }

        if (points_in_hull == 1) {
            rectangle[2] = p2;
            rectangle[3] = p1;
            upper.push_back(p1);
            lower.push_back(p2);
            ++points_in_hull;
            return true;
        }

        auto slope1 = rectangle[2] - rectangle[0];
        auto slope2 = rectangle[3] - rectangle[1];
        bool outside_line1 = p1 - rectangle[2] < slope1;
        bool outside_line2 = p2 - rectangle[3] > slope2;

        if (outside_line1 || outside_line2) {
            points_in_hull = 0;
            return false;
        }

        if (p1 - rectangle[1] < slope2) {
            // Find extreme slope
            auto min = lower[lower_start] - p1;
            auto min_i = lower_start;
            for (auto i = lower_start + 1; i < lower.size(); i++) {
                auto val = lower[i] - p1;
                if (val > min)
                    break;
                min = val;
                min_i = i;
            }

            rectangle[1] = lower[min_i];
            rectangle[3] = p1;
            lower_start = min_i;

            // Hull update
            auto end = upper.size();
            for (; end >= upper_start + 2 && cross(upper[end - 2], upper[end - 1], p1) <= 0; --end)
                continue;
            upper.resize(end);
            upper.push_back(p1);
        }

        if (p2 - rectangle[0] > slope1) {
            // Find extreme slope
            auto max = upper[upper_start] - p2;
            auto max_i = upper_start;
            for (auto i = upper_start + 1; i < upper.size(); i++) {
                auto val = upper[i] - p2;
                if (val < max)
                    break;
                max = val;
                max_i = i;
            }

            rectangle[0] = upper[max_i];
            rectangle[2] = p2;
            upper_start = max_i;

            // Hull update
            auto end = lower.size();
            for (; end >= lower_start + 2 && cross(lower[end - 2], lower[end - 1], p2) >= 0; --end)
                continue;
            lower.resize(end);
            lower.push_back(p2);
        }

        ++points_in_hull;
        return true;
    }

    CanonicalSegment get_segment() {
        if (points_in_hull == 1)
            return CanonicalSegment(rectangle[0], rectangle[1], first_x);
        return CanonicalSegment(rectangle, first_x);
    }

    CanonicalSegment get_segment(X last_x) {
        if (points_in_hull == 1)
            return CanonicalSegment(rectangle[0], rectangle[1], first_x);
        return CanonicalSegment(rectangle, first_x, last_x);
    }

    void reset() {
        points_in_hull = 0;
        lower.clear();
        upper.clear();
    }
};

template<typename X, typename Y>
class OptimalPiecewiseLinearModel<X, Y>::CanonicalSegment {
    friend class OptimalPiecewiseLinearModel;

    Point rectangle[4];
    X first;
    X last; 
    CanonicalSegment(const Point &p0, const Point &p1, X first) : rectangle{p0, p1, p0, p1}, first(first) {};

    CanonicalSegment(const Point (&rectangle)[4], X first)
        : rectangle{rectangle[0], rectangle[1], rectangle[2], rectangle[3]}, first(first) {};

    CanonicalSegment(const Point &p0, const Point &p1, X first, X last) : rectangle{p0, p1, p0, p1}, first(first), last(last) {};

    CanonicalSegment(const Point (&rectangle)[4], X first, X last)
        : rectangle{rectangle[0], rectangle[1], rectangle[2], rectangle[3]}, first(first), last(last) {};
    bool one_point() const {
        return rectangle[0].x == rectangle[2].x && rectangle[0].y == rectangle[2].y
            && rectangle[1].x == rectangle[3].x && rectangle[1].y == rectangle[3].y;
    }

public:

    CanonicalSegment() = default;

    X get_first_x() const { return first; }
    
    X get_last_x() const { return last; }
    std::pair<long double, long double> get_intersection() const {
        auto &p0 = rectangle[0];
        auto &p1 = rectangle[1];
        auto &p2 = rectangle[2];
        auto &p3 = rectangle[3];
        auto slope1 = p2 - p0;
        auto slope2 = p3 - p1;

        if (one_point() || slope1 == slope2)
            return {p0.x, p0.y};

        auto p0p1 = p1 - p0;
        auto a = slope1.dx * slope2.dy - slope1.dy * slope2.dx;
        auto b = (p0p1.dx * slope2.dy - p0p1.dy * slope2.dx) / static_cast<long double>(a);
        auto i_x = p0.x + b * slope1.dx;
        auto i_y = p0.y + b * slope1.dy;
        return {i_x, i_y};
    }

    std::pair<long double, SY> get_floating_point_segment(const X &origin) const {
        if (one_point())
            return {0, (rectangle[0].y + rectangle[1].y) / 2};

        if constexpr (std::is_integral_v<X> && std::is_integral_v<Y>) {
            auto slope = rectangle[3] - rectangle[1];
            auto intercept_n = slope.dy * (SX(origin) - rectangle[1].x);
            auto intercept_d = slope.dx;
            auto rounding_term = ((intercept_n < 0) ^ (intercept_d < 0) ? -1 : +1) * intercept_d / 2;
            auto intercept = (intercept_n + rounding_term) / intercept_d + rectangle[1].y;
            return {static_cast<long double>(slope), intercept};
        }

        auto[i_x, i_y] = get_intersection();
        auto[min_slope, max_slope] = get_slope_range();
        auto slope = (min_slope + max_slope) / 2.;
        auto intercept = i_y - (i_x - origin) * slope;
        return {slope, intercept};
    }

    std::pair<long double, long double> get_slope_range() const {
        if (one_point())
            return {0, 1};

        auto min_slope = static_cast<long double>(rectangle[2] - rectangle[0]);
        auto max_slope = static_cast<long double>(rectangle[3] - rectangle[1]);
        return {min_slope, max_slope};
    }
};

template<typename Fin, typename Fout>
size_t make_segmentation(size_t n, size_t start, size_t end, size_t epsilon, Fin in, Fout out) {
    using K = typename std::invoke_result_t<Fin, size_t>;
    size_t c = 0;
    OptimalPiecewiseLinearModel<K, size_t> opt(epsilon);
    auto add_point = [&](K x, size_t y) {
        if (!opt.add_point(x, y)) {
            out(opt.get_segment(x));
            opt.add_point(x, y);
            ++c;
        }
    };

    add_point(in(start), start);
    for (size_t i = start + 1; i < end - 1; ++i) {
        if (in(i) == in(i - 1)) {
            // Here there is an adjustment for inputs with duplicate keys: at the end of a run of duplicate keys equal
            // to x=in(i) such that x+1!=in(i+1), we map the values x+1,...,in(i+1)-1 to their correct rank i.
            // For floating-point keys, the value x+1 above is replaced by the next representable value following x.
            if constexpr (std::is_floating_point_v<K>) {
                K next;
                if ((next = std::nextafter(in(i), std::numeric_limits<K>::infinity())) < in(i + 1))
                    add_point(next, i);
            } else {
                if (in(i) + 1 < in(i + 1))
                    add_point(in(i) + 1, i);
            }
        } else {
            add_point(in(i), i);
        }
    }
    if (end >= start + 2 && in(end - 1) != in(end - 2))
        add_point(in(end - 1), end - 1);

    if (end == n) {
        // Ensure values greater than the last one are mapped to n
        if constexpr (std::is_floating_point_v<K>) {
            add_point(std::nextafter(in(n - 1), std::numeric_limits<K>::infinity()), n);
        } else {
            add_point(in(n - 1) + 1, n);
        }
    }

    out(opt.get_segment(in(end - 1)));
    return ++c;
}

template<typename Fin, typename Fout>
size_t make_segmentation(size_t n, size_t epsilon, Fin in, Fout out) {
    return make_segmentation(n, 0, n, epsilon, in, out);
}

template<typename Fin, typename Fout>
size_t make_segmentation_par(size_t n, size_t epsilon, Fin in, Fout out) {
    auto parallelism = std::min(std::min(omp_get_num_procs(), omp_get_max_threads()), 20);
    auto chunk_size = n / parallelism;
    auto c = 0ull;

    if (parallelism == 1 || n < 1ull << 15)
        return make_segmentation(n, epsilon, in, out);

    using K = typename std::invoke_result_t<Fin, size_t>;
    using canonical_segment = typename OptimalPiecewiseLinearModel<K, size_t>::CanonicalSegment;
    std::vector<std::vector<canonical_segment>> results(parallelism);

    #pragma omp parallel for reduction(+:c) num_threads(parallelism)
    for (auto i = 0; i < parallelism; ++i) {
        auto first = i * chunk_size;
        auto last = i == parallelism - 1 ? n : first + chunk_size;
        if (first > 0) {
            for (; first < last; ++first)
                if (in(first) != in(first - 1))
                    break;
            if (first == last)
                continue;
        }

        auto in_fun = [in](auto j) { return in(j); };
        auto out_fun = [&results, i](const auto &cs) { results[i].emplace_back(cs); };
        results[i].reserve(chunk_size / (epsilon > 0 ? epsilon * epsilon : 16));
        c += make_segmentation(n, first, last, epsilon, in_fun, out_fun);
    }

    for (auto &v : results)
        for (auto &cs : v)
            out(cs);

    return c;
}

}
