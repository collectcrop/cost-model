/*
 * This example shows how to index and query a vector of random integers with the PGM-index.
 * Compile with:
 *   g++ simple.cpp -std=c++17 -I../include -o simple
 * Run with:
 *   ./simple
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <unordered_map>
#include <algorithm>
#include <cstdint>
#include <cassert>
#include <chrono>
#include <fstream>
#include "pgm/pgm_index.hpp"
#include "utils/include.hpp"
#include "utils/utils.hpp"

using KeyType = uint64_t;
#define DIRECTORY "/mnt/home/zwshi/learned-index/cost-model/experiments/"
#define DATASETS "/mnt/home/zwshi/Datasets/SOSD/"

using timer = std::chrono::high_resolution_clock;

template <size_t Epsilon>
void benchmark(std::vector<KeyType> data) {
    pgm::PGMIndex<KeyType, Epsilon, 20> index(data);
    auto height = index.height();
    auto size = index.size_in_bytes();
    std::cout << "epsilon: " << Epsilon << " height: " << height << " index size: " << size << std::endl;
}

int main() {
    std::string filename = "osm_cellids_100M_uint64_unique";
    // std::string query_filename = "range_query_fb_uu.bin";
    std::string file = DATASETS + filename;
    // std::string query_file = DATASETS + query_filename;
    std::vector<KeyType> data = load_data(file,100000000);
    // std::vector<RangeQuery> queries = load_queries(query_file);
    // const size_t MemoryBudget = 80*1024*1024;

    for (size_t epsilon : {4,8,16,32,64,128,256}){
        switch (epsilon) {
            case 4: 
                benchmark<4>(data);
                break;
            case 8:
                benchmark<8>(data);
                break;
            case 16:
                benchmark<16>(data);
                break;
            case 32:
                benchmark<32>(data);
                break;
            case 64:
                benchmark<64>(data);
                break;
            case 128:
                benchmark<128>(data);
                break;
            case 256:
                benchmark<256>(data);
                break;
        }
    }
    
    return 0;
}


