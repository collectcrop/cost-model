#include <vector>
#include <cstdlib>
#include <iostream>
#include <algorithm>
#include "FALCON/pgm/pgm_index.hpp"
#include "FALCON/utils/include.hpp"
#include "FALCON/utils/utils.hpp"
#include "FALCON/utils/LatencyRecorder.hpp"
#include "FALCON/Falcon.hpp"     
#include "FALCON/cache/CacheInterface.hpp" 

using KeyType = uint64_t;
int main() {
    std::string dataset_name = "books_200M_uint64_unique";
    std::string file         = falcon::DATASETS + dataset_name;
    size_t num_keys = 1000000;
    size_t M = 10*1024*1024;
    std::vector<KeyType> data    = load_data_pgm_safe<KeyType>(file, num_keys);
    
    // Open data file
    int data_fd = ::open(file.c_str(), O_RDONLY|O_DIRECT);
    if (data_fd < 0) {
        perror("open data");
        std::exit(1);
    }

    // Construct the PGM-index
    const int Epsilon = 64; 
    pgm::PGMIndex<KeyType, Epsilon> index(data);
    
    // Construct FALCON
    falcon::FalconPGM<uint64_t, Epsilon, /*EpsRec*/ 4> F(
        index,
        data_fd,
        falcon::IO_URING,
        /*memory_budget_bytes=*/ M,
        /*cache_policy=*/ falcon::CachePolicy::LRU
    );

    // Point Query test
    auto q = 100;
    auto pointRes = F.point_lookup(q).get();
    if (pointRes.found) {
        std::cout << "Found " << pointRes.key << std::endl;
    } else {
        std::cout << "Did not find " << q << std::endl;
    }

    // Range Query test
    auto rangeRes = F.range_lookup(100, 1000).get();
    if (rangeRes.keys.size() > 0) {
        std::cout << "Found " << rangeRes.keys.size() << " keys in range" << std::endl;
        for (auto key : rangeRes.keys) {
            std::cout << key << " ";
        }
        std::cout << std::endl;
    } else {
        std::cout << "Did not find any keys in range" << std::endl;
    }
    return 0;
}