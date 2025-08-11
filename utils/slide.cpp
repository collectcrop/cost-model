#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>

std::vector<uint64_t> load_first_m_keys(const std::string& filename, size_t m) {
    std::ifstream infile(filename, std::ios::binary);
    if (!infile) {
        std::cerr << "Failed to open input file: " << filename << "\n";
        return {};
    }

    // Read total number of keys in the original file
    uint64_t total_keys;
    infile.read(reinterpret_cast<char*>(&total_keys), sizeof(uint64_t));

    size_t read_count = std::min(m, static_cast<size_t>(total_keys));
    std::vector<uint64_t> keys(read_count);
    infile.read(reinterpret_cast<char*>(keys.data()), read_count * sizeof(uint64_t));

    if (!infile) {
        std::cerr << "Error while reading input file data.\n";
    }

    return keys;
}

void write_keys_with_size(const std::string& filename, const std::vector<uint64_t>& keys) {
    std::ofstream outfile(filename, std::ios::binary);
    if (!outfile) {
        std::cerr << "Failed to open output file: " << filename << "\n";
        return;
    }

    uint64_t size = keys.size();
    outfile.write(reinterpret_cast<const char*>(&size), sizeof(uint64_t));  // write number of keys
    outfile.write(reinterpret_cast<const char*>(keys.data()), size * sizeof(uint64_t));

    if (!outfile) {
        std::cerr << "Error while writing output file.\n";
    }
}

void slice(std::string infile, std::string outfile, size_t m){
    std::vector<uint64_t> keys;
    keys = load_first_m_keys(infile, m);
    std::cout << "Loaded " << keys.size() << " keys.\n";
    write_keys_with_size(outfile, keys);
    std::cout << "Wrote sliced keys to " << outfile << "\n";
}
int main() {
    std::string output_file;
    std::vector<size_t> mlist = {10000000ULL, 20000000ULL, 30000000ULL, 50000000ULL, 70000000ULL, 
        90000000ULL, 100000000ULL};  
    std::string prefix = "wiki_ts";
    std::string input_file = prefix + "_200M_uint64";
    for (auto &m : mlist){
        output_file = prefix + "_" + std::to_string((size_t)(m/1e6)) + "M_uint64_unique";
        slice(input_file,output_file,m);
    }
    return 0;
}
