#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <shared_mutex>
#include <unistd.h>
#include <functional> 
#include "FALCON/utils/include.hpp"
#include "btree.h"

class StxDiskKV {
public:
  using Key     = uint64_t;
  using Offset  = uint64_t;  
  using BTree   = stx::btree<Key, Offset>;

  StxDiskKV();
  ~StxDiskKV();

  int open_data_file(const std::string& path, bool direct_io=false);

  void close_data_file();

  // insert or update：key -> offset
  void put(Key k, Offset off);

  // return true/false；existing results will be written to off
  bool get_offset(Key k, Offset& off) const;

  
  // for all keys in [lo, hi] , calling cb(key, offset) in ascending order, return the number of results
  size_t range_scan(Key lo, Key hi,
                      const std::function<void(Key, Offset)>& cb) const;

  ssize_t read_at(Offset off, void* buf, size_t len) const;

  ssize_t get_record(Key k, void* buf, size_t len) const;

  void bulk_build(std::vector<KeyType> keys, const Offset* offs, size_t n);

  // index size
  size_t size() const;

private:
  int data_fd_ = -1;
  bool direct_io_ = false;
  size_t dio_align_ = 4096;

  BTree index_;
  mutable std::shared_mutex mu_; 
  ssize_t pread_aligned(void* user_buf, size_t len, off_t off) const;
};
