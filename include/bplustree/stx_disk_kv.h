#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <shared_mutex>
#include <unistd.h>
#include <functional> 
#include "utils/include.hpp"
#include "btree.h"

class StxDiskKV {
public:
  using Key     = uint64_t;
  using Offset  = uint64_t;  // 文件偏移（字节）
  using BTree   = stx::btree<Key, Offset>;

  // 构造 / 析构
  StxDiskKV();
  ~StxDiskKV();

  // 打开数据文件（只读即可）。如果 direct_io=true，则以 O_DIRECT 打开并自动做对齐读。
  // 返回 0 表示成功，<0 表示失败（errno）。
  int open_data_file(const std::string& path, bool direct_io=false);

  // 关闭数据文件
  void close_data_file();

  // 索引操作（线程安全）
  // 插入或更新：key -> offset
  void put(Key k, Offset off);

  // 查 offset（existence）：返回 true/false；如果存在写回 off
  bool get_offset(Key k, Offset& off) const;

  
  // 对 [lo, hi] 范围内所有 key，按 key 升序回调 cb(key, offset)，返回命中数量
  size_t range_scan(Key lo, Key hi,
                      const std::function<void(Key, Offset)>& cb) const;

  ssize_t read_at(Offset off, void* buf, size_t len) const;
  // 从数据文件读取记录：先查 offset，再 pread 到 buf
  //   适用于“知道记录长度”的固定长/可变长协议（你自己控制 len）
  // 返回读到的字节数；<0 表示失败（errno）
  ssize_t get_record(Key k, void* buf, size_t len) const;

  // 批量构建（keys 与 offsets 等长，通常从已知数据源导入）
  void bulk_build(std::vector<KeyType> keys, const Offset* offs, size_t n);

  // 索引大小
  size_t size() const;

private:
  int data_fd_ = -1;
  bool direct_io_ = false;
  size_t dio_align_ = 4096;

  BTree index_;
  mutable std::shared_mutex mu_; // 写少读多：shared_mutex 更合适

  // 处理 O_DIRECT 对齐读：如果 direct_io_ 开启，会自动做对齐、拷贝
  ssize_t pread_aligned(void* user_buf, size_t len, off_t off) const;
};
