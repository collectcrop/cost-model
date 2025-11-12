#define _GNU_SOURCE
#include "stx_disk_kv.h"
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <errno.h>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <condition_variable>

StxDiskKV::StxDiskKV() = default;
StxDiskKV::~StxDiskKV() { close_data_file(); }

int StxDiskKV::open_data_file(const std::string& path, bool direct_io) {
  close_data_file();
  int flags = O_RDONLY;
  if (direct_io) flags |= O_DIRECT;
#ifdef O_NOATIME
  flags |= O_NOATIME;
#endif
  int fd = ::open(path.c_str(), flags);
  if (fd < 0 && errno == EPERM && (flags & O_NOATIME)) {
    // 再试一次去掉 NOATIME
    flags &= ~O_NOATIME;
    fd = ::open(path.c_str(), flags);
  }
  if (fd < 0) return -errno;

  data_fd_   = fd;
  direct_io_ = direct_io;
  dio_align_ = 4096; // 简化：按 4K 对齐；必要时可用 fstatfs/blkid 探测
  return 0;
}

void StxDiskKV::close_data_file() {
  if (data_fd_ >= 0) {
    ::close(data_fd_);
    data_fd_ = -1;
  }
}

void StxDiskKV::put(Key k, Offset off) {
  std::unique_lock lk(mu_);
  auto it = index_.find(k);
  if (it == index_.end()) index_.insert(k, off);
  else it->second = off;
}

bool StxDiskKV::get_offset(Key k, Offset& off) const {
  std::shared_lock lk(mu_);
  auto it = index_.find(k);
  if (it == index_.end()) return false;
  off = it->second;
  return true;
}

ssize_t StxDiskKV::get_record(Key k, void* buf, size_t len) const {
  if (data_fd_ < 0) return -EBADF;
  Offset off;
  {
    std::shared_lock lk(mu_);
    auto it = index_.find(k);
    if (it == index_.end()) return -1; // not found
    off = it->second;
  }
  return pread_aligned(buf, len, (off_t)off);
}

void StxDiskKV::bulk_build(const Key* keys, const Offset* offs, size_t n) {
  std::unique_lock lk(mu_);
  for (size_t i = 0; i < n; ++i) index_.insert(keys[i], offs[i]);
}

size_t StxDiskKV::size() const {
  std::shared_lock lk(mu_);
  return index_.size();
}

ssize_t StxDiskKV::pread_aligned(void* user_buf, size_t len, off_t off) const {
  if (!direct_io_) {
    // 普通 pread
    ssize_t n = ::pread(data_fd_, user_buf, len, off);
    return (n < 0) ? -errno : n;
  }

  // O_DIRECT：要求 (buf, len, off) 全部按对齐
  const size_t A = dio_align_;
  bool aligned_buf = ((uintptr_t)user_buf % A) == 0;
  bool aligned_len = (len % A) == 0;
  bool aligned_off = ((off % (off_t)A) == 0);

  if (aligned_buf && aligned_len && aligned_off) {
    ssize_t n = ::pread(data_fd_, user_buf, len, off);
    return (n < 0) ? -errno : n;
  }

  // 需要 bounce buffer：扩大到对齐边界
  off_t  start_off = (off_t)((off / (off_t)A) * (off_t)A);
  size_t head_skip = (size_t)(off - start_off);
  size_t need      = head_skip + len;
  size_t io_len    = ((need + A - 1) / A) * A;

  void* bounce = nullptr;
  if (posix_memalign(&bounce, A, io_len) != 0) return -ENOMEM;

  ssize_t n = ::pread(data_fd_, bounce, io_len, start_off);
  if (n < 0) {
    int e = errno;
    free(bounce);
    return -e;
  }
  size_t can_copy = std::min((size_t)n, io_len);
  if (can_copy < head_skip) {
    free(bounce);
    return 0; // 文件太短
  }
  size_t avail = can_copy - head_skip;
  size_t tocpy = std::min(avail, len);
  std::memcpy(user_buf, (char*)bounce + head_skip, tocpy);
  free(bounce);
  return (ssize_t)tocpy;
}
