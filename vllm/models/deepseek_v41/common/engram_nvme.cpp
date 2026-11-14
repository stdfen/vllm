// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <linux/stat.h>
#include <sys/stat.h>
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <memory>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

class PackReader {
  struct Job {
    PackReader* owner;
    int64_t ids, weight, scale, count, status;
    bool captured;
  };
  struct Item {
    int64_t id, slot;
  };
  int fd_ = -1;
  int64_t rows_, dim_, scale_, stride_, capacity_ = 0, slots_ = 0;
  int threads_;
  bool direct_io_;
  std::vector<Item> items_;
  std::vector<int64_t> groups_, tags_;
  std::vector<uint8_t> cache_;
  std::unique_ptr<std::mutex[]> stripes_;
  std::mutex run_mu_, queue_mu_;
  std::condition_variable work_, done_;
  std::vector<std::thread> workers_;
  std::vector<void*> scratch_;
  std::vector<std::unique_ptr<Job>> captured_;
  std::atomic<int64_t> next_{0};
  int64_t total_ = 0, finished_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
  uint8_t *out_w_ = nullptr, *out_s_ = nullptr;
  std::atomic<int> failed_{0};
  std::atomic<uint64_t> calls_{0}, logical_{0}, unique_{0}, hits_{0}, reads_{0},
      bytes_{0};

  void read_group(int64_t group, void* scratch) {
    int64_t begin = groups_[group], end = groups_[group + 1];
    int64_t id = items_[begin].id;
    int64_t slot = slots_ ? id % slots_ : 0;
    std::unique_lock<std::mutex> lock;
    if (slots_) lock = std::unique_lock<std::mutex>(stripes_[slot % 64]);
    uint8_t* raw = static_cast<uint8_t*>(scratch);
    if (slots_ && tags_[slot] == id) {
      raw = cache_.data() + slot * (dim_ + scale_);
      hits_++;
    } else {
      ssize_t got;
      do {
        got = ::pread(fd_, raw, stride_, 4096 + id * stride_);
      } while (got < 0 && errno == EINTR);
      reads_++;
      if (got != stride_) {
        if (!failed_.exchange(1))
          std::fprintf(stderr,
                       "P0 Engram read failed: row=%ld offset=%ld got=%ld "
                       "expected=%ld errno=%d\n",
                       id, 4096 + id * stride_, got, stride_,
                       got < 0 ? errno : 0);
        return;
      }
      bytes_ += stride_;
      // Large prefill reads bypass admission to avoid evicting the hot decode
      // set.
      if (slots_ && total_ <= std::max<int64_t>(1, slots_ / 4)) {
        auto target = cache_.data() + slot * (dim_ + scale_);
        std::memcpy(target, raw, dim_ + scale_);
        tags_[slot] = id;
        raw = target;
      }
    }
    for (int64_t i = begin; i < end; ++i) {
      int64_t destination = items_[i].slot;
      std::memcpy(out_w_ + destination * dim_, raw, dim_);
      std::memcpy(out_s_ + destination * scale_, raw + dim_, scale_);
    }
  }
  void worker(void* scratch) {
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> lock(queue_mu_);
      work_.wait(lock, [&] { return stop_ || generation_ != seen; });
      if (stop_) break;
      seen = generation_;
      lock.unlock();
      for (int64_t i; (i = next_.fetch_add(1)) < total_;)
        read_group(i, scratch);
      lock.lock();
      if (++finished_ == threads_) done_.notify_one();
    }
  }
  void perform(Job* job) noexcept {
    std::lock_guard<std::mutex> run(run_mu_);
    auto status = reinterpret_cast<int32_t*>(job->status);
    try {
      *status = failed_.load();
      if (*status) return;
      int64_t n = job->count;
      auto ids = reinterpret_cast<const int64_t*>(job->ids);
      out_w_ = reinterpret_cast<uint8_t*>(job->weight);
      out_s_ = reinterpret_cast<uint8_t*>(job->scale);
      std::memset(out_w_, 0, n * dim_);
      std::memset(out_s_, 0, n * scale_);
      int64_t valid = 0;
      for (int64_t i = 0; i < n; ++i)
        if (ids[i] >= 0 && ids[i] < rows_) items_[valid++] = {ids[i], i};
      std::sort(items_.begin(), items_.begin() + valid,
                [](const Item& a, const Item& b) { return a.id < b.id; });
      int64_t groups = 0;
      for (int64_t i = 0; i < valid; ++i)
        if (i == 0 || items_[i].id != items_[i - 1].id) groups_[groups++] = i;
      groups_[groups] = valid;
      calls_++;
      logical_ += n;
      unique_ += groups;
      if (groups) {
        std::unique_lock<std::mutex> lock(queue_mu_);
        total_ = groups;
        next_ = 0;
        finished_ = 0;
        ++generation_;
        work_.notify_all();
        done_.wait(lock, [&] { return finished_ == threads_; });
      }
      *status = failed_.load();
    } catch (...) {
      failed_.store(1);
      *status = 1;
    }
  }
  static void CUDART_CB callback(void* pointer) {
    auto job = static_cast<Job*>(pointer);
    job->owner->perform(job);
    if (!job->captured) delete job;
  }

 public:
  PackReader(const std::string& path, int64_t rows, int64_t dim, int64_t scale,
             int64_t stride, int threads, bool direct_io = true)
      : rows_(rows),
        dim_(dim),
        scale_(scale),
        stride_(stride),
        threads_(threads),
        direct_io_(direct_io) {
    if (rows <= 0 || dim <= 0 || scale <= 0 || stride < dim + scale ||
        stride % 512 || threads < 1 || threads > 128)
      throw std::invalid_argument("Invalid Engram pack reader geometry");
    fd_ = ::open(path.c_str(),
                 O_RDONLY | O_CLOEXEC | (direct_io_ ? O_DIRECT : 0));
    if (fd_ < 0) throw std::runtime_error("Engram pack open failed");
    if (!direct_io_) posix_fadvise(fd_, 0, 0, POSIX_FADV_RANDOM);
    struct statx info{};
    if (direct_io_ &&
        (::statx(fd_, "", AT_EMPTY_PATH, STATX_DIOALIGN, &info) != 0 ||
         !(info.stx_mask & STATX_DIOALIGN) || !info.stx_dio_offset_align ||
         stride % info.stx_dio_offset_align || !info.stx_dio_mem_align ||
         4096 % info.stx_dio_mem_align)) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("Engram filesystem DIO alignment unsupported");
    }
    void* probe = nullptr;
    if (posix_memalign(&probe, 4096, stride_) != 0) throw std::bad_alloc();
    auto got = ::pread(fd_, probe, stride_, 4096);
    std::free(probe);
    if (got != stride_) {
      ::close(fd_);
      fd_ = -1;
      throw std::runtime_error("Engram initial direct read failed");
    }
    try {
      for (int i = 0; i < threads_; ++i) {
        void* scratch = nullptr;
        if (posix_memalign(&scratch, 4096, stride_) != 0)
          throw std::bad_alloc();
        scratch_.push_back(scratch);
        workers_.emplace_back([this, scratch] { worker(scratch); });
      }
    } catch (...) {
      {
        std::lock_guard<std::mutex> lock(queue_mu_);
        stop_ = true;
      }
      work_.notify_all();
      for (auto& worker : workers_) worker.join();
      for (auto scratch : scratch_) std::free(scratch);
      ::close(fd_);
      fd_ = -1;
      throw;
    }
  }
  ~PackReader() {
    // Graph host nodes retain Job pointers; drain them before releasing the
    // table.
    cudaDeviceSynchronize();
    {
      std::lock_guard<std::mutex> lock(queue_mu_);
      stop_ = true;
    }
    work_.notify_all();
    for (auto& worker : workers_) worker.join();
    for (auto scratch : scratch_) std::free(scratch);
    if (fd_ >= 0) ::close(fd_);
  }
  void allocate(int64_t capacity, int64_t cache_bytes) {
    if (capacity_ || capacity < 1 || cache_bytes < 0)
      throw std::invalid_argument(
          "Engram reader staging can only be allocated once");
    capacity_ = capacity;
    items_.resize(capacity);
    groups_.resize(capacity + 1);
    int64_t fixed = 64 * sizeof(std::mutex);
    slots_ = cache_bytes > fixed
                 ? (cache_bytes - fixed) / (dim_ + scale_ + sizeof(int64_t))
                 : 0;
    if (slots_) stripes_ = std::make_unique<std::mutex[]>(64);
    tags_.assign(slots_, -1);
    cache_.resize(slots_ * (dim_ + scale_));
  }
  void enqueue_read(int64_t stream, int64_t ids, int64_t weight, int64_t scale,
                    int64_t count, int64_t status, bool capturing) {
    if (count < 0 || count > capacity_)
      throw std::invalid_argument("Engram staging overflow");
    Job* job = nullptr;
    if (capturing) {
      if (captured_.size() >= 64)
        throw std::runtime_error("Engram captured job limit exceeded");
      captured_.push_back(std::make_unique<Job>(
          Job{this, ids, weight, scale, count, status, true}));
      job = captured_.back().get();
    } else
      job = new Job{this, ids, weight, scale, count, status, false};
    auto error = cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream),
                                    callback, job);
    if (error != cudaSuccess) {
      if (!capturing) delete job;
      throw std::runtime_error(cudaGetErrorString(error));
    }
  }
  void read_rows(int64_t ids, int64_t weight, int64_t scale, int64_t count) {
    if (count < 0 || count > capacity_)
      throw std::invalid_argument("Engram staging overflow");
    int32_t status = 0;
    Job job{this, ids, weight, scale, count, reinterpret_cast<int64_t>(&status),
            false};
    perform(&job);
    if (status)
      throw std::runtime_error("Engram direct read failed (short read/EIO)");
  }
  std::map<std::string, uint64_t> stats() const {
    return {
        {"calls", calls_.load()},
        {"logical_rows", logical_.load()},
        {"unique_rows", unique_.load()},
        {"cache_hits", hits_.load()},
        {"disk_reads", reads_.load()},
        {"disk_bytes", bytes_.load()},
        {"failed", failed_.load()},
        {"direct_io", static_cast<uint64_t>(direct_io_)},
        {"cache_allocated_bytes", cache_.capacity() + tags_.capacity() * 8 +
                                      (slots_ ? 64 * sizeof(std::mutex) : 0)},
        {"reader_fixed_bytes", sizeof(*this)},
        {"staging_metadata_bytes",
         items_.size() * sizeof(Item) + groups_.size() * 8},
        {"worker_scratch_bytes", static_cast<uint64_t>(threads_ * stride_)}};
  }
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  pybind11::class_<PackReader>(module, "PackReader")
      .def(pybind11::init<const std::string&, int64_t, int64_t, int64_t,
                          int64_t, int, bool>(),
           pybind11::arg("path"), pybind11::arg("rows"), pybind11::arg("dim"),
           pybind11::arg("scale"), pybind11::arg("stride"),
           pybind11::arg("threads"), pybind11::arg("direct_io") = true)
      .def("allocate", &PackReader::allocate)
      .def("enqueue_read", &PackReader::enqueue_read)
      .def("read_rows", &PackReader::read_rows,
           pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("stats", &PackReader::stats);
}
