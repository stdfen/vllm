// engram_disk: disk-backed Engram table reads for vLLM on unified-memory boxes (DGX Spark / GB10), where the
// ~100 GB of pinned host Engram rows per rank that upstream keeps cannot fit next to the model.
//
// The table stays in the checkpoint's safetensors shards. Each lookup step:
//   1. the caller copies this step's local row ids (int64, relative to the rank's first row) into a pinned buffer,
//   2. `enqueue_read` launches a host function on the CUDA stream (cudaLaunchHostFunc, so it is captured into CUDA
//      graphs as a host node) that fills pinned staging rows (FP8 weight rows + ue8m0 scale rows) from a hot-row
//      cache or from disk with a persistent thread pool of preads,
//   3. the caller runs a device kernel over the staging rows (vLLM reuses its own Engram lookup kernel).
// Host functions must not call CUDA APIs; this one only reads pinned memory and files.
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <dlfcn.h>
#include <linux/aio_abi.h>
#include <sys/syscall.h>
#include <chrono>
#include <algorithm>
#include <fcntl.h>
#include <unistd.h>
#include <limits>
#include <atomic>
#include <cerrno>
#include <exception>
#include <cstdlib>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <unordered_map>
#include <tuple>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Pool {
  std::vector<std::thread> threads;
  std::mutex mu;
  std::condition_variable cv, done_cv;
  std::function<void(int64_t)> task;
  std::atomic<int64_t> next{0};
  int64_t total = 0, finished_workers = 0;
  uint64_t generation = 0;
  bool stop = false;
  std::exception_ptr error;

  explicit Pool(int n) {
    try {
      for (int i = 0; i < n; i++) threads.emplace_back([this] { loop(); });
    } catch (...) {
      { std::lock_guard<std::mutex> g(mu); stop = true; }
      cv.notify_all();
      for (auto& t : threads) t.join();
      throw;
    }
  }
  ~Pool() {
    { std::lock_guard<std::mutex> g(mu); stop = true; }
    cv.notify_all();
    for (auto& t : threads) t.join();
  }
  void loop() {
    uint64_t seen = 0;
    for (;;) {
      std::unique_lock<std::mutex> l(mu);
      cv.wait(l, [&] { return stop || generation != seen; });
      if (stop) return;
      seen = generation;
      l.unlock();
      // Every worker must leave the generation before run() can return.
      for (int64_t i; (i = next.fetch_add(1)) < total;) {
        try { task(i); }
        catch (...) {
          std::lock_guard<std::mutex> guard(mu);
          if (!error) error = std::current_exception();
        }

      }
      l.lock();
      finished_workers++;
      if (finished_workers == static_cast<int64_t>(threads.size())) done_cv.notify_all();
    }
  }
  void run(int64_t n, std::function<void(int64_t)> f) {
    if (n <= 0) return;
    std::unique_lock<std::mutex> l(mu);
    error = nullptr; task = std::move(f); total = n; finished_workers = 0; next = 0; generation++;
    cv.notify_all();
    done_cv.wait(l, [&] { return finished_workers == static_cast<int64_t>(threads.size()); });
    if (error) std::rethrow_exception(error);
  }
};

// Fail-closed stream-memory proxy. A graph may retain these mapped words
// indefinitely; their lifetime is process-long, matching host-node tombstones.
using MemopFn = CUresult (*)(CUstream, CUdeviceptr, cuuint32_t, unsigned int);
struct StreamMemops {
  MemopFn write = nullptr, wait = nullptr;
  StreamMemops() {
    void* h = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) return;
    auto sym = [h](const char* a, const char* b) {
      void* p = dlsym(h, a); return p ? p : dlsym(h, b);
    };
    write = reinterpret_cast<MemopFn>(sym("cuStreamWriteValue32_v2", "cuStreamWriteValue32"));
    wait = reinterpret_cast<MemopFn>(sym("cuStreamWaitValue32_v2", "cuStreamWaitValue32"));
  }
};
const StreamMemops& stream_memops() { static StreamMemops m; return m; }

}  // namespace

class EngramDiskTable {
 public:
  // weight/scale rows live at file offsets off_w + row * row_bytes and off_s + row * scale_bytes, where row is the
  // table-global row; rows [row_start, row_start + num_rows) belong to this rank.
  EngramDiskTable(const std::string& weight_file, int64_t off_w, const std::string& scale_file, int64_t off_s,
                  int64_t row_start, int64_t num_rows, int64_t row_bytes, int64_t scale_bytes, int64_t cache_rows,
                  int threads, bool direct_io = false)
      : off_w_(off_w), off_s_(off_s), row_start_(row_start), num_rows_(num_rows), row_bytes_(row_bytes),
        scale_bytes_(scale_bytes), direct_io_(direct_io), pool_(threads > 0 ? threads : 1) {
    if (threads < 1 || row_start < 0 || num_rows < 1 || row_bytes < 1 || scale_bytes < 1 || off_w < 0 || off_s < 0)
      throw std::invalid_argument("engram_disk: invalid table geometry");
    const int flags = O_RDONLY | O_CLOEXEC | (direct_io_ ? O_DIRECT : 0);
    fd_w_ = ::open(weight_file.c_str(), flags);
    fd_s_ = scale_file == weight_file ? fd_w_ : ::open(scale_file.c_str(), flags);
    if (fd_w_ < 0 || fd_s_ < 0) {
      if (fd_s_ >= 0 && fd_s_ != fd_w_) ::close(fd_s_);
      if (fd_w_ >= 0) ::close(fd_w_);
      throw std::runtime_error("engram_disk: cannot open checkpoint table");
    }
    try {
    posix_fadvise(fd_w_, 0, 0, POSIX_FADV_RANDOM);
    if (fd_s_ != fd_w_) posix_fadvise(fd_s_, 0, 0, POSIX_FADV_RANDOM);
    // direct-mapped hot cache: slot = row % cache_rows; tag -1 = empty
    cache_rows_ = cache_rows > 0 ? cache_rows : 0;
    if (cache_rows_) {
      tags_.assign(cache_rows_, -1);
      cache_w_.resize(cache_rows_ * row_bytes_);
      cache_s_.resize(cache_rows_ * scale_bytes_);
    }
    } catch (...) {
      if (fd_s_ >= 0 && fd_s_ != fd_w_) ::close(fd_s_);
      if (fd_w_ >= 0) ::close(fd_w_);
      throw;
    }
  }
  ~EngramDiskTable() { close(); }

  // The CUDA graph owns a raw host-node argument. Keep that argument alive
  // even when Python drops the table: a later replay must fail closed rather
  // than dereference a freed table or pinned staging memory. Its reclaim is
  // intentionally process-lifetime; graph executables are not enumerable here.
  // This is a process-lifetime safety tombstone, not graph ownership:
  // the runner must still destroy graph executables before model teardown.
  void close() {
    std::lock_guard<std::mutex> guard(lifetime_mu_);
    if (closed_.load()) return;
    closed_ = true;
    if (spin_enabled_.load(std::memory_order_acquire)) shared_spin_proxy().remove(this);
    for (auto& pair : jobs_) {
      std::lock_guard<std::mutex> job_guard(pair.second->mu);
      pair.second->t = nullptr;
    }
    std::lock_guard<std::mutex> run_guard(run_mu_);
    // The caller must synchronize all streams before dropping staging.
    // A queued callback may still run, but it holds this lock and sees closed_.
    if (fd_s_ >= 0 && fd_s_ != fd_w_) ::close(fd_s_);
    if (fd_w_ >= 0) ::close(fd_w_);
    fd_s_ = fd_w_ = -1;
  }
  // ids: pinned int64 [n] local rows (-1 = skip, row left zero); staging_w: pinned uint8 [n, row_bytes];
  // staging_s: pinned uint8 [n, scale_bytes]. Synchronous; also used by the host function.
  void read_rows(int64_t ids_ptr, int64_t w_ptr, int64_t s_ptr, int64_t n) {
    std::lock_guard<std::mutex> run_guard(run_mu_);  // one pool.run at a time
    if (closed_.load()) throw std::runtime_error("engram_disk: reader closed");
    if (n < 0) throw std::invalid_argument("engram_disk: negative row count");
    const int64_t* ids = reinterpret_cast<const int64_t*>(ids_ptr);
    uint8_t* w = reinterpret_cast<uint8_t*>(w_ptr);
    uint8_t* s = reinterpret_cast<uint8_t*>(s_ptr);
    pool_.run(n, [&](int64_t i) {
      std::unique_lock<std::mutex> cache_guard(cache_mu_, std::defer_lock);
      if (cache_rows_) cache_guard.lock();
      int64_t local = ids[i];
      uint8_t* dw = w + i * row_bytes_;
      uint8_t* ds = s + i * scale_bytes_;
      if (local == -1) { std::memset(dw, 0, row_bytes_); std::memset(ds, 0, scale_bytes_); return; }
      if (local < 0 || local >= num_rows_) throw std::out_of_range("engram_disk: invalid local row");
      if (cache_rows_) {
        int64_t slot = local % cache_rows_;
        if (__atomic_load_n(&tags_[slot], __ATOMIC_ACQUIRE) == local) {
          std::memcpy(dw, &cache_w_[slot * row_bytes_], row_bytes_);
          std::memcpy(ds, &cache_s_[slot * scale_bytes_], scale_bytes_);
          // a concurrent refill of this slot could have raced; re-check the tag after the copy
          if (__atomic_load_n(&tags_[slot], __ATOMIC_ACQUIRE) == local) { hits_.fetch_add(1); return; }
        }
      }
      const int64_t row = row_start_ + local;
      pread_full(fd_w_, dw, row_bytes_, off_w_ + row * row_bytes_);
      pread_full(fd_s_, ds, scale_bytes_, off_s_ + row * scale_bytes_);
      misses_.fetch_add(1);
      if (local < 0 || local >= num_rows_) throw std::out_of_range("engram_disk: invalid local row");
      if (cache_rows_) {
        int64_t slot = local % cache_rows_;
        __atomic_store_n(&tags_[slot], (int64_t)-2, __ATOMIC_RELEASE);  // invalidate while writing
        std::memcpy(&cache_w_[slot * row_bytes_], dw, row_bytes_);
        std::memcpy(&cache_s_[slot * scale_bytes_], ds, scale_bytes_);
        __atomic_store_n(&tags_[slot], local, __ATOMIC_RELEASE);
      }
    });
  }

  // Graph host nodes retain the Job* beyond this object's lifetime. Stable
  // tombstones prevent UAF; a replay after close exits the worker with code 74.
  void enqueue_read(int64_t stream, int64_t ids_ptr, int64_t w_ptr, int64_t s_ptr, int64_t n) {
    if (n < 0) throw std::invalid_argument("engram_disk: negative row count");
    if (spin_enabled_.load(std::memory_order_acquire) && n > 0 && n <= 512 &&
        spin_stream_compatible(stream, ids_ptr, w_ptr, s_ptr)) {
      // A retained graph must fail after close, not wait forever on a dead proxy.
      enqueue_read(stream, ids_ptr, w_ptr, s_ptr, 0);
      enqueue_spin(stream, ids_ptr, w_ptr, s_ptr, n);
      return;
    }
    Job* job;
    {
      std::lock_guard<std::mutex> guard(lifetime_mu_);
      if (closed_.load()) throw std::runtime_error("engram_disk: reader closed");
      std::lock_guard<std::mutex> jobs_guard(jobs_mu_);
      auto key = std::make_tuple(ids_ptr, w_ptr, s_ptr, n);
      auto& slot = jobs_[key];
      if (!slot) {
        slot = std::make_shared<Job>();
        slot->t = this;
        slot->ids = ids_ptr; slot->w = w_ptr; slot->s = s_ptr; slot->n = n;
        std::lock_guard<std::mutex> registry_guard(registry_mu());
        registry().push_back(slot);
      }
      job = slot.get();
    }
    cudaError_t e = cudaLaunchHostFunc(reinterpret_cast<cudaStream_t>(stream), &EngramDiskTable::host_fn, job);
    if (e != cudaSuccess) throw std::runtime_error(std::string("engram_disk: cudaLaunchHostFunc: ") + cudaGetErrorString(e));
  }
  // One stable buffer tuple per table; additional tuples fail rather than
  // racing graph replays. The proxy uses native AIO direct reads.
  std::string enable_spin(int64_t spin_ms) {
    std::lock_guard<std::mutex> guard(lifetime_mu_);
    if (closed_.load() || spin_enabled_.load() || spin_ms <= 0 || spin_ms > 60000 || !direct_io_)
      throw std::invalid_argument("engram_disk: invalid spin configuration");
    int64_t keepalive_us = 2000;
    if (const char* value = std::getenv("VLLM_ENGRAM_DISK_KEEPALIVE_US")) {
      errno = 0;
      char* end = nullptr;
      keepalive_us = std::strtoll(value, &end, 10);
      if (errno || end == value || *end || keepalive_us < 0 || keepalive_us > 60000000)
        throw std::invalid_argument("engram_disk: invalid keepalive us");
    }
    keepalive_ns_ = keepalive_us * 1000LL;
    const auto& ops = stream_memops();
    if (!ops.write || !ops.wait) throw std::runtime_error("engram_disk: no stream memory operations");
    void* ptr = nullptr;
    if (posix_memalign(&ptr, 4096, 4096)) throw std::bad_alloc();
    std::memset(ptr, 0, 4096);
    cudaError_t e = cudaHostRegister(ptr, 4096, cudaHostRegisterMapped | cudaHostRegisterPortable);
    if (e != cudaSuccess) { std::free(ptr); throw std::runtime_error("engram_disk: register spin words failed"); }
    void* dev = nullptr;
    e = cudaHostGetDevicePointer(&dev, ptr, 0);
    if (e != cudaSuccess) { cudaHostUnregister(ptr); std::free(ptr); throw std::runtime_error("engram_disk: map spin words failed"); }
    spin_ctl_ = static_cast<int32_t*>(ptr);
    spin_req_dev_ = reinterpret_cast<CUdeviceptr>(dev);
    spin_done_dev_ = spin_req_dev_ + 128;
    spin_window_ns_ = spin_ms * 1000000LL;
    try { shared_spin_proxy().add(this); }
    catch (...) { cudaHostUnregister(ptr); std::free(ptr); spin_ctl_ = nullptr; throw; }
    spin_enabled_.store(true, std::memory_order_release);
    return "spin proxy + native AIO O_DIRECT (experimental)";
  }

  std::vector<int64_t> stats() const { return {hits_.load(), misses_.load()}; }

 private:
  bool spin_stream_compatible(int64_t /*stream*/, int64_t ids, int64_t w, int64_t sc) {
    std::lock_guard<std::mutex> lock(spin_buffers_mu_);
    // The stream executing a captured graph is not necessarily the stream
    // used to capture it. Mixing proxy and callback on the same staging
    // buffers would permit overlapping reads and silent corruption.
    if (!spin_ids_) {
      spin_ids_ = ids; spin_w_ = w; spin_s_ = sc;
    }
    if (spin_ids_ != ids || spin_w_ != w || spin_s_ != sc)
      throw std::runtime_error("engram_disk: spin buffers changed");
    // CUDA graph replays may use a different stream from capture. The caller
    // must serialize replays and staging ownership across streams.
    return true;
  }
  void enqueue_spin(int64_t stream, int64_t ids, int64_t w, int64_t sc, int64_t n) {
    if (n <= 0) return;
    std::lock_guard<std::mutex> guard(lifetime_mu_);
    if (closed_.load()) throw std::runtime_error("engram_disk: reader closed");
    const auto& ops = stream_memops();
    auto st = reinterpret_cast<CUstream>(stream);
    CUresult r = ops.write(st, spin_req_dev_, static_cast<cuuint32_t>(n), CU_STREAM_WRITE_VALUE_DEFAULT);
    if (r == CUDA_SUCCESS) r = ops.wait(st, spin_done_dev_, 1, CU_STREAM_WAIT_VALUE_EQ);
    if (r == CUDA_SUCCESS) r = ops.write(st, spin_done_dev_, 0, CU_STREAM_WRITE_VALUE_DEFAULT);
    if (r != CUDA_SUCCESS) throw std::runtime_error("engram_disk: stream memory operation failed");
  }
  // Batch direct reads with Linux native AIO. A short or failed completion is
  // fatal: never silently switch to buffered I/O or publish partial rows.
  // A context is intentionally scoped to one request until lifecycle and
  // reusable context shutdown have been verified against graph replay.
  void read_rows_aio(int64_t ids_ptr, int64_t w_ptr, int64_t s_ptr, int64_t n) {
    std::lock_guard<std::mutex> guard(run_mu_);
    if (closed_.load() || n <= 0 || n > 512 || !direct_io_)
      throw std::runtime_error("engram_disk: invalid AIO request");
    struct Piece { int fd; int64_t off, len; uint8_t* dst; };
    constexpr int64_t align = 4096;
    const auto* ids = reinterpret_cast<const int64_t*>(ids_ptr);
    auto* w = reinterpret_cast<uint8_t*>(w_ptr);
    auto* sc = reinterpret_cast<uint8_t*>(s_ptr);
    std::vector<Piece> pieces;
    std::vector<std::pair<int64_t, int64_t>> pending;
    pieces.reserve(n * 2);
    pending.reserve(n);
    for (int64_t i = 0; i < n; ++i) {
      auto* dw = w + i * row_bytes_;
      auto* ds = sc + i * scale_bytes_;
      if (ids[i] == -1) { std::memset(dw, 0, row_bytes_); std::memset(ds, 0, scale_bytes_); continue; }
      if (ids[i] < 0 || ids[i] >= num_rows_) throw std::out_of_range("engram_disk: invalid AIO row");
      if (cache_rows_) {
        std::lock_guard<std::mutex> cache_guard(cache_mu_);
        const int64_t slot = ids[i] % cache_rows_;
        if (tags_[slot] == ids[i]) {
          std::memcpy(dw, &cache_w_[slot * row_bytes_], row_bytes_);
          std::memcpy(ds, &cache_s_[slot * scale_bytes_], scale_bytes_);
          hits_.fetch_add(1);
          continue;
        }
      }
      pending.emplace_back(i, ids[i]);
      int64_t row = row_start_ + ids[i];
      pieces.push_back({fd_w_, off_w_ + row * row_bytes_, row_bytes_, dw});
      pieces.push_back({fd_s_, off_s_ + row * scale_bytes_, scale_bytes_, ds});
    }
    if (pieces.empty()) return;
    aio_context_t context = 0;
    constexpr int depth = 64;
    if (syscall(SYS_io_setup, depth, &context) != 0)
      throw std::runtime_error("engram_disk: io_setup failed");
    struct ContextGuard { aio_context_t& ctx; ~ContextGuard() { if (ctx) syscall(SYS_io_destroy, ctx); } } context_guard{context};
    for (size_t base = 0; base < pieces.size(); base += depth) {
      const size_t count = std::min<size_t>(depth, pieces.size() - base);
      std::vector<iocb> cbs(count);
      std::vector<iocb*> ptrs(count);
      std::vector<void*> buffers(count, nullptr);
      struct BuffersGuard { aio_context_t& ctx; std::vector<void*>& v; bool done = false; ~BuffersGuard() { if (!done && ctx) { syscall(SYS_io_destroy, ctx); ctx = 0; } for (void* p : v) std::free(p); } } buffer_guard{context, buffers};
      for (size_t i = 0; i < count; ++i) {
        const auto& p = pieces[base+i];
        int64_t aligned_off = p.off / align * align;
        int64_t prefix = p.off - aligned_off;
        int64_t size = (prefix + p.len + align - 1) / align * align;
        if (posix_memalign(&buffers[i], align, size) != 0) throw std::bad_alloc();
        auto& cb = cbs[i]; std::memset(&cb, 0, sizeof(cb));
        cb.aio_lio_opcode = IOCB_CMD_PREAD; cb.aio_fildes = p.fd;
        cb.aio_buf = reinterpret_cast<uint64_t>(buffers[i]);
        cb.aio_nbytes = size; cb.aio_offset = aligned_off; cb.aio_data = i;
        ptrs[i] = &cb;
      }
      std::vector<bool> seen(count, false);
      size_t submitted = 0, completed = 0;
      while (completed < count) {
        if (submitted < count) {
          long nr = syscall(SYS_io_submit, context, count-submitted, ptrs.data()+submitted);
          if (nr < 0 && errno == EINTR) continue;
          if (nr < 0 && errno == EAGAIN && submitted > completed) nr = 0;
          if (nr < 0 || (nr == 0 && submitted == completed))
            throw std::runtime_error("engram_disk: io_submit failed");
          submitted += nr;
        }
        io_event events[depth];
        long nr = syscall(SYS_io_getevents, context, 1L, static_cast<long>(submitted-completed), events, nullptr);
        if (nr < 0 && errno == EINTR) continue;
        if (nr <= 0) throw std::runtime_error("engram_disk: io_getevents failed");
        for (long j = 0; j < nr; ++j) {
          size_t i = events[j].data;
          if (i >= submitted || seen[i])
            throw std::runtime_error("engram_disk: invalid or duplicate AIO completion");
          seen[i] = true;
          const auto& p = pieces[base+i];
          int64_t prefix = p.off % align;
          if (events[j].res < prefix + p.len || events[j].res2 != 0)
            throw std::runtime_error("engram_disk: AIO read failed or short");
          std::memcpy(p.dst, static_cast<uint8_t*>(buffers[i]) + prefix, p.len);
        }
        completed += nr;
      }
      buffer_guard.done = true;
    }
    misses_.fetch_add(static_cast<int64_t>(pending.size()));
    if (cache_rows_) {
      std::lock_guard<std::mutex> cache_guard(cache_mu_);
      for (const auto& [i, row] : pending) {
        const int64_t slot = row % cache_rows_;
        tags_[slot] = -2;
        std::memcpy(&cache_w_[slot * row_bytes_], w + i * row_bytes_, row_bytes_);
        std::memcpy(&cache_s_[slot * scale_bytes_], sc + i * scale_bytes_, scale_bytes_);
        tags_[slot] = row;
      }
    }
  }
  bool spin_poll() {
    int32_t n = __atomic_exchange_n(&spin_ctl_[0], 0, __ATOMIC_ACQ_REL);
    if (!n) return false;
    try {
      int64_t ids, w, sc;
      {
        std::lock_guard<std::mutex> lock(spin_buffers_mu_);
        ids = spin_ids_; w = spin_w_; sc = spin_s_;
      }
      if (n < 0 || n > 512 || !ids) throw std::runtime_error("invalid spin request");
      read_rows_aio(ids, w, sc, n);
    } catch (...) {
      constexpr char message[] = "engram_disk: fatal spin read failure\n";
      const auto ignored = ::write(STDERR_FILENO, message, sizeof(message)-1);
      (void)ignored; std::_Exit(74);
    }
    __atomic_store_n(&spin_ctl_[32], 1, __ATOMIC_RELEASE);
    return true;
  }
  void keepalive(std::chrono::steady_clock::time_point now) {
    if (!keepalive_ns_ || now - last_keepalive_ < std::chrono::nanoseconds(keepalive_ns_)) return;
    if (fd_w_ < 0) throw std::runtime_error("engram_disk: keepalive fd closed");
    if (!keepalive_buf_ && posix_memalign(&keepalive_buf_, 4096, 4096))
      throw std::bad_alloc();
    ssize_t n;
    do { n = ::pread(fd_w_, keepalive_buf_, 4096, 0); } while (n < 0 && errno == EINTR);
    if (n < 0) throw std::runtime_error(std::string("engram_disk: keepalive pread failed: ") + std::strerror(errno));
    // A valid table may be shorter than one sector (e.g. a tiny test shard).
    // Non-negative EOF is not an I/O error; real payload reads still fail closed.
    last_keepalive_ = now;
  }
  class SharedSpinProxy {
   public:
    SharedSpinProxy() : worker_([this] { loop(); }) {}
    void add(EngramDiskTable* t) {
      std::lock_guard<std::mutex> lock(mu_);
      tables_.push_back(t);
    }
    void remove(EngramDiskTable* t) {
      std::lock_guard<std::mutex> lock(mu_);
      tables_.erase(std::remove(tables_.begin(), tables_.end(), t), tables_.end());
    }
   private:
    void loop() {
      auto last = std::chrono::steady_clock::now();
      for (;;) {
        bool served = false;
        int64_t window = 0;
        {
          // Removal cannot race a poll or free a table under the worker.
          std::lock_guard<std::mutex> lock(mu_);
          for (auto* t : tables_) {
            served |= t->spin_poll();
            window = std::max(window, t->spin_window_ns_);
          }
          if (served) last = std::chrono::steady_clock::now();
          if (!served && !tables_.empty() && std::chrono::steady_clock::now() - last <= std::chrono::nanoseconds(window)) {
            try { for (auto* t : tables_) t->keepalive(std::chrono::steady_clock::now()); }
            catch (const std::exception& e) { std::string msg = std::string("engram_disk: fatal keepalive failure: ") + e.what() + "\n"; ::write(2, msg.c_str(), msg.size()); std::_Exit(74); }
          }
        }
        if (std::chrono::steady_clock::now() - last > std::chrono::nanoseconds(window))
          std::this_thread::sleep_for(std::chrono::microseconds(50));
        else std::this_thread::yield();
      }
    }
    std::mutex mu_;
    std::vector<EngramDiskTable*> tables_;
    std::thread worker_;
  };
  static SharedSpinProxy& shared_spin_proxy() {
    static auto* proxy = new SharedSpinProxy();
    return *proxy;
  }
  int64_t keepalive_ns_ = 0;
  void* keepalive_buf_ = nullptr;
  std::chrono::steady_clock::time_point last_keepalive_{};
  std::atomic<bool> spin_enabled_{false};
  std::mutex spin_buffers_mu_;
  int32_t* spin_ctl_ = nullptr;  // registered process-lifetime graph tombstone
  CUdeviceptr spin_req_dev_ = 0, spin_done_dev_ = 0;
  int64_t spin_ids_ = 0, spin_w_ = 0, spin_s_ = 0, spin_window_ns_ = 0;
  struct Job { std::mutex mu; EngramDiskTable* t = nullptr; int64_t ids, w, s, n; };
  static std::vector<std::shared_ptr<Job>>& registry() {
    static auto* jobs = new std::vector<std::shared_ptr<Job>>();
    return *jobs;
  }
  static std::mutex& registry_mu() {
    static auto* mu = new std::mutex();
    return *mu;
  }
  static void CUDART_CB host_fn(void* p) {
    auto* j = static_cast<Job*>(p);
    try {
      std::lock_guard<std::mutex> job_guard(j->mu);
      auto* t = j->t;
      if (!t) throw std::runtime_error("engram_disk: closed graph job");
      t->read_rows(j->ids, j->w, j->s, j->n);
    } catch (...) {
      const auto ignored = ::write(STDERR_FILENO, "engram_disk: fatal checkpoint read failure\n", sizeof("engram_disk: fatal checkpoint read failure\n") - 1);
      (void)ignored;
      std::_Exit(74);
    }
  }
  void pread_full(int fd, uint8_t* dst, int64_t len, int64_t off) {
    if (direct_io_) {
      pread_direct(fd, dst, len, off);
      return;
    }
    while (len > 0) {
      ssize_t r = ::pread(fd, dst, len, off);
      if (r <= 0) {
        if (r < 0 && errno == EINTR) continue;
        throw std::runtime_error(r == 0 ? "engram_disk: unexpected EOF" : "engram_disk: pread failed");
      }
      dst += r; len -= r; off += r;
    }
  }
  // 4096-aligned requests on the validated ext4 deployment volumes. A final
  // short read is legal only if it includes every byte of the requested row.
  // Never retry a short read at an unaligned address or fall back to buffered.
  static void pread_direct(int fd, uint8_t* dst, int64_t len, int64_t off) {
    constexpr int64_t align = 4096;
    if (len <= 0 || off < 0 || len > std::numeric_limits<int64_t>::max() - off - align)
      throw std::invalid_argument("engram_disk: invalid direct read range");
    const int64_t start = off / align * align;
    const int64_t prefix = off - start;
    const int64_t bytes = (prefix + len + align - 1) / align * align;
    void* raw = nullptr;
    if (::posix_memalign(&raw, align, bytes) != 0)
      throw std::runtime_error("engram_disk: direct bounce allocation failed");
    std::unique_ptr<void, decltype(&std::free)> bounce(raw, &std::free);
    ssize_t got;
    do { got = ::pread(fd, raw, bytes, start); } while (got < 0 && errno == EINTR);
    if (got < 0)
      throw std::runtime_error("engram_disk: O_DIRECT pread failed (no buffered fallback)");
    if (got < prefix + len)
      throw std::runtime_error("engram_disk: unexpected EOF in direct row");
    std::memcpy(dst, static_cast<uint8_t*>(raw) + prefix, len);
  }
  int fd_w_ = -1, fd_s_ = -1;
  int64_t off_w_, off_s_, row_start_, num_rows_, row_bytes_, scale_bytes_, cache_rows_ = 0;
  bool direct_io_ = false;
  std::vector<int64_t> tags_;
  std::vector<uint8_t> cache_w_, cache_s_;
  std::atomic<int64_t> hits_{0}, misses_{0};
  std::mutex jobs_mu_, run_mu_, cache_mu_, lifetime_mu_;
  std::atomic<bool> closed_{false};
  std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, std::shared_ptr<Job>> jobs_;
  Pool pool_;
};

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  pybind11::class_<EngramDiskTable>(m, "EngramDiskTable")
      .def(pybind11::init<const std::string&, int64_t, const std::string&, int64_t, int64_t, int64_t, int64_t, int64_t,
                          int64_t, int, bool>(),
           pybind11::arg("weight_file"), pybind11::arg("weight_offset"), pybind11::arg("scale_file"),
           pybind11::arg("scale_offset"), pybind11::arg("row_start"), pybind11::arg("num_rows"),
           pybind11::arg("row_bytes"), pybind11::arg("scale_bytes"), pybind11::arg("cache_rows") = 0,
           pybind11::arg("threads") = 32, pybind11::arg("direct_io") = false)
      .def("read_rows", &EngramDiskTable::read_rows, pybind11::call_guard<pybind11::gil_scoped_release>())
      .def("enqueue_read", &EngramDiskTable::enqueue_read)
      .def("enable_spin", &EngramDiskTable::enable_spin)
      .def("stats", &EngramDiskTable::stats)
      .def("close", &EngramDiskTable::close, pybind11::call_guard<pybind11::gil_scoped_release>());
}
