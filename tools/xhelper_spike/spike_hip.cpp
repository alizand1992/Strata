// tools/xhelper_spike/spike_hip.cpp - the AMD half of the Phase 1 spike.
//
// The other half is spike_cuda.cu: a CUDA process makes a memfd, maps it, registers it with CUDA, and forks this
// program with the fd.  Here the same pages are mapped again and registered with HIP, and the two programs ping-pong
// through it.  Nothing in this file belongs to the engine: Phase 1 exists to find out whether this transport works at
// all before any engine code is written (docs/plans/OPTION_A_EXCLUSIVE_RAM.md Gate 1).  The engine's real child is
// src/xhelper/helper_main.cpp, Phase 3.
//
// What it answers, in the order Gate 1 asks:
//   1. does hipHostRegister accept a mapping CUDA already registered?  (record and continue, never abort)
//   2. does data written by the CUDA card arrive byte-correct, read here with hipMemcpyAsync and checked on the GPU,
//      and the other way round, and what does the zero-copy alias give?
//   3. H2D / D2H bandwidth through the shared mapping, against a private hipHostMalloc buffer.
//   4. round-trip latency, spin / futex.
//   5. a round trip that carries 80 KiB in and 200 KiB back through a kernel.
//
// Build: hipcc -O3 -std=c++20 --offload-arch=gfx1200 spike_hip.cpp -o strata-xh-spike-hip   (run.sh does it)
// Run:   strata-xh-spike-hip --shm-fd N     (the parent forks it; HIP_VISIBLE_DEVICES picks the card)

#include "spike_layout.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace xhelper_spike;

#define HIP_CHECK(call)                                                                                                  \
    do {                                                                                                                 \
        const hipError_t err_ = (call);                                                                                  \
        if (err_ != hipSuccess) {                                                                                        \
            std::snprintf(ctl->child_error, sizeof(ctl->child_error), "%s:%d %s", __FILE__, __LINE__,                   \
                          hipGetErrorString(err_));                                                                      \
            std::fprintf(stderr, "[hip-spike] %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err_));                \
            ctl->state.store(SPIKE_FAILED);                                                                              \
            spike_wake(ctl->done);                                                                                       \
            return 3;                                                                                                    \
        }                                                                                                                \
    } while (0)

// Kernels: fill a buffer with the pattern, check it, and do the small amount of work test 5 wants.
__global__ void hip_fill_pattern(uint32_t* p, uint64_t n) {
    const uint64_t stride = (uint64_t) blockDim.x * gridDim.x;
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        p[i] = spike_pattern(i);
}

__global__ void hip_check_pattern(const uint32_t* p, uint64_t n, unsigned long long* sum, unsigned long long* bad) {
    unsigned long long s = 0, b = 0;
    const uint64_t stride = (uint64_t) blockDim.x * gridDim.x;
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const uint32_t v = p[i];
        s += (unsigned long long) v;
        if (v != spike_pattern(i)) ++b;
    }
    atomicAdd(sum, s);
    atomicAdd(bad, b);
}

__global__ void hip_publish_sum(const unsigned long long* sum, unsigned long long* host_sum) { *host_sum = *sum; }

__global__ void hip_work_on(const uint32_t* in, uint64_t nin, uint32_t* out, uint64_t nout,
                            unsigned long long* sum) {
    unsigned long long s = 0;
    const uint64_t stride = (uint64_t) blockDim.x * gridDim.x;
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < nout; i += stride) {
        const uint32_t v = in[i % nin] * 2u + 1u;
        out[i] = v;
        s += (unsigned long long) v;
    }
    atomicAdd(sum, s);
}

static uint64_t grid_for(uint64_t n, uint32_t block = 256) {
    const uint64_t g = (n + block - 1) / block;
    return g > 8192 ? 8192 : (g ? g : 1);
}

static uint64_t pct(const std::vector<uint64_t>& v, double q) {
    if (v.empty()) return 0;
    size_t i = (size_t) (q * (double) (v.size() - 1) + 0.5);
    return v[i];
}

struct Stats {
    std::vector<uint64_t> us;
    uint64_t bytes = 0;
    void finish(Sample& s) {
        std::sort(us.begin(), us.end());
        s.bytes = bytes;
        s.p50_us = pct(us, 0.5);
        s.p99_us = pct(us, 0.99);
        double sum_us = 0;
        for (uint64_t u : us) sum_us += (double) u;
        // GiB/s over the whole run: bytes moved divided by the time it took.
        s.gib_s = sum_us > 0 ? ((double) bytes * (double) us.size()) / (sum_us * 1000.0) * 1e9 / 1073741824.0 : 0.0;
    }
};

int main(int argc, char** argv) {
    int shm_fd = -1;
    uint32_t spin_us = 200;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--shm-fd" && i + 1 < argc) shm_fd = std::atoi(argv[++i]);
        else if (a == "--spin-us" && i + 1 < argc) spin_us = (uint32_t) std::atoi(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else {
            std::fprintf(stderr, "[hip-spike] unknown argument %s\n", a.c_str());
            return 2;
        }
    }
    if (shm_fd < 0) {
        std::fprintf(stderr, "[hip-spike] --shm-fd N is required\n");
        return 2;
    }

    struct stat st{};
    if (fstat(shm_fd, &st) != 0 || st.st_size < (off_t) kTotalBytes) {
        std::fprintf(stderr, "[hip-spike] the inherited fd is not a %llu byte region\n",
                     (unsigned long long) kTotalBytes);
        return 2;
    }
    void* base = mmap(nullptr, kTotalBytes, PROT_READ | PROT_WRITE, MAP_SHARED, shm_fd, 0);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "[hip-spike] mmap: %s\n", std::strerror(errno));
        return 2;
    }
    Ctl* ctl = reinterpret_cast<Ctl*> (reinterpret_cast<char*> (base) + ctl_off());
    Report* rep = reinterpret_cast<Report*> (reinterpret_cast<char*> (base) + report_off());
    if (ctl->magic != kMagic || ctl->version != kVersion) {
        std::fprintf(stderr, "[hip-spike] the region is not a spike region (magic %08x version %u)\n", ctl->magic,
                     ctl->version);
        return 2;
    }
    char* a_region = (char*) base + ctl->a_off;
    char* b_region = (char*) base + ctl->b_off;
    char* in_region = (char*) base + ctl->in_off;
    char* out_region = (char*) base + ctl->out_off;

    HIP_CHECK(hipSetDevice(0));   // HIP_VISIBLE_DEVICES has already narrowed this process to the AMD card
    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    if (verbose) std::fprintf(stderr, "[hip-spike] device 0 = %s, %llu MiB\n", prop.name,
                             (unsigned long long) (prop.totalGlobalMem / kMiB));

    // ---- 1. the whole point of Phase 1: can HIP register pages CUDA already registered?
    const hipError_t reg = hipHostRegister(base, kTotalBytes, hipHostRegisterPortable | hipHostRegisterMapped);
    ctl->child_register = reg == hipSuccess ? SPIKE_REG_OK : SPIKE_REG_FAILED;
    ctl->child_register_err = (int32_t) reg;
    const bool registered = reg == hipSuccess;
    if (!registered)
        std::snprintf(ctl->child_error, sizeof(ctl->child_error), "hipHostRegister refused this mapping: %s",
                      hipGetErrorString(reg));
    if (verbose)
        std::fprintf(stderr, "[hip-spike] hipHostRegister(%llu MiB): %s\n", (unsigned long long) (kTotalBytes / kMiB),
                     registered ? "ok" : hipGetErrorString(reg));

    void* dev_alias = nullptr;
    if (registered) {
        const hipError_t g = hipHostGetDevicePointer(&dev_alias, base, 0);
        ctl->child_zero_copy = g == hipSuccess ? 1u : 0u;
    }

    // Scratch on the AMD side.
    uint32_t* dev_buf = nullptr;   // big enough for a 64 MiB region
    uint32_t* dev_in = nullptr;
    uint32_t* dev_out = nullptr;
    unsigned long long* dev_sum = nullptr;
    unsigned long long* dev_bad = nullptr;
    HIP_CHECK(hipMalloc(&dev_buf, kRegionBytes));
    HIP_CHECK(hipMalloc(&dev_in, kWorkInBytes));
    HIP_CHECK(hipMalloc(&dev_out, kWorkOutBytes));
    HIP_CHECK(hipMalloc(&dev_sum, 2 * sizeof(unsigned long long)));   // [0] the checksum, [1] the mismatch count
    dev_bad = dev_sum + 1;
    hipStream_t stream{};
    // Test 5's stage timing, only with SPIKE_STAGES=1: the events cost the round ~35 us on the RX 9060 XT
    const bool stages = std::getenv("SPIKE_STAGES") && std::getenv("SPIKE_STAGES")[0] == '1';
    hipEvent_t ev[4]{};
    if (stages)
        for (auto& e : ev) HIP_CHECK(hipEventCreate(&e));
    HIP_CHECK(hipStreamCreate(&stream));

    // A private pinned buffer, the comparison the bandwidth test is against.
    void* priv = nullptr;
    const size_t priv_bytes =
        hipHostMalloc(&priv, kRegionBytes, hipHostMallocPortable) == hipSuccess ? kRegionBytes : 0;

    rep->magic = kMagic;
    std::snprintf(rep->device_name, sizeof(rep->device_name), "%.48s", prop.name);
    ctl->state.store(SPIKE_READY);
    spike_wake(ctl->done);

    uint32_t last_seq = 0;
    bool running = true;
    while (running) {
        // SPIKE_BOTH_SPIN: both sides spin.  SPIKE_CHILD_FUTEX: this side sleeps, the parent spins.
        // SPIKE_BOTH_FUTEX: both sleep.  These are the three rows test 4 prints.
        const uint32_t use_futex = ctl->variant == SPIKE_BOTH_SPIN ? 0u : 1u;
        const uint64_t deadline = spike_now_ns() + 120ull * 1000000000ull;
        const uint32_t spin = ctl->spin_us ? ctl->spin_us : spin_us;
        const uint32_t s = spike_wait(ctl->seq, spin, use_futex != 0, deadline,
                                      [&](uint32_t v) { return v != last_seq; });
        if (s == 0) {   // nobody is there any more
            if (getppid() == 1) break;
            continue;
        }
        last_seq = s;
        const uint32_t cmd = ctl->cmd.load();
        const uint64_t t_seen = spike_now_ns();

        switch (cmd) {
            case SPIKE_CHECK_A: {
                if (!registered) break;   // without a registration there is nothing to copy into
                const uint64_t n = ctl->a_bytes / 4;
                HIP_CHECK(hipMemsetAsync(dev_sum, 0, 2 * sizeof(unsigned long long), stream));
                HIP_CHECK(hipMemcpyAsync(dev_buf, a_region, ctl->a_bytes, hipMemcpyHostToDevice, stream));
                hip_check_pattern<<<(unsigned) grid_for(n), 256, 0, stream>>>(dev_buf, n, dev_sum, dev_bad);
                HIP_CHECK(hipStreamSynchronize(stream));
                unsigned long long out[2] = {0, 0};
                HIP_CHECK(hipMemcpy(&out, dev_sum, 2 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
                ctl->child_checksum_a = out[0];
                ctl->child_mismatch_a = out[1];
                break;
            }
            case SPIKE_CHECK_B: {
                if (!registered) break;
                const uint64_t n = ctl->b_bytes / 4;
                hip_fill_pattern<<<(unsigned) grid_for(n), 256, 0, stream>>>(dev_buf, n);
                HIP_CHECK(hipMemsetAsync(dev_sum, 0, 2 * sizeof(unsigned long long), stream));
                hip_check_pattern<<<(unsigned) grid_for(n), 256, 0, stream>>>(dev_buf, n, dev_sum, dev_bad);
                HIP_CHECK(hipMemcpyAsync(b_region, dev_buf, ctl->b_bytes, hipMemcpyDeviceToHost, stream));
                HIP_CHECK(hipStreamSynchronize(stream));
                unsigned long long out[2] = {0, 0};
                HIP_CHECK(hipMemcpy(&out, dev_sum, 2 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
                ctl->child_checksum_b = out[0];
                ctl->child_mismatch_b = out[1];
                break;
            }
            case SPIKE_ZERO_COPY: {
                if (!dev_alias) break;   // no alias, nothing to test
                const uint64_t na = ctl->a_bytes / 4, nb = ctl->b_bytes / 4;
                uint32_t* dev_a = reinterpret_cast<uint32_t*> (reinterpret_cast<char*> (dev_alias) + ctl->a_off);
                uint32_t* dev_b = reinterpret_cast<uint32_t*> (reinterpret_cast<char*> (dev_alias) + ctl->b_off);
                HIP_CHECK(hipMemsetAsync(dev_sum, 0, 2 * sizeof(unsigned long long), stream));
                hip_check_pattern<<<(unsigned) grid_for(na), 256, 0, stream>>>(dev_a, na, dev_sum, dev_bad);
                HIP_CHECK(hipStreamSynchronize(stream));
                unsigned long long out[2] = {0, 0};
                HIP_CHECK(hipMemcpy(&out, dev_sum, 2 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
                ctl->child_checksum_zero = out[0];
                ctl->child_mismatch_zero = out[1];
                // and the other direction, with no memcpy at all
                hip_fill_pattern<<<(unsigned) grid_for(nb), 256, 0, stream>>>(dev_b, nb);
                HIP_CHECK(hipStreamSynchronize(stream));
                break;
            }
            case SPIKE_BANDWIDTH: {
                if (!registered) break;
                const uint64_t sizes[3] = {160 * 1024ull, 800 * 1024ull, 64ull * kMiB};
                rep->sizes_n = 3;
                for (int i = 0; i < 3; ++i) rep->size[i] = sizes[i];
                auto bench = [&](void* src, void* dst, uint64_t bytes, Stats& up, Stats& down) -> int {
                    up.bytes = bytes;
                    down.bytes = bytes;
                    for (int r = 0; r < 50; ++r) {
                        uint64_t t0 = spike_now_ns();
                        HIP_CHECK(hipMemcpyAsync(dst, src, bytes, hipMemcpyHostToDevice, stream));
                        HIP_CHECK(hipStreamSynchronize(stream));
                        up.us.push_back((spike_now_ns() - t0) / 1000);
                        t0 = spike_now_ns();
                        HIP_CHECK(hipMemcpyAsync(src, dst, bytes, hipMemcpyDeviceToHost, stream));
                        HIP_CHECK(hipStreamSynchronize(stream));
                        down.us.push_back((spike_now_ns() - t0) / 1000);
                    }
                    return 0;
                };
                for (int i = 0; i < 3; ++i) {
                    Stats s, d;
                    if (bench(a_region, dev_buf, std::min(sizes[i], kRegionBytes), s, d)) return 3;
                    s.finish(rep->h2d_shared[i]);
                    d.finish(rep->d2h_shared[i]);
                    if (priv_bytes >= sizes[i]) {
                        Stats ps, pd;
                        if (bench(priv, dev_buf, sizes[i], ps, pd)) return 3;
                        ps.finish(rep->h2d_private[i]);
                        pd.finish(rep->d2h_private[i]);
                    }
                }
                break;
            }
            case SPIKE_PING: {
                // The parent measures the round trip.  This side only records how long the wait looked from here.
                ctl->done.store(s);
                spike_wake(ctl->done);
                const uint32_t v = ctl->variant;
                if (v < 4) {
                    rep->ping_wait_sum_us[v] += (spike_now_ns() - t_seen) / 1000;
                    rep->ping_rounds++;
                }
                running = ctl->quit.load() == 0;
                continue;   // done is already set for this seq
            }
            case SPIKE_WORK: {
                if (!registered) break;
                const uint64_t nin = ctl->in_bytes / 4, nout = ctl->out_bytes / 4;
                const uint64_t t_start = spike_now_ns();
                if (stages) HIP_CHECK(hipEventRecord(ev[0], stream));
                HIP_CHECK(hipMemsetAsync(dev_sum, 0, 2 * sizeof(unsigned long long), stream));
                HIP_CHECK(hipMemcpyAsync(dev_in, in_region, ctl->in_bytes, hipMemcpyHostToDevice, stream));
                if (stages) HIP_CHECK(hipEventRecord(ev[1], stream));
                hip_work_on<<<(unsigned) grid_for(nout), 256, 0, stream>>>(dev_in, nin, dev_out, nout, dev_sum);
                if (stages) HIP_CHECK(hipEventRecord(ev[2], stream));
                HIP_CHECK(hipMemcpyAsync(out_region, dev_out, ctl->out_bytes, hipMemcpyDeviceToHost, stream));
                if (stages) HIP_CHECK(hipEventRecord(ev[3], stream));
                HIP_CHECK(hipStreamSynchronize(stream));
                const uint64_t t_synced = spike_now_ns();
                unsigned long long out[2] = {0, 0};
                HIP_CHECK(hipMemcpy(&out, dev_sum, 2 * sizeof(unsigned long long), hipMemcpyDeviceToHost));
                const uint64_t t_end = spike_now_ns();
                ctl->child_work_sum = out[0];
                for (int k = 0; stages && k < 3; ++k) {
                    float ms = 0.f;
                    HIP_CHECK(hipEventElapsedTime(&ms, ev[k], ev[k + 1]));
                    rep->work_stage_ns[k] += (uint64_t) (ms * 1e6f);
                }
                rep->work_stage_ns[3] += t_end - t_synced;
                rep->work_stage_ns[4] += t_end - t_start;
                rep->work_rounds++;
                break;
            }
            case SPIKE_WORK_ZERO: {
                if (!registered || !dev_alias) break;
                const uint64_t nin = ctl->in_bytes / 4, nout = ctl->out_bytes / 4;
                char* alias = reinterpret_cast<char*> (dev_alias);
                const uint32_t* in_dev = reinterpret_cast<const uint32_t*> (alias + (in_region - (char*) base));
                uint32_t* out_dev = reinterpret_cast<uint32_t*> (alias + (out_region - (char*) base));
                auto* sum_dev = reinterpret_cast<unsigned long long*> (
                    alias + (reinterpret_cast<char*> (&ctl->child_work_sum) - (char*) base));
                HIP_CHECK(hipMemsetAsync(dev_sum, 0, 2 * sizeof(unsigned long long), stream));
                hip_work_on<<<(unsigned) grid_for(nout), 256, 0, stream>>>(in_dev, nin, out_dev, nout, dev_sum);
                hip_publish_sum<<<1, 1, 0, stream>>>(dev_sum, sum_dev);
                HIP_CHECK(hipStreamSynchronize(stream));
                break;
            }
            case SPIKE_QUIT:
                running = false;
                break;
            default:
                break;
        }
        ctl->done.store(s);
        spike_wake(ctl->done);
    }

    ctl->state.store(SPIKE_EXITED);
    spike_wake(ctl->done);
    if (priv) hipHostFree(priv);
    hipStreamDestroy(stream);
    hipFree(dev_buf);
    hipFree(dev_in);
    hipFree(dev_out);
    hipFree(dev_sum);
    if (ctl->child_register == SPIKE_REG_OK) hipHostUnregister(base);
    return 0;
}
