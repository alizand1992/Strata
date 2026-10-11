// tools/xhelper_spike/spike_cuda.cu - the NVIDIA half of the Phase 1 spike, and the one that runs.
//
// What this is: the smallest thing that can answer the questions docs/plans/OPTION_A_EXCLUSIVE_RAM.md Phase 1 asks,
// before any engine code exists.  It makes a memfd, maps 256 MiB of it, registers it with CUDA, forks the HIP half
// (spike_hip.cpp) with the fd, and drives five tests over that one mapping:
//   1  dual registration      - cudaHostRegister and hipHostRegister on the same pages, with the error recorded
//   2  integrity              - 64 MiB each way through the API, and through the zero-copy alias, checked on the GPU
//                               and again on the host, with an exact integer pattern so a PASS means byte-equal
//   3  bandwidth              - 160 KiB, 800 KiB, 64 MiB, 50 reps, p50/p99 and GiB/s, against a private pinned buffer
//   4  round-trip latency     - 10000 pings on two futex words, spin / child-futex / both-futex
//   5  round trip with work   - 80 KiB in, 128 kernel on the AMD card, 200 KiB back, timed end to end
//
// That is the whole engine/helper transport in miniature.  Nothing here is used by the engine: the real channel is
// include/strata/xhelper/protocol.hpp (Phase 2) and src/xhelper/helper_main.cpp (Phase 3).  If Gate 1 fails, Option A
// stops here, which is the point of writing 600 throwaway lines first.
//
// Build: nvcc -O3 -std=c++20 spike_cuda.cu -o strata-xh-spike-cuda   (run.sh does it)
// Run:   ./run.sh
// Needs: an NVIDIA card this process owns, the HIP half built, Linux.  Nothing else.

#include "spike_layout.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace xhelper_spike;

#define CUDA_CHECK(call)                                                                                                 \
    do {                                                                                                                 \
        const cudaError_t err_ = (call);                                                                                 \
        if (err_ != cudaSuccess) {                                                                                       \
            std::fprintf(stderr, "[cuda-spike] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_));             \
            return 1;                                                                                                    \
        }                                                                                                                \
    } while (0)

__global__ void cuda_fill_pattern(uint32_t* p, uint64_t n) {
    const uint64_t stride = (uint64_t) blockDim.x * gridDim.x;
    for (uint64_t i = (uint64_t) blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        p[i] = spike_pattern(i);
}

__global__ void cuda_check_pattern(const uint32_t* p, uint64_t n, unsigned long long* sum, unsigned long long* bad) {
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

static uint64_t grid_for(uint64_t n, uint32_t block = 256) {
    const uint64_t g = (n + block - 1) / block;
    return g > 8192 ? 8192 : (g ? g : 1);
}

static uint64_t pct(const std::vector<uint64_t>& sorted, double q) {
    if (sorted.empty()) return 0;
    return sorted[(size_t) (q * (double) (sorted.size() - 1) + 0.5)];
}

static void summarize(std::vector<uint64_t> us, uint64_t bytes, Sample& s) {
    std::sort(us.begin(), us.end());
    s.bytes = bytes;
    s.p50_us = pct(us, 0.5);
    s.p99_us = pct(us, 0.99);
    double sum_us = 0;
    for (uint64_t u : us) sum_us += (double) u;
    s.gib_s = sum_us > 0 ? ((double) bytes * (double) us.size()) / (sum_us * 1000.0) * 1e9 / 1073741824.0 : 0.0;
}

static void print_sample(const char* label, const Sample& s) {
    std::printf("  %-26s %8llu KiB  p50 %8.1f us  p99 %9.1f us  %7.2f GiB/s\n", label,
                (unsigned long long) (s.bytes / 1024), (double) s.p50_us, (double) s.p99_us, s.gib_s);
}

// Host-side check of a region: exact, and independent of either card.
static uint64_t host_mismatches(const void* p, uint64_t bytes) {
    const uint32_t* v = (const uint32_t*) p;
    const uint64_t n = bytes / 4;
    uint64_t bad = 0;
    for (uint64_t i = 0; i < n; ++i)
        if (v[i] != spike_pattern(i)) ++bad;
    return bad;
}

static int send_command(Ctl* ctl, uint32_t cmd, uint32_t spin_us, bool parent_futex, uint64_t timeout_ms) {
    ctl->cmd.store(cmd);
    const uint32_t seq = ctl->seq.load() + 1;
    ctl->seq.store(seq);
    spike_wake(ctl->seq);
    const uint64_t deadline = spike_now_ns() + timeout_ms * 1000000ull;
    const uint32_t got = spike_wait(ctl->done, spin_us, parent_futex, deadline, [&](uint32_t v) { return v == seq; });
    return got == seq ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string child = "./strata-xh-spike-hip";
    int hip_device = 0;
    uint32_t rounds = 10000;
    uint32_t work_rounds = 500;
    uint32_t spin_us = 200;
    bool verbose = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--child" && i + 1 < argc) child = argv[++i];
        else if (a == "--hip-device" && i + 1 < argc) hip_device = std::atoi(argv[++i]);
        else if (a == "--rounds" && i + 1 < argc) rounds = (uint32_t) std::atoi(argv[++i]);
        else if (a == "--work-rounds" && i + 1 < argc) work_rounds = (uint32_t) std::atoi(argv[++i]);
        else if (a == "--spin-us" && i + 1 < argc) spin_us = (uint32_t) std::atoi(argv[++i]);
        else if (a == "--verbose") verbose = true;
        else {
            std::fprintf(stderr, "usage: %s [--child PATH] [--hip-device N] [--rounds N] [--work-rounds N]"
                                 " [--spin-us N] [--verbose]\n",
                         argv[0]);
            return 2;
        }
    }

    // ---- the mapping the two processes share
    const int fd = (int) syscall(SYS_memfd_create, "strata-xh-spike", 0);
    if (fd < 0) {
        std::fprintf(stderr, "[cuda-spike] memfd_create: %s\n", std::strerror(errno));
        return 1;
    }
    if (ftruncate(fd, (off_t) kTotalBytes) != 0) {
        std::fprintf(stderr, "[cuda-spike] ftruncate: %s\n", std::strerror(errno));
        return 1;
    }
    void* base = mmap(nullptr, kTotalBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "[cuda-spike] mmap: %s\n", std::strerror(errno));
        return 1;
    }
    std::memset(base, 0, kTotalBytes);
    Ctl* ctl = reinterpret_cast<Ctl*> (reinterpret_cast<char*> (base) + ctl_off());
    Report* rep = reinterpret_cast<Report*> (reinterpret_cast<char*> (base) + report_off());
    char* a_region = (char*) base + a_off();
    char* b_region = (char*) base + b_off();
    char* in_region = (char*) base + in_off();
    char* out_region = (char*) base + out_off();
    ctl->magic = kMagic;
    ctl->version = kVersion;
    ctl->total_bytes = kTotalBytes;
    ctl->a_off = a_off();
    ctl->a_bytes = kRegionBytes;
    ctl->b_off = b_off();
    ctl->b_bytes = kRegionBytes;
    ctl->in_off = in_off();
    ctl->in_bytes = kWorkInBytes;
    ctl->out_off = out_off();
    ctl->out_bytes = kWorkOutBytes;
    ctl->report_off = report_off();
    ctl->rounds = rounds;
    ctl->spin_us = spin_us;

    // ---- 1. dual registration, part one
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const cudaError_t reg = cudaHostRegister(base, kTotalBytes, cudaHostRegisterPortable | cudaHostRegisterMapped);
    ctl->parent_register = reg == cudaSuccess ? SPIKE_REG_OK : SPIKE_REG_FAILED;
    ctl->parent_register_err = (int32_t) reg;
    char parent_reg_error[128] = "";
    if (reg != cudaSuccess) std::snprintf(parent_reg_error, sizeof(parent_reg_error), "%s", cudaGetErrorString(reg));
    void* dev_alias = nullptr;
    if (reg == cudaSuccess)
        ctl->parent_zero_copy =
            cudaHostGetDevicePointer(&dev_alias, base, 0) == cudaSuccess ? 1u : 0u;

    // ---- start the HIP half with the fd
    fflush(nullptr);
    const pid_t kid = fork();
    if (kid < 0) {
        std::fprintf(stderr, "[cuda-spike] fork: %s\n", std::strerror(errno));
        return 1;
    }
    if (kid == 0) {
        char fd_arg[16], spin_arg[16], dev_arg[16];
        std::snprintf(fd_arg, sizeof(fd_arg), "%d", fd);
        std::snprintf(spin_arg, sizeof(spin_arg), "%u", spin_us);
        std::snprintf(dev_arg, sizeof(dev_arg), "%d", hip_device);
        setenv("HIP_VISIBLE_DEVICES", dev_arg, 1);
        setenv("CUDA_VISIBLE_DEVICES", "", 1);   // the helper must never see the NVIDIA card
        execl(child.c_str(), child.c_str(), "--shm-fd", fd_arg, "--spin-us", spin_arg, verbose ? "--verbose" : nullptr,
              (char*) nullptr);
        std::fprintf(stderr, "[cuda-spike] exec %s: %s\n", child.c_str(), std::strerror(errno));
        _exit(127);
    }

    std::printf("=== Option A Phase 1 spike ===\n");
    std::printf("NVIDIA  device 0 : %s\n", prop.name);
    std::printf("HIP     device   : HIP_VISIBLE_DEVICES=%d (the child's device 0)\n", hip_device);
    std::printf("mapping          : memfd %d, %llu MiB, spin_us %u\n", fd, (unsigned long long) (kTotalBytes / kMiB),
                spin_us);

    // Wait for the child's setup.  HIP init on a real card takes seconds, so give it plenty.
    bool kid_ready = false;
    for (uint32_t waited = 0; waited < 120000; waited += 100) {
        if (ctl->state.load() == SPIKE_READY) { kid_ready = true; break; }
        if (ctl->state.load() == SPIKE_FAILED) break;
        usleep(100 * 1000);   // 100 ms a step: 120 s in all (usleep takes microseconds)
    }
    if (!kid_ready) {
        std::printf("\nTEST 1  dual registration : the HIP half never got ready\n");
        std::printf("  child error : %s\n", ctl->child_error[0] ? ctl->child_error : "(none)");
        std::printf("  parent cudaHostRegister : %s\n",
                    ctl->parent_register == SPIKE_REG_OK ? "ok" : parent_reg_error);
        kill(kid, SIGTERM);
        int st = 0;
        waitpid(kid, &st, 0);
        return 1;
    }

    // Scratch on the NVIDIA side.
    uint32_t* dev_buf = nullptr;
    unsigned long long* dev_sum = nullptr;
    CUDA_CHECK(cudaMalloc(&dev_buf, kRegionBytes));
    CUDA_CHECK(cudaMalloc(&dev_sum, 2 * sizeof(unsigned long long)));
    unsigned long long* dev_bad = dev_sum + 1;
    cudaStream_t stream{};
    CUDA_CHECK(cudaStreamCreate(&stream));
    auto read_counters = [&](unsigned long long& sum, unsigned long long& bad) -> int {
        unsigned long long out[2] = {0, 0};
        const cudaError_t e = cudaMemcpy(&out, dev_sum, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "[cuda-spike] reading the counters: %s\n", cudaGetErrorString(e));
            return 1;
        }
        sum = out[0];
        bad = out[1];
        return 0;
    };

    // ---- TEST 1
    std::printf("\nTEST 1  dual registration (same %llu MiB mapping, both APIs)\n",
                (unsigned long long) (kTotalBytes / kMiB));
    std::printf("  cudaHostRegister  : %s\n", ctl->parent_register == SPIKE_REG_OK ? "ok" : parent_reg_error);
    std::printf("  hipHostRegister   : %s\n",
                ctl->child_register == SPIKE_REG_OK ? "ok"
                                                    : (ctl->child_error[0] ? ctl->child_error : "failed"));
    std::printf("  zero-copy alias   : CUDA %s, HIP %s\n",
                ctl->parent_zero_copy ? "cudaHostGetDevicePointer ok" : "no",
                ctl->child_zero_copy ? "hipHostGetDevicePointer ok" : "no");
    const bool both_registered = ctl->parent_register == SPIKE_REG_OK && ctl->child_register == SPIKE_REG_OK;
    std::printf("  verdict           : %s\n", both_registered ? "PASS - one mapping, both cards"
                                                             : "FAIL - Option A needs a bounce buffer, see the plan");

    // ---- TEST 2  integrity
    std::printf("\nTEST 2  integrity (64 MiB each way, pattern checked on the GPU and on the host)\n");
    unsigned long long sum = 0, bad = 0;
    CUDA_CHECK(cudaMemset(dev_sum, 0, 2 * sizeof(unsigned long long)));
    cuda_fill_pattern<<<(unsigned) grid_for(kRegionBytes / 4), 256, 0, stream>>>(dev_buf, kRegionBytes / 4);
    cuda_check_pattern<<<(unsigned) grid_for(kRegionBytes / 4), 256, 0, stream>>>(dev_buf, kRegionBytes / 4, dev_sum,
                                                                                 dev_bad);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (read_counters(sum, bad)) return 1;
    const unsigned long long pattern_sum = sum;
    if (bad) std::printf("      the NVIDIA card's own fill did not match the pattern (%llu bad words)\n",
                         (unsigned long long) bad);
    CUDA_CHECK(cudaMemcpyAsync(a_region, dev_buf, kRegionBytes, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const uint64_t host_bad_a = host_mismatches(a_region, kRegionBytes);
    const int r1 = send_command(ctl, SPIKE_CHECK_A, spin_us, false, 60000);
    std::printf("  CUDA writes, HIP reads  (A) : %s  HIP checksum %s  HIP mismatch %llu\n",
                r1 == 0 ? "answered" : "TIMEOUT",
                r1 == 0 && ctl->child_checksum_a == pattern_sum ? "matches" : "differs",
                (unsigned long long) ctl->child_mismatch_a);
    std::printf("      host-side re-check of region A : %s (%llu bad words)\n",
                host_bad_a == 0 ? "byte-exact" : "CORRUPT", (unsigned long long) host_bad_a);

    const int r2 = send_command(ctl, SPIKE_CHECK_B, spin_us, false, 60000);
    CUDA_CHECK(cudaMemset(dev_sum, 0, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpyAsync(dev_buf, b_region, kRegionBytes, cudaMemcpyHostToDevice, stream));
    cuda_check_pattern<<<(unsigned) grid_for(kRegionBytes / 4), 256, 0, stream>>>(dev_buf, kRegionBytes / 4, dev_sum,
                                                                                 dev_bad);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (read_counters(sum, bad)) return 1;
    const uint64_t host_bad_b = host_mismatches(b_region, kRegionBytes);
    std::printf("  HIP writes, CUDA reads  (B) : %s  CUDA checksum %s  CUDA mismatch %llu\n",
                r2 == 0 ? "answered" : "TIMEOUT",
                ctl->child_checksum_b == sum ? "matches the HIP checksum" : "differs", (unsigned long long) bad);
    std::printf("      host-side re-check of region B : %s (%llu bad words)\n",
                host_bad_b == 0 ? "byte-exact" : "CORRUPT", (unsigned long long) host_bad_b);

    if (ctl->parent_zero_copy && ctl->child_zero_copy) {
        uint32_t* dev_a = reinterpret_cast<uint32_t*> (reinterpret_cast<char*> (dev_alias) + ctl->a_off);
        uint32_t* dev_b = reinterpret_cast<uint32_t*> (reinterpret_cast<char*> (dev_alias) + ctl->b_off);
        CUDA_CHECK(cudaMemset(dev_sum, 0, 2 * sizeof(unsigned long long)));
        cuda_fill_pattern<<<(unsigned) grid_for(kRegionBytes / 4), 256, 0, stream>>>(dev_a, kRegionBytes / 4);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        const int r3 = send_command(ctl, SPIKE_ZERO_COPY, spin_us, false, 60000);
        CUDA_CHECK(cudaMemset(dev_sum, 0, 2 * sizeof(unsigned long long)));
        cuda_check_pattern<<<(unsigned) grid_for(kRegionBytes / 4), 256, 0, stream>>>(dev_b, kRegionBytes / 4, dev_sum,
                                                                                     dev_bad);
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (read_counters(sum, bad)) return 1;
        const uint64_t host_bad_c = host_mismatches(b_region, kRegionBytes);
        std::printf("  zero-copy kernels, no memcpy      : %s  HIP read A checksum %s  HIP wrote B, CUDA read %s"
                    "  host re-check %s\n",
                    r3 == 0 ? "answered" : "TIMEOUT",
                    ctl->child_checksum_zero == pattern_sum ? "matches" : "differs",
                    sum == pattern_sum ? "matches" : "differs", host_bad_c == 0 ? "byte-exact" : "CORRUPT");
    } else {
        std::printf("  zero-copy kernels, no memcpy      : skipped, no alias on both sides\n");
    }

    // ---- TEST 3  bandwidth
    std::printf("\nTEST 3  bandwidth (50 reps each, median and p99)\n");
    const uint64_t sizes[3] = {160 * 1024ull, 800 * 1024ull, 64ull * kMiB};
    void* priv = nullptr;
    const bool have_priv = cudaHostAlloc(&priv, kRegionBytes, cudaHostAllocPortable) == cudaSuccess;
    if (!have_priv) std::printf("  cudaHostAlloc failed: the private-buffer rows will be missing\n");
    // Ask the child to measure its own card first, and wait: the two bandwidth runs must not overlap.
    const int rbw = send_command(ctl, SPIKE_BANDWIDTH, spin_us, false, 120000);
    if (rbw != 0) std::printf("  the child did not answer the bandwidth command\n");
    for (int i = 0; i < 3; ++i) {
        const uint64_t bytes = sizes[i];
        std::vector<uint64_t> up, down, pup, pdown;
        for (int r = 0; r < 50; ++r) {
            uint64_t t0 = spike_now_ns();
            CUDA_CHECK(cudaMemcpyAsync(dev_buf, a_region, bytes, cudaMemcpyHostToDevice, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            up.push_back((spike_now_ns() - t0) / 1000);
            t0 = spike_now_ns();
            CUDA_CHECK(cudaMemcpyAsync(a_region, dev_buf, bytes, cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            down.push_back((spike_now_ns() - t0) / 1000);
            if (have_priv) {
                t0 = spike_now_ns();
                CUDA_CHECK(cudaMemcpyAsync(dev_buf, priv, bytes, cudaMemcpyHostToDevice, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                pup.push_back((spike_now_ns() - t0) / 1000);
                t0 = spike_now_ns();
                CUDA_CHECK(cudaMemcpyAsync(priv, dev_buf, bytes, cudaMemcpyDeviceToHost, stream));
                CUDA_CHECK(cudaStreamSynchronize(stream));
                pdown.push_back((spike_now_ns() - t0) / 1000);
            }
        }
        Sample s_shared_up, s_shared_down, s_priv_up, s_priv_down;
        summarize(up, bytes, s_shared_up);
        summarize(down, bytes, s_shared_down);
        summarize(pup, bytes, s_priv_up);
        summarize(pdown, bytes, s_priv_down);
        std::printf("  %s (this process)\n", prop.name);
        print_sample("shared mapping H2D", s_shared_up);
        print_sample("shared mapping D2H", s_shared_down);
        if (have_priv) {
            print_sample("private pinned H2D", s_priv_up);
            print_sample("private pinned D2H", s_priv_down);
        }
        if (rep->sizes_n == 3 && rep->magic == kMagic) {
            std::printf("  %s (the child process)\n", rep->device_name[0] ? rep->device_name : "the HIP device");
            print_sample("shared mapping H2D", rep->h2d_shared[i]);
            print_sample("shared mapping D2H", rep->d2h_shared[i]);
            print_sample("private pinned H2D", rep->h2d_private[i]);
            print_sample("private pinned D2H", rep->d2h_private[i]);
        }
    }
    std::printf("  verdict : %s\n",
                rep->magic == kMagic ? "shared pinned mapping is usable for bulk transfer on both cards"
                                     : "the child never reported - fill the table with the parent's numbers alone");

    // ---- TEST 4  round-trip latency
    std::printf("\nTEST 4  round-trip latency on the two doorbell words (%u pings per variant)\n", rounds);
    // Nanoseconds, not microseconds: a spinning ping on one machine is well under 1 us, which whole microseconds
    // round to 0.
    uint64_t ping_p50_ns[3] = {0, 0, 0};
    for (uint32_t v = 0; v < 3; ++v) {
        ctl->variant.store(v);
        const bool parent_futex = v == SPIKE_BOTH_FUTEX;
        std::vector<uint64_t> ns;
        ns.reserve(rounds);
        uint32_t timeouts = 0;
        for (uint32_t r = 0; r < rounds; ++r) {
            const uint64_t t0 = spike_now_ns();
            const uint32_t seq = ctl->seq.load() + 1;
            ctl->cmd.store(SPIKE_PING);
            ctl->seq.store(seq);
            spike_wake(ctl->seq);
            const uint64_t deadline = spike_now_ns() + 5000000ull;
            const uint32_t got =
                spike_wait(ctl->done, spin_us, parent_futex, deadline, [&](uint32_t x) { return x == seq; });
            if (got != seq) { ++timeouts; continue; }
            ns.push_back(spike_now_ns() - t0);
        }
        std::sort(ns.begin(), ns.end());
        ping_p50_ns[v] = pct(ns, 0.5);
        const char* names[3] = {"both spin         ", "child futex, spin ", "both futex        "};
        std::printf("  %s p50 %8.2f us  p99 %8.2f us  p999 %8.2f us  timeouts %u\n", names[v],
                    (double) pct(ns, 0.5) / 1000.0, (double) pct(ns, 0.99) / 1000.0, (double) pct(ns, 0.999) / 1000.0,
                    timeouts);
    }
    std::printf("  the child's own view: %llu answered pings, mean wait %llu us (spin) / %llu us (child futex) /"
                " %llu us (both futex)\n",
                (unsigned long long) rep->ping_rounds,
                rep->ping_rounds ? (unsigned long long) (rep->ping_wait_sum_us[0] / rep->ping_rounds) : 0ull,
                rep->ping_rounds ? (unsigned long long) (rep->ping_wait_sum_us[1] / rep->ping_rounds) : 0ull,
                rep->ping_rounds ? (unsigned long long) (rep->ping_wait_sum_us[2] / rep->ping_rounds) : 0ull);
    std::printf("  target                : p50 < 150 us on the spin variant (Gate 1)\n");
    std::printf("  verdict               : %s\n", ping_p50_ns[0] > 0 && ping_p50_ns[0] < 150000 ? "PASS" : "FAIL");

    // ---- TEST 5  round trip with work
    std::printf("\nTEST 5  round trip with work (%u rounds: 80 KiB in, a kernel on the AMD card, 200 KiB back)\n",
                work_rounds);
    cuda_fill_pattern<<<(unsigned) grid_for(kWorkInBytes / 4), 256, 0, stream>>>(dev_buf, kWorkInBytes / 4);
    CUDA_CHECK(cudaMemcpyAsync(in_region, dev_buf, kWorkInBytes, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<uint64_t> work_us;
    uint32_t work_bad = 0, work_timeouts = 0;
    for (uint32_t r = 0; r < work_rounds; ++r) {
        const uint64_t t0 = spike_now_ns();
        const int rc = send_command(ctl, SPIKE_WORK, spin_us, false, 5000);
        if (rc != 0) { ++work_timeouts; continue; }
        work_us.push_back((spike_now_ns() - t0) / 1000);
        const uint32_t* out = (const uint32_t*) out_region;
        uint64_t sum_out = 0;
        for (uint64_t i = 0; i < kWorkOutBytes / 4; ++i) sum_out += out[i];
        if (sum_out != ctl->child_work_sum) ++work_bad;
    }
    Sample work;
    summarize(work_us, kWorkInBytes + kWorkOutBytes, work);
    print_sample("full round trip", work);
    std::printf("  wrong results %u, timeouts %u\n", work_bad, work_timeouts);

    // The same work with no DMA (SPIKE_WORK_ZERO): the AMD kernel reads and writes the mapping over PCIe directly.
    // Not the plan's design; measured to see whether this link can meet the target at all.
    std::vector<uint64_t> zero_us;
    uint32_t zero_bad = 0, zero_timeouts = 0;
    for (uint32_t r = 0; r < work_rounds; ++r) {
        std::memset(out_region, 0, kWorkOutBytes);
        ctl->child_work_sum = 0;
        const uint64_t t0 = spike_now_ns();
        const int rc = send_command(ctl, SPIKE_WORK_ZERO, spin_us, false, 5000);
        if (rc != 0) { ++zero_timeouts; continue; }
        zero_us.push_back((spike_now_ns() - t0) / 1000);
        const uint32_t* out = (const uint32_t*) out_region;
        uint64_t sum_out = 0;
        for (uint64_t i = 0; i < kWorkOutBytes / 4; ++i) sum_out += out[i];
        if (sum_out != ctl->child_work_sum || sum_out == 0) ++zero_bad;
    }
    Sample zero;
    summarize(zero_us, kWorkInBytes + kWorkOutBytes, zero);
    print_sample("zero-copy round trip", zero);
    std::printf("  wrong results %u, timeouts %u (zero-copy)\n", zero_bad, zero_timeouts);
    if (rep->work_rounds && rep->work_stage_ns[0]) {   // the child timed the stages (SPIKE_STAGES=1)
        const double n = (double) rep->work_rounds * 1000.0;
        std::printf("  child's stages (mean us): memset + H2D %.1f, kernel %.1f, D2H %.1f, checksum read-back %.1f;"
                    " whole round on the child %.1f\n",
                    rep->work_stage_ns[0] / n, rep->work_stage_ns[1] / n, rep->work_stage_ns[2] / n,
                    rep->work_stage_ns[3] / n, rep->work_stage_ns[4] / n);
    }
    std::printf("  target                : p50 < 150 us, p99 < 500 us (Gate 1)\n");
    std::printf("  verdict               : %s\n",
                work.p50_us > 0 && work.p50_us < 150 && work.p99_us < 500 ? "PASS" : "FAIL");

    // ---- wind it down
    send_command(ctl, SPIKE_QUIT, spin_us, false, 5000);
    ctl->quit.store(1);
    spike_wake(ctl->seq);
    int status = 0;
    waitpid(kid, &status, 0);
    if (ctl->parent_register == SPIKE_REG_OK) cudaHostUnregister(base);
    if (have_priv) cudaFreeHost(priv);
    cudaStreamDestroy(stream);
    cudaFree(dev_buf);
    cudaFree(dev_sum);
    std::printf("\nchild exit status %d%s\n", WEXITSTATUS(status),
                ctl->child_error[0] ? (", child error: " + std::string(ctl->child_error)).c_str() : "");
    if (WIFSIGNALED(status)) std::printf("child was killed by signal %d\n", WTERMSIG(status));
    return 0;
}
