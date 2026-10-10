// tools/xhelper_spike/spike_layout.hpp - what the two spike programs agree on, and nothing else.
//
// docs/plans/OPTION_A_EXCLUSIVE_RAM.md Phase 1: prove the transport (dual registration, integrity, bandwidth,
// round-trip latency, round trip with work) between a CUDA parent and a HIP child sharing one memfd mapping, before
// any engine code is touched.  This header is plain C++20 with no CUDA and no HIP include, which is the same rule
// Phase 2 writes into `include/strata/xhelper/protocol.hpp`: the part both binaries share must compile in both builds.
//
// Everything is one mapping of 256 MiB: the control block, two 64 MiB integrity/bandwidth regions, the small
// work-in/work-out regions test 5 uses, and a report block the child fills in for the parent to print.
//
// The child is a worker here: the parent writes a command, rings `seq`, and waits for `done`.  `seq` and `done` are
// the two futex words Phase 1 exists to measure, and they are woken with FUTEX_WAKE without FUTEX_PRIVATE_FLAG -
// a private futex is keyed on one process's address space and would wake nobody in the other.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include <ctime>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#define STRATA_SPIKE_FUTEX 1
#else
#define STRATA_SPIKE_FUTEX 0
#include <unistd.h>
#endif

namespace xhelper_spike {

inline constexpr uint32_t kMagic = 0x50534858u;    // 'XHSP' read little-endian
inline constexpr uint32_t kVersion = 1;
inline constexpr uint64_t kMiB = 1048576ull;
inline constexpr uint64_t kTotalBytes = 256ull * kMiB;

inline constexpr uint64_t kCtlBytes = 4096ull;
inline constexpr uint64_t kRegionBytes = 64ull * kMiB;      // region A (CUDA writes) and region B (HIP writes)
inline constexpr uint64_t kWorkInBytes = 80ull * 1024ull;   // test 5: what the parent writes per round
inline constexpr uint64_t kWorkOutBytes = 200ull * 1024ull; // test 5: what the child writes back per round
inline constexpr uint64_t kReportBytes = 4096ull;

// The pattern every integrity and work test uses.  Integer on purpose: a checksum of these values is exact, so a
// PASS/FAIL here never depends on a tolerance.  __host__ __device__ so the same function fills a buffer on the host
// and checks it in a kernel on either card.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define SPIKE_HOSTDEV __host__ __device__ inline
#else
#define SPIKE_HOSTDEV inline
#endif

SPIKE_HOSTDEV uint32_t spike_pattern(uint64_t i) { return (uint32_t) (i * 2654435761ull) ^ (uint32_t) (i >> 3); }

enum Cmd : uint32_t {
    SPIKE_NONE = 0,
    SPIKE_CHECK_A = 1,     // copy region A host->device on the AMD side and checksum it
    SPIKE_CHECK_B = 2,     // fill region B from AMD device memory, for the CUDA side to read back
    SPIKE_ZERO_COPY = 3,   // read region A and write region B through the mapped device pointers, no memcpy
    SPIKE_BANDWIDTH = 4,   // the child measures its own H2D/D2H, shared mapping and private pinned buffer
    SPIKE_PING = 5,        // answer a ping; ctl::variant says how the child waits
    SPIKE_WORK = 6,        // test 5: 80 KiB in, a kernel, 200 KiB back
    SPIKE_STATS = 7,
    SPIKE_QUIT = 8
};

enum State : uint32_t { SPIKE_STARTING = 0, SPIKE_READY = 1, SPIKE_FAILED = 2, SPIKE_EXITED = 3 };

// The three latency variants test 4 runs, plus one diagnostic: with no spin budget on either side both processes go
// straight into FUTEX_WAIT, so the row measures the cross-process wake itself and nothing else.
enum Variant : uint32_t { SPIKE_BOTH_SPIN = 0, SPIKE_CHILD_FUTEX = 1, SPIKE_BOTH_FUTEX = 2, SPIKE_BOTH_SLEEP = 3 };

enum Register : uint32_t { SPIKE_REG_NOT_TRIED = 0, SPIKE_REG_OK = 1, SPIKE_REG_FAILED = 2 };

struct alignas(64) Ctl {
    uint32_t magic;
    uint32_t version;
    uint64_t total_bytes;
    uint64_t a_off, a_bytes;
    uint64_t b_off, b_bytes;
    uint64_t in_off, in_bytes;
    uint64_t out_off, out_bytes;
    uint64_t report_off;

    std::atomic<uint32_t> cmd;      // parent -> child
    std::atomic<uint32_t> seq;      // parent -> child: bumped after the payload is written (the futex word)
    std::atomic<uint32_t> done;     // child -> parent: set to the seq it answered (the other futex word)
    std::atomic<uint32_t> state;    // child -> parent: SPIKE_READY once its setup is done
    std::atomic<uint32_t> variant;  // how the child waits during SPIKE_PING
    std::atomic<uint32_t> quit;     // parent -> child: leave after the command it is given

    uint32_t rounds;
    uint32_t spin_us;
    uint32_t parent_register;   // Register: cudaHostRegister on the shared mapping
    uint32_t child_register;    // Register: hipHostRegister on the same pages
    int32_t parent_register_err;
    int32_t child_register_err;
    uint32_t parent_zero_copy;  // 1 when cudaHostGetDevicePointer gave a usable alias
    uint32_t child_zero_copy;
    uint64_t parent_checksum;   // what the parent computed, for the child to be compared with
    uint64_t parent_mismatch;   // elements that were not spike_pattern(i), read on the CUDA side
    uint64_t child_checksum_a;
    uint64_t child_checksum_b;
    uint64_t child_checksum_zero;
    uint64_t child_mismatch_a;
    uint64_t child_mismatch_b;
    uint64_t child_mismatch_zero;
    uint64_t child_work_sum;
    char child_error[256];
};
static_assert(sizeof(Ctl) <= kCtlBytes, "the control block must fit its region");
// Both sides use these words as futex words, so they have to be a plain 4-byte lock-free atomic: no mutex inside.
static_assert(std::atomic<uint32_t>::is_always_lock_free, "the two words must be futex-able, not a mutex");
static_assert(sizeof(std::atomic<uint32_t>) == 4, "a futex word is 4 bytes");

struct Sample {
    uint64_t bytes;
    uint64_t p50_us;
    uint64_t p99_us;
    double gib_s;
};

struct alignas(64) Report {
    uint32_t magic;
    uint32_t sizes_n;
    uint64_t size[4];
    Sample h2d_shared[4];      // the AMD card: shared mapping -> its VRAM
    Sample d2h_shared[4];      // its VRAM -> the shared mapping
    Sample h2d_private[4];     // the same from a hipHostMalloc buffer, the comparison Phase 1 asks for
    Sample d2h_private[4];
    uint64_t ping_rounds;         // pings the child answered, all variants together
    uint64_t ping_wait_sum_us[4]; // what the wait looked like from the child, summed per variant
    uint64_t work_rounds;
    uint64_t work_p50_us;
    uint64_t work_p99_us;
    char device_name[64];
};
static_assert(sizeof(Report) <= kReportBytes, "the report block must fit its region");

constexpr uint64_t ctl_off() { return 0; }
constexpr uint64_t a_off() { return kCtlBytes; }
constexpr uint64_t b_off() { return a_off() + kRegionBytes; }
constexpr uint64_t in_off() { return b_off() + kRegionBytes; }
constexpr uint64_t out_off() { return in_off() + kWorkInBytes; }
constexpr uint64_t report_off() { return out_off() + kWorkOutBytes; }
static_assert(report_off() + kReportBytes <= kTotalBytes, "the layout must fit the mapping");

// ------------------------------------------------------------------ the two waits
//
// Phase 1's point is these four lines: a spin, and a futex that a *different process* can wake.

inline uint64_t spike_now_ns() {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

inline void spike_pause() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    asm volatile("" ::: "memory");
#endif
}

/// Spin until `word` satisfies pred, up to `spin_us`; then, if `use_futex`, sleep on it.  Returns the value that
/// ended the wait, or 0 on timeout.
template <class Pred>
inline uint32_t spike_wait(const volatile std::atomic<uint32_t>& word, uint32_t spin_us, bool use_futex,
                           uint64_t deadline_ns, Pred pred) {
    const volatile uint32_t* p = reinterpret_cast<const volatile uint32_t*>(&word);
    for (;;) {
        uint32_t v = p[0];
        if (pred(v)) return v;
        const uint64_t spin_until = spike_now_ns() + 1000ull * spin_us;
        while (spike_now_ns() < spin_until) {
            spike_pause();
            v = p[0];
            if (pred(v)) return v;
        }
        if (spike_now_ns() > deadline_ns) return 0;
        if (!use_futex) continue;
#if STRATA_SPIKE_FUTEX
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 2000000;   // 2 ms, then look again: a wake that lands between the read and the wait costs 2 ms
        syscall(SYS_futex, (void*) p, FUTEX_WAIT, v, &ts, nullptr, 0);
#else
        usleep(2000);
#endif
    }
}

inline void spike_wake(const volatile std::atomic<uint32_t>& word) {
    const volatile uint32_t* p = reinterpret_cast<const volatile uint32_t*>(&word);
#if STRATA_SPIKE_FUTEX
    syscall(SYS_futex, (void*) p, FUTEX_WAKE, 0x7fffffff, nullptr, nullptr, 0);
#endif
}

}  // namespace xhelper_spike
