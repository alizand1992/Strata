// include/strata/xhelper/protocol.hpp - the only thing the CUDA engine and the HIP expert helper both understand.
//
// docs/plans/OPTION_A_EXCLUSIVE_RAM.md Phase 2.  Strata compiles the same sources for CUDA or HIP and cannot link
// two copies of either runtime (CMakeLists.txt refuses STRATA_ENABLE_CUDA with STRATA_ENABLE_HIP), so a CUDA primary
// and a HIP helper can only be two processes.  This header is the shared memory they talk through: one mapping, made
// by the engine with `memfd_create`, inherited by the helper through `fork` + `execv`, with the engine's assignment,
// one request in flight, one response, and two futex words so neither side has to poll.
//
// Rules, from the plan and enforced below:
//   - Plain C++20.  No CUDA include, no HIP include.  It has to compile in both builds and in a CPU-only test.
//   - The writer fills its payload, then `seq.store(n, release)`; the reader checks `seq.load(acquire)`.  A payload
//     field is never read while the seq that published it has not been seen.
//   - `FUTEX_WAKE` is skipped unless the other side's `*_sleeping` flag says it is parked in `FUTEX_WAIT`.  Both sides
//     spin for `spin_us` (200 by default) before parking, because a hand-off through the futex costs about 26 us on a
//     quiet CPU and the engine's per-layer round trip has to stay far under that (tools/xhelper_spike measured
//     0.25 us p50 for a spin-only hand-off, 26 us p50 when both sides sleep first).
//   - `FUTEX_WAIT`/`FUTEX_WAKE`, never the `_PRIVATE` variants: a private futex is keyed on one process's address
//     space and would wake nobody in the other.
//   - Exactly one request in flight.  The engine serialises its pool workers on it (Phase 4), the helper never has to
//     think about concurrency.
//
// The geometry (`kH`, `kMaxTok`, `kK`) has to match the kernels the helper runs.  It is asserted against
// `strata::kernels::cpu::H` and `MAXT` in the engine-side translation units that include this header
// (`tests/xhelper/protocol_test.cpp`, `src/core/external_experts.cpp`), not here: this file must also compile on its
// own, in the helper, which does not link the CPU pool.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <ctime>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#include <unistd.h>
#define STRATA_XHELPER_FUTEX 1
#else
#define STRATA_XHELPER_FUTEX 0
#include <unistd.h>
#endif

namespace strata::xhelper {

inline constexpr uint64_t kMagic = 0x31504C484B525453ull;   // "STRKHLP1" read little-endian
inline constexpr uint32_t kVersion = 2;   // 2: the prompt kind and the bulk region (Phase 7.1)

inline constexpr int kH = 2560;              // n_embd: strata::kernels::cpu::H
inline constexpr int kMaxTok = 8;            // strata::kernels::cpu::MAXT
inline constexpr int kK = 10;                // experts per token
inline constexpr int kCap = kMaxTok * kK;    // the remote tier's CAP: rows one request can carry

// Tuning the plan settles on: the helper spins this long before parking, the engine waits this long for a response,
// the helper refreshes its heartbeat this often while idle, and a heartbeat this old means the helper is gone.
inline constexpr uint32_t kDefaultSpinUs = 200;
inline constexpr uint32_t kDefaultTimeoutMs = 2000;
inline constexpr uint64_t kHeartbeatIdleNs = 100ull * 1000000ull;
inline constexpr uint64_t kHelperDeadAfterNs = 5ull * 1000000000ull;

// One mapping holds the header, both pair arrays, the request and the response.  For a 48 x 512 model that is about
// 1.3 MiB; 4 MiB is what both sides allocate, with room for a bigger pair table.  With the prompt kind on, the bulk
// region (bulk_layout below) follows at kMappingBytes, so the mapping is kMappingBytes + bulk_bytes.
inline constexpr uint64_t kMiB = 1048576ull;
inline constexpr uint64_t kMappingBytes = 4ull * kMiB;
inline constexpr uint64_t kCacheLine = 64;
inline constexpr uint64_t kPage = 4096;

// Phase 7.1: a prompt layer's request.  The rows the helper holds are computed for up to bulk_max_tok tokens at a time
// and come back as one weighted sum per token (n_tok x kH floats), not one row per routed pair: K times less over the
// link, and the engine adds the sums in its combine.  A chunk longer than bulk_max_tok goes as several requests.
enum : int32_t { kKindRows = 0, kKindPrompt = 1 };
inline constexpr uint32_t kMaxPromptTok = 16384;

enum class EngineState : uint32_t { Init = 0, Assigned = 1, Running = 2, Shutdown = 3 };
enum class HelperState : uint32_t { Starting = 0, Loading = 1, Ready = 2, Failed = 3, Exited = 4 };
enum : int32_t { kStatusOk = 0, kStatusFailed = 1, kStatusDropped = 2 };

struct alignas(64) Header {
    uint64_t magic, total_bytes;
    uint32_t version, header_bytes;
    uint32_t n_layers, n_expert, h, cap, max_tok;
    std::atomic<uint32_t> engine_state, helper_state;
    std::atomic<uint64_t> helper_heartbeat_ns;   // CLOCK_MONOTONIC; the helper bumps it every 100 ms while idle
    int32_t helper_pid;
    uint64_t vram_budget_bytes;                  // 0 = the helper decides: free VRAM - 1 GiB
    uint32_t n_assigned, n_loaded;               // pairs offered by the engine / actually loaded by the helper
    uint64_t assigned_off, loaded_off;
    uint64_t request_off, response_off;
    char model_pack[1024];                       // the engine's --pack path
    char native_preset[1024];                    // the engine's --native argument, "" if none
    char error[512];                             // the helper's last fatal error, NUL-terminated
    // Phase 7.1, the prompt kind: 0 bytes = off.  The engine sets them; the helper reserves its prompt buffers for
    // bulk_max_tok tokens before it loads experts, and marks the layers whose prompt products it can run.
    uint64_t bulk_off, bulk_bytes;
    uint32_t bulk_max_tok, prompt_ok;            // prompt_ok: the helper set prompt_layers (0 = the prompt kind is off)
    uint8_t prompt_layers[128];                  // [layer]: 1 = the helper runs this layer's prompt products (kMaxLayers)
};

struct Pair {
    int32_t layer, expert;
};

struct alignas(64) Request {   // engine -> helper; exactly one in flight
    std::atomic<uint32_t> seq;               // futex word; the engine increments it after writing the payload
    std::atomic<uint32_t> helper_sleeping;   // the helper sets 1 before FUTEX_WAIT, 0 after
    int32_t layer, n_tok, n_entries;
    // kKindPrompt: the payload is in the bulk region; n_tok tokens, n_entries experts, n_rows routed rows (below)
    int32_t kind, n_rows;
    int32_t entry[kCap];                     // index i into the n_tok * k routed list
    int32_t expert[kCap];                    // expert id of entry[j]
    alignas(64) float x[kMaxTok * kH];       // the activations of the n_tok tokens, as the CPU pool sees x_f
};

struct alignas(64) Response {   // helper -> engine
    std::atomic<uint32_t> seq;               // futex word; the helper sets it = Request::seq when done
    std::atomic<uint32_t> engine_sleeping;
    int32_t status;                          // kStatusOk, else see Header::error
    int32_t n_rows;                          // == Request::n_entries on success (kKindPrompt: == Request::n_rows)
    int32_t entry[kCap];                     // row r belongs to routed entry entry[r]
    alignas(64) float rows[kCap * kH];
};

static_assert(sizeof(Pair) == 8, "the pair table is two int32 per expert");
static_assert(kCap == kMaxTok * kK, "the request and response carry kMaxTok x kK rows");
static_assert(std::atomic<uint32_t>::is_always_lock_free && sizeof(std::atomic<uint32_t>) == 4,
              "seq is a futex word: 4 bytes, no lock inside");
static_assert(std::atomic<uint64_t>::is_always_lock_free && sizeof(std::atomic<uint64_t>) == 8,
              "the heartbeat is read without a lock");
static_assert(alignof(Header) == kCacheLine && alignof(Request) == kCacheLine && alignof(Response) == kCacheLine,
              "every region starts on a cache line");
static_assert(sizeof(Header) % kCacheLine == 0, "the header is a whole number of cache lines");
static_assert(sizeof(Request) % kCacheLine == 0 && sizeof(Response) % kCacheLine == 0,
              "so is the request and the response");

inline constexpr uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

// The one place the offsets are computed.  Both sides call it with the same geometry and get the same answer, which
// is what makes `header_ok` below able to catch a mismatched pair of binaries.
struct Layout {
    uint64_t header_bytes;
    uint64_t pairs_capacity;   // pairs each of the two tables can hold
    uint64_t assigned_off, loaded_off, request_off, response_off, total_bytes;
};

constexpr Layout layout_of(uint32_t n_layers, uint32_t n_expert) {
    Layout l{};
    l.header_bytes = align_up(sizeof(Header), kCacheLine);
    l.pairs_capacity = (uint64_t) n_layers * (uint64_t) n_expert;
    const uint64_t pair_bytes = align_up(l.pairs_capacity * sizeof(Pair), kCacheLine);
    l.assigned_off = l.header_bytes;
    l.loaded_off = l.assigned_off + pair_bytes;
    l.request_off = l.loaded_off + pair_bytes;
    l.response_off = l.request_off + align_up(sizeof(Request), kCacheLine);
    l.total_bytes = align_up(l.response_off + sizeof(Response), kCacheLine);
    return l;
}

// The bulk region, at Header::bulk_off: every array page-aligned (each side registers it with its GPU runtime).
//   x      [max_tok * kH]  float  the request's tokens' MoE input (the prompt path's `mixed` rows)
//   sum    [max_tok * kH]  float  the answer: per token, sum over its helper rows of weight * row
//   w      [max_tok * kK]  float  the routing weights of the request's tokens, entry t * kK + k
//   pair   [max_tok * kK]  int32  per helper row, its routed entry t * kK + k (t relative to the request), the
//                                 rows of experts[0] first, then experts[1], ... (each expert's rows in token order)
//   expert [n_expert]      int32  the request's experts, ascending
//   count  [n_expert]      int32  their row counts
struct BulkLayout {
    uint64_t x_off, sum_off, w_off, pair_off, expert_off, count_off, total_bytes;
};

constexpr BulkLayout bulk_layout(uint32_t max_tok, uint32_t n_expert) {
    BulkLayout b{};
    const uint64_t rows = (uint64_t) max_tok * (uint64_t) kH * sizeof(float);
    const uint64_t entries = (uint64_t) max_tok * (uint64_t) kK * 4ull;
    b.x_off = 0;
    b.sum_off = align_up(b.x_off + rows, kPage);
    b.w_off = align_up(b.sum_off + rows, kPage);
    b.pair_off = align_up(b.w_off + entries, kPage);
    b.expert_off = align_up(b.pair_off + entries, kPage);
    b.count_off = align_up(b.expert_off + (uint64_t) n_expert * 4ull, kPage);
    b.total_bytes = align_up(b.count_off + (uint64_t) n_expert * 4ull, kPage);
    return b;
}

// A 48 x 512 model is the biggest one the plan names (24576 pairs, 48 x 512 = 24576 experts).
constexpr uint32_t kMaxLayers = 128;
constexpr uint32_t kMaxExpert = 1024;
static_assert(sizeof(Header::prompt_layers) == kMaxLayers, "one prompt flag per layer the protocol allows");
static_assert(layout_of(48, 512).total_bytes <= kMappingBytes, "the plan's model fits the plan's 4 MiB mapping");
static_assert(layout_of(48, 512).total_bytes / kMiB == 1, "about 1.3 MiB, as the plan says");

inline bool geometry_ok(uint32_t n_layers, uint32_t n_expert) {
    return n_layers > 0 && n_expert > 0 && n_layers <= kMaxLayers && n_expert <= kMaxExpert;
}

// The helper's first act after mmap: prove the engine it woke inside agrees about what the mapping is.  Everything
// after this point trusts the header.
inline bool header_ok(const Header& h, uint64_t mapped_bytes) {
    if (h.magic != kMagic || h.version != kVersion) return false;
    if (h.total_bytes > mapped_bytes) return false;
    if (h.h != (uint32_t) kH || h.cap != (uint32_t) kCap || h.max_tok != (uint32_t) kMaxTok) return false;
    if (!geometry_ok(h.n_layers, h.n_expert)) return false;
    const Layout l = layout_of(h.n_layers, h.n_expert);
    if (h.header_bytes != (uint32_t) l.header_bytes) return false;
    if (h.assigned_off != l.assigned_off || h.loaded_off != l.loaded_off) return false;
    if (h.request_off != l.request_off || h.response_off != l.response_off) return false;
    if (h.total_bytes < l.total_bytes) return false;
    if (h.n_assigned > l.pairs_capacity || h.n_loaded > l.pairs_capacity) return false;
    if (h.bulk_bytes != 0) {   // the prompt kind: the bulk region right after the fixed part, the size its layout says
        if (h.bulk_off != kMappingBytes || h.bulk_max_tok == 0 || h.bulk_max_tok > kMaxPromptTok) return false;
        if (h.bulk_bytes != bulk_layout(h.bulk_max_tok, h.n_expert).total_bytes) return false;
        if (h.bulk_off + h.bulk_bytes > mapped_bytes || h.bulk_off + h.bulk_bytes > h.total_bytes) return false;
    }
    return true;
}

// The engine's first act: describe the model and lay the mapping out.  The helper then checks it with header_ok.
inline bool init_header(Header& h, uint32_t n_layers, uint32_t n_expert, uint64_t mapped_bytes, const char* model_pack,
                        const char* native_preset) {
    if (!geometry_ok(n_layers, n_expert)) return false;
    const Layout l = layout_of(n_layers, n_expert);
    if (l.total_bytes > mapped_bytes) return false;
    // Field by field rather than assigning a Header{}: Header holds atomics, so it has no copy assignment.
    h.magic = kMagic;
    h.total_bytes = mapped_bytes;
    h.version = kVersion;
    h.header_bytes = (uint32_t) l.header_bytes;
    h.n_layers = n_layers;
    h.n_expert = n_expert;
    h.h = (uint32_t) kH;
    h.cap = (uint32_t) kCap;
    h.max_tok = (uint32_t) kMaxTok;
    h.engine_state.store((uint32_t) EngineState::Init, std::memory_order_relaxed);
    h.helper_state.store((uint32_t) HelperState::Starting, std::memory_order_relaxed);
    h.helper_heartbeat_ns.store(0, std::memory_order_relaxed);
    h.helper_pid = 0;
    h.vram_budget_bytes = 0;
    h.n_assigned = 0;
    h.n_loaded = 0;
    h.assigned_off = l.assigned_off;
    h.loaded_off = l.loaded_off;
    h.request_off = l.request_off;
    h.response_off = l.response_off;
    if (model_pack) {
        std::snprintf(h.model_pack, sizeof(h.model_pack), "%.1023s", model_pack);
        h.model_pack[sizeof(h.model_pack) - 1] = '\0';
    }
    if (native_preset) {
        std::snprintf(h.native_preset, sizeof(h.native_preset), "%.1023s", native_preset);
        h.native_preset[sizeof(h.native_preset) - 1] = '\0';
    }
    h.error[0] = '\0';
    h.bulk_off = 0;
    h.bulk_bytes = 0;
    h.bulk_max_tok = 0;
    h.prompt_ok = 0;
    for (auto& v : h.prompt_layers) v = 0;
    return true;
}

/// The engine, after init_header and before the helper starts: turn the prompt kind on for `max_tok` tokens per
/// request.  The mapping must be kMappingBytes + bulk_layout(max_tok, n_expert).total_bytes.
inline bool init_bulk(Header& h, uint32_t max_tok, uint64_t mapped_bytes) {
    if (max_tok == 0 || max_tok > kMaxPromptTok) return false;
    const BulkLayout b = bulk_layout(max_tok, h.n_expert);
    if (kMappingBytes + b.total_bytes > mapped_bytes) return false;
    h.bulk_off = kMappingBytes;
    h.bulk_bytes = b.total_bytes;
    h.bulk_max_tok = max_tok;
    h.total_bytes = mapped_bytes;
    return true;
}

inline Pair* pairs_at(void* base, uint64_t off) {
    return reinterpret_cast<Pair*> (reinterpret_cast<char*> (base) + off);
}
inline const Pair* pairs_at(const void* base, uint64_t off) {
    return reinterpret_cast<const Pair*> (reinterpret_cast<const char*> (base) + off);
}
inline Request* request_at(void* base, uint64_t off) {
    return reinterpret_cast<Request*> (reinterpret_cast<char*> (base) + off);
}
inline Response* response_at(void* base, uint64_t off) {
    return reinterpret_cast<Response*> (reinterpret_cast<char*> (base) + off);
}
template <class T>
inline T* bulk_at(void* base, const Header& h, uint64_t off) {
    return reinterpret_cast<T*> (reinterpret_cast<char*> (base) + h.bulk_off + off);
}

inline bool pair_ok(const Header& h, int32_t layer, int32_t expert) {
    return layer >= 0 && expert >= 0 && (uint32_t) layer < h.n_layers && (uint32_t) expert < h.n_expert;
}

// ------------------------------------------------------------------ time
inline uint64_t monotonic_ns() {
    struct timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t) ts.tv_sec * 1000000000ull + (uint64_t) ts.tv_nsec;
}

inline void cpu_pause() {
#if defined(__x86_64__) || defined(__i386__)
    __builtin_ia32_pause();
#else
    asm volatile("" ::: "memory");
#endif
}

// ------------------------------------------------------------------ the two words
//
// `word` is Request::seq, Response::seq, or a test's own word.  `mine` is this side's sleeping flag in the OTHER
// struct: the helper parks on Request::helper_sleeping, the engine parks on Response::engine_sleeping, and each side
// only calls FUTEX_WAKE when the other one's flag says it is actually parked.

/// Wait until pred(word) holds.  Spin `spin_us` first, then park in FUTEX_WAIT until `deadline_ns`.  `mine` may be
/// null (a test that has no partner to wake).  Returns true when pred became true.
template <class Pred>
inline bool wait_until(const volatile std::atomic<uint32_t>* word, std::atomic<uint32_t>* mine, uint32_t spin_us,
                       uint64_t deadline_ns, Pred pred) {
    const volatile uint32_t* p = reinterpret_cast<const volatile uint32_t*> (word);
    for (;;) {
        uint32_t v = p[0];
        if (pred(v)) return true;
        const uint64_t spin_until = monotonic_ns() + 1000ull * spin_us;
        while (monotonic_ns() < spin_until) {
            cpu_pause();
            v = p[0];
            if (pred(v)) return true;
        }
        if (monotonic_ns() > deadline_ns) return false;
        if (!mine) continue;
        mine->store(1, std::memory_order_release);
        v = p[0];
        if (pred(v)) {
            mine->store(0, std::memory_order_release);
            return true;
        }
#if STRATA_XHELPER_FUTEX
        struct timespec ts;
        ts.tv_sec = 0;
        ts.tv_nsec = 2000000;   // 2 ms ceiling: a wake lost between this load and the wait costs 2 ms, no more
        syscall(SYS_futex, (void*) p, FUTEX_WAIT, v, &ts, nullptr, 0);
#else
        usleep(2000);
#endif
        mine->store(0, std::memory_order_release);
    }
}

/// Wake the waiter on `word`, but only if its own flag says it is parked.  On the fast path the other side is
/// spinning and a syscall would be pure cost.
inline void wake_if_parked(const volatile std::atomic<uint32_t>* word, const std::atomic<uint32_t>* other_sleeping) {
    if (other_sleeping && other_sleeping->load(std::memory_order_acquire) == 0) return;
#if STRATA_XHELPER_FUTEX
    syscall(SYS_futex, (void*) word, FUTEX_WAKE, 0x7fffffff, nullptr, nullptr, 0);
#endif
}

inline void wake_always(const volatile std::atomic<uint32_t>* word) {
#if STRATA_XHELPER_FUTEX
    syscall(SYS_futex, (void*) word, FUTEX_WAKE, 0x7fffffff, nullptr, nullptr, 0);
#endif
}

// ------------------------------------------------------------------ the helper's pulse
//
// The engine cannot waitpid() a helper it did not fork in the current process, and a helper killed with SIGKILL
// leaves every flag exactly as it was.  The heartbeat is what makes that visible.
inline void bump_heartbeat(Header& h) { h.helper_heartbeat_ns.store(monotonic_ns(), std::memory_order_release); }

inline bool helper_alive(const Header& h, uint64_t now_ns) {
    const uint64_t beat = h.helper_heartbeat_ns.load(std::memory_order_acquire);
    if (beat == 0) return h.helper_state.load(std::memory_order_acquire) != (uint32_t) HelperState::Exited;
    if (now_ns < beat) return true;   // a clock reading before the heartbeat: not evidence of death
    return now_ns - beat <= kHelperDeadAfterNs;
}

}  // namespace strata::xhelper
