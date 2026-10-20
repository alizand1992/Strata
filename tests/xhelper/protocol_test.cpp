// tests/xhelper/protocol_test.cpp - the Option A Gate 2: the engine and the helper agree on one block of memory.
//
// CPU only, no CUDA, no HIP, no model, no device: it is the protocol header of docs/plans/OPTION_A_EXCLUSIVE_RAM.md
// Phase 2 checked on its own, in both the CUDA build and the HIP build, plus a two-thread ping-pong that uses the
// same publish/read/wake discipline the engine and the helper will use across a real memfd.  Same process here rather
// than two, which is the one thing a real hand-off adds; tools/xhelper_spike/spike_doorbell.cpp covers the two-process
// form without a card.
#include "strata/xhelper/protocol.hpp"

// The engine-side half of the plan's Phase 2 rule: the protocol's geometry must be the kernels' geometry.
#include "strata/kernels/cpu/expert.hpp"

static_assert(strata::xhelper::kH == strata::kernels::cpu::H, "Header/Request/Response rows are the CPU expert's H");
static_assert(strata::xhelper::kMaxTok == strata::kernels::cpu::MAXT, "one request carries the CPU pool's token window");
static_assert(strata::xhelper::kCap == strata::kernels::cpu::MAXT * 10, "kCap is the remote tier's CAP = MAXT * 10");

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <tuple>
#include <thread>
#include <vector>

using namespace strata::xhelper;

namespace {

int g_fails = 0;

void check(bool ok, const char* what) {
    std::printf("  %-58s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++g_fails;
}

alignas(64) unsigned char g_map[kMappingBytes];

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t i = (size_t) (p * (double) (v.size() - 1) + 0.5);
    return v[i];
}

// -------------------------------------------------------------------------------- mapping
void test_sizes() {
    std::printf("\nsizes, alignment and offsets\n");
    check(alignof(Header) == 64 && alignof(Request) == 64 && alignof(Response) == 64, "all three regions are 64-byte aligned");
    check(sizeof(Header) % 64 == 0, "the header ends on a cache line");
    check(sizeof(Request) % 64 == 0 && sizeof(Response) % 64 == 0, "so do the request and the response");
    check(sizeof(Pair) == 8, "a pair is two int32");
    check(offsetof(Request, x) % 64 == 0, "the activations start on a cache line");
    check(offsetof(Response, rows) % 64 == 0, "so do the result rows");
    check(sizeof(Request) == offsetof(Request, x) + (size_t) kMaxTok * kH * 4, "Request holds kMaxTok x H floats");
    check(sizeof(Response) == offsetof(Response, rows) + (size_t) kCap * kH * 4, "Response holds kCap x H floats");

    const Layout l = layout_of(48, 512);
    std::printf("  48 x 512: header %llu  assigned @ %llu  loaded @ %llu  request @ %llu  response @ %llu  total %llu\n",
                (unsigned long long) l.header_bytes, (unsigned long long) l.assigned_off,
                (unsigned long long) l.loaded_off, (unsigned long long) l.request_off,
                (unsigned long long) l.response_off, (unsigned long long) l.total_bytes);
    check(l.assigned_off == l.header_bytes, "the pair tables start after the header");
    check(l.loaded_off > l.assigned_off && l.request_off > l.loaded_off && l.response_off > l.request_off,
          "the regions follow each other with no overlap");
    check(l.request_off % 64 == 0 && l.response_off % 64 == 0, "each region starts on a cache line");
    check(l.pairs_capacity == 24576, "48 x 512 gives 24576 pairs");
    check(l.total_bytes <= kMappingBytes, "the plan's model fits the 4 MiB mapping the plan allocates");
    check(layout_of(1, 1).total_bytes == align_up(sizeof(Header), 64) + align_up(sizeof(Pair), 64) * 2 +
              align_up(sizeof(Request), 64) + sizeof(Response),
          "a one-expert model's layout is the sum of its parts");
    check(!geometry_ok(0, 512) && !geometry_ok(48, 0) && !geometry_ok(kMaxLayers + 1, 512), "geometry_ok rejects a model it cannot describe");
}

void test_header() {
    std::printf("\nwhat the helper will accept\n");
    std::memset(g_map, 0, sizeof(g_map));
    auto* h = reinterpret_cast<Header*> (g_map);
    check(init_header(*h, 48, 512, kMappingBytes, "/models/qwen38/native", ""), "init_header fills a mapping");
    check(header_ok(*h, kMappingBytes), "and the helper's check accepts it");
    check(std::string(h->model_pack) == "/models/qwen38/native", "the pack path crosses over intact");

    const uint64_t magic = h->magic, hh = h->h, cap = h->cap;
    const uint64_t req_off = h->request_off, n_assigned = h->n_assigned;
    const auto saved_loaded = h->n_loaded;

    h->magic = 0;
    check(!header_ok(*h, kMappingBytes), "a wrong magic is refused");
    h->magic = magic;
    h->version = kVersion + 1;
    check(!header_ok(*h, kMappingBytes), "a version the helper does not know is refused");
    h->version = kVersion;
    h->h = 2048;
    check(!header_ok(*h, kMappingBytes), "a model whose H is not the helper's H is refused");
    h->h = hh;
    h->cap = 40;
    check(!header_ok(*h, kMappingBytes), "a request wider than the helper's rows is refused");
    h->cap = cap;
    h->request_off += 64;
    check(!header_ok(*h, kMappingBytes), "offsets the helper cannot recompute are refused");
    h->request_off = req_off;
    h->total_bytes = kMappingBytes + 1;
    check(!header_ok(*h, kMappingBytes), "a mapping smaller than the header claims is refused");
    h->total_bytes = kMappingBytes;
    h->n_assigned = layout_of(48, 512).pairs_capacity + 1;
    check(!header_ok(*h, kMappingBytes), "an offer longer than the pair table is refused");
    h->n_assigned = (uint32_t) n_assigned;
    check(header_ok(*h, kMappingBytes), "and the untouched header still passes");

    check(pair_ok(*h, 47, 511), "the last pair of the model is in range");
    check(!pair_ok(*h, 48, 511) && !pair_ok(*h, 47, 512) && !pair_ok(*h, -1, 0), "one past the end, and a negative, are not");

    // The pair tables have to sit where the header says, at the size the header says.
    const Layout l = layout_of(48, 512);
    auto* assigned = pairs_at(g_map, h->assigned_off);
    auto* loaded = pairs_at(g_map, h->loaded_off);
    check(reinterpret_cast<char*> (loaded) - reinterpret_cast<char*> (assigned) ==
              (ptrdiff_t) align_up(l.pairs_capacity * sizeof(Pair), 64),
          "the loaded table starts where the assigned table ends");
    for (uint32_t i = 0; i < l.pairs_capacity; ++i) assigned[i] = Pair{ (int32_t) (i / 512), (int32_t) (i % 512) };
    bool pairs_ok = true;
    for (uint32_t i = 0; i < l.pairs_capacity; ++i) {
        const Pair p = assigned[i];
        if (!pair_ok(*h, p.layer, p.expert) || p.layer != (int32_t) (i / 512) || p.expert != (int32_t) (i % 512)) pairs_ok = false;
    }
    check(pairs_ok, "the whole 24576-pair table is writable, in range, and readable");
    (void) loaded;
    (void) saved_loaded;
}

// -------------------------------------------------------------------------------- the bulk region (Phase 7.1)
void test_bulk() {
    std::printf("\nthe prompt kind's bulk region\n");
    const uint32_t tok = 2048;
    const BulkLayout b = bulk_layout(tok, 512);
    const uint64_t mapped = kMappingBytes + b.total_bytes;
    std::vector<unsigned char> map(mapped + 64);
    auto* base = reinterpret_cast<unsigned char*> ((reinterpret_cast<uintptr_t> (map.data()) + 63) / 64 * 64);
    auto* h = reinterpret_cast<Header*> (base);
    check(init_header(*h, 48, 512, mapped, "/models/qwen38/native", ""), "init_header on a mapping with room for the bulk");
    check(h->bulk_bytes == 0 && h->prompt_ok == 0 && header_ok(*h, mapped), "the prompt kind starts off, and passes");
    check(!init_bulk(*h, tok, kMappingBytes + b.total_bytes - 1), "init_bulk refuses a mapping one byte short");
    check(!init_bulk(*h, 0, mapped) && !init_bulk(*h, kMaxPromptTok + 1, mapped), "and zero or too many tokens");
    check(init_bulk(*h, tok, mapped) && header_ok(*h, mapped), "init_bulk turns it on, and the helper's check accepts it");
    check(h->bulk_off == kMappingBytes && h->bulk_bytes == b.total_bytes, "the bulk region follows the fixed 4 MiB");
    check(!header_ok(*h, mapped - 1), "a mapping smaller than the bulk region is refused");
    h->bulk_bytes += kPage;
    check(!header_ok(*h, mapped + kPage), "bulk bytes the layout does not give are refused");
    h->bulk_bytes -= kPage;
    h->bulk_off += kPage;
    check(!header_ok(*h, mapped + kPage), "a bulk region somewhere else is refused");
    h->bulk_off -= kPage;
    h->bulk_max_tok = tok + 1;
    check(!header_ok(*h, mapped), "a token count the size was not made for is refused");
    h->bulk_max_tok = tok;
    check(header_ok(*h, mapped), "and the untouched header still passes");

    const uint64_t offs[] = {b.x_off, b.sum_off, b.w_off, b.pair_off, b.expert_off, b.count_off, b.total_bytes};
    bool aligned = true, ordered = true;
    for (size_t i = 0; i < 7; ++i) {
        aligned = aligned && offs[i] % kPage == 0;
        if (i > 0) ordered = ordered && offs[i] > offs[i - 1];
    }
    check(aligned && ordered, "every array page-aligned, in order, none empty");
    check(b.sum_off - b.x_off >= (uint64_t) tok * kH * 4 && b.w_off - b.sum_off >= (uint64_t) tok * kH * 4 &&
              b.pair_off - b.w_off >= (uint64_t) tok * kK * 4 && b.expert_off - b.pair_off >= (uint64_t) tok * kK * 4 &&
              b.count_off - b.expert_off >= 512 * 4 && b.total_bytes - b.count_off >= 512 * 4,
          "each array holds what the layout says it holds");
    check(b.total_bytes / kMiB == 40, "2048 tokens: about 40 MiB (x and the sums, 20 MiB each)");
    // the last element of each array is writable and lands inside the mapping
    bulk_at<float>(base, *h, b.sum_off)[(size_t) tok * kH - 1] = 1.0f;
    bulk_at<int32_t>(base, *h, b.count_off)[511] = 7;
    check(bulk_at<int32_t>(base, *h, b.count_off)[511] == 7 &&
              reinterpret_cast<unsigned char*> (bulk_at<int32_t>(base, *h, b.count_off) + 512) <= base + mapped,
          "the arrays' ends are inside the mapping");
}

void test_heartbeat() {
    std::printf("\nthe helper's pulse\n");
    std::memset(g_map, 0, sizeof(g_map));
    auto* h = reinterpret_cast<Header*> (g_map);
    init_header(*h, 48, 512, kMappingBytes, "", "");
    const uint64_t now = monotonic_ns();
    check(helper_alive(*h, now), "a helper that has not had to bump yet is taken as alive");
    h->helper_state.store((uint32_t) HelperState::Exited, std::memory_order_relaxed);
    check(!helper_alive(*h, now), "until it says it exited");
    h->helper_state.store((uint32_t) HelperState::Ready, std::memory_order_relaxed);
    bump_heartbeat(*h);
    check(helper_alive(*h, monotonic_ns()), "a fresh heartbeat is alive");
    h->helper_heartbeat_ns.store(now - kHelperDeadAfterNs - 1, std::memory_order_relaxed);
    check(!helper_alive(*h, now), "a heartbeat five seconds old is a dead helper");
    h->helper_heartbeat_ns.store(now - kHeartbeatIdleNs, std::memory_order_relaxed);
    check(helper_alive(*h, now), "one idle period old is not");
}

// -------------------------------------------------------------------------------- ping-pong
//
// One request in flight, exactly as Phase 4 will use it: the engine fills the payload, publishes it with seq, waits on
// the response's seq, and the helper validates the payload, writes its rows, and publishes them back.
struct Handoff {
    uint32_t rounds;
    uint32_t spin_us;
    uint32_t max_entries;
    std::vector<double> us;    // engine-measured round trip
    uint32_t mismatches = 0;
    uint32_t timeouts = 0;
    uint32_t bad_requests = 0;
};

Handoff run_handoff(uint32_t rounds, uint32_t spin_us, uint32_t max_entries) {
    std::memset(g_map, 0, sizeof(g_map));
    auto* h = reinterpret_cast<Header*> (g_map);
    init_header(*h, 48, 512, kMappingBytes, "/models/qwen38/native", "q5_1");
    auto* req = request_at(g_map, h->request_off);
    auto* res = response_at(g_map, h->response_off);

    Handoff out{rounds, spin_us, max_entries, {}, 0, 0, 0};

    std::thread helper([&] {
        uint32_t last = 0;
        for (;;) {
            uint32_t seq = req->seq.load(std::memory_order_acquire);
            if (seq == last) {
                // Same discipline as the real helper: spin a while, then park.
                if (!wait_until(&req->seq, &req->helper_sleeping, spin_us, monotonic_ns() + 2000000000ull,
                               [last] (uint32_t v) { return v != last; })) {
                    ++out.bad_requests;   // only the engine's quit can end this loop
                    break;
                }
                seq = req->seq.load(std::memory_order_acquire);
            }
            last = seq;
            if (req->layer < 0) break;

            const int n = req->n_entries;
            if (n <= 0 || n > kCap || req->n_tok <= 0 || req->n_tok > kMaxTok) {
                ++out.bad_requests;
                res->status = kStatusFailed;
                res->n_rows = 0;
                res->seq.store(seq, std::memory_order_release);
                wake_if_parked(&res->seq, &res->engine_sleeping);
                continue;
            }
            for (int r = 0; r < n; ++r) {
                const int e = req->expert[r];
                const float* x = &req->x[(uint32_t) (r % req->n_tok) * kH];
                float* row = &res->rows[(uint32_t) r * kH];
                for (int c = 0; c < kH; ++c) row[c] = x[c] * 2.0f + (float) e;
                res->entry[r] = req->entry[r];
            }
            res->status = kStatusOk;
            res->n_rows = n;
            res->seq.store(seq, std::memory_order_release);      // the payload is written; the row set is now visible
            wake_if_parked(&res->seq, &res->engine_sleeping);
            bump_heartbeat(*h);
        }
    });

    uint32_t seq = 0;
    for (uint32_t it = 0; it < rounds; ++it) {
        const int n_tok = 1 + (int) (it % kMaxTok);
        const int n = 1 + (int) (it % max_entries);
        req->layer = (int32_t) (it % 48);
        req->n_tok = n_tok;
        req->n_entries = n;
        for (int i = 0; i < n; ++i) {
            req->entry[i] = i * 3;
            req->expert[i] = (int32_t) ((it + (uint32_t) i) % 512);
        }
        for (int i = 0; i < n_tok; ++i)
            for (int c = 0; c < kH; ++c) req->x[(uint32_t) i * kH + c] = (float) (i * 1000 + c % 977) * 0.25f;

        ++seq;
        const uint64_t t0 = monotonic_ns();
        req->seq.store(seq, std::memory_order_release);
        wake_if_parked(&req->seq, &req->helper_sleeping);

        if (!wait_until(&res->seq, &res->engine_sleeping, spin_us, t0 + 2000000000ull,
                        [seq] (uint32_t v) { return v == seq; })) {
            ++out.timeouts;
            break;
        }
        out.us.push_back((double) (monotonic_ns() - t0) / 1000.0);

        if (res->status != kStatusOk || res->n_rows != n) ++out.mismatches;
        else {
            for (int r = 0; r < n; ++r) {
                const float* x = &req->x[(uint32_t) (r % n_tok) * kH];
                const float* row = &res->rows[(uint32_t) r * kH];
                if (res->entry[r] != req->entry[r]) { ++out.mismatches; break; }
                for (int c = 0; c < kH; c += 251) {
                    const float want = x[c] * 2.0f + (float) req->expert[r];
                    if (row[c] != want) { ++out.mismatches; break; }
                }
            }
        }
    }

    seq++;
    req->layer = -1;                     // the quit request: the helper's own loop condition
    req->seq.store(seq, std::memory_order_release);
    wake_always(&req->seq);
    helper.join();
    return out;
}

void test_handoff() {
    std::printf("\ntwo threads, one request in flight, the plan's publish/wake/wait order\n");
    for (auto [rounds, spin, entries, name] : std::vector<std::tuple<uint32_t, uint32_t, uint32_t, const char*>> {
             {20000, 200, 1, "the transport alone: one row each way"},
             {20000, 200, kCap, "a full request: up to 80 rows of 2560 floats answered"},
             {2000, 0, kCap, "the same, both sides parked in the futex (no spin budget)"},
         }) {
        const Handoff h = run_handoff(rounds, spin, entries);
        std::printf("  %-34s p50 %8.2f us  p99 %8.2f us  p99.9 %8.2f us  timeouts %u  mismatches %u\n", name,
                    pct(h.us, 0.50), pct(h.us, 0.99), pct(h.us, 0.999), h.timeouts, h.mismatches);
        check(h.timeouts == 0 && h.mismatches == 0 && h.bad_requests == 0 && h.us.size() == rounds,
              "every round trip answered, and every row was the row that was asked for");
    }
}

void test_no_partner() {
    std::printf("\na partner that never answers\n");
    std::memset(g_map, 0, sizeof(g_map));
    auto* h = reinterpret_cast<Header*> (g_map);
    init_header(*h, 48, 512, kMappingBytes, "", "");
    auto* res = response_at(g_map, h->response_off);
    const uint64_t t0 = monotonic_ns();
    const bool got = wait_until(&res->seq, &res->engine_sleeping, 200, t0 + 150000000ull,
                                [] (uint32_t v) { return v == 7; });
    const double ms = (double) (monotonic_ns() - t0) / 1000000.0;
    std::printf("  gave up after %.1f ms of a 150 ms patience window\n", ms);
    check(!got, "the wait returns false instead of hanging");
    check(ms >= 100.0 && ms <= 400.0, "and it used the whole window, not a spin and not forever");
}

}  // namespace

int main() {
    std::printf("xhelper protocol test - Option A Phase 2 (Gate 2)\n");
    std::printf("kH %d  kMaxTok %d  kK %d  kCap %d  mapping %llu bytes\n", kH, kMaxTok, kK, kCap,
                (unsigned long long) kMappingBytes);
    test_sizes();
    test_header();
    test_bulk();
    test_heartbeat();
    test_handoff();
    test_no_partner();
    std::printf("\n%s (%d failed)\n", g_fails ? "FAIL" : "PASS", g_fails);
    return g_fails ? 1 : 0;
}
