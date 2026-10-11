// tests/xhelper/helper_tier_parity.cpp - the rows the expert helper returns are the rows the CPU pool would return.
//
// docs/plans/OPTION_A_EXCLUSIVE_RAM.md Gate 3.  HIP build only, and it needs a real pack: with STRATA_TEST_PACK unset
// it says so and exits 77, which ctest reports as skipped.
//
//     STRATA_TEST_PACK=/path/to/pack STRATA_TEST_NATIVE=/path/to/gguf ./build/helper_tier_parity
//
// It loads 32 pairs spread over three layers into the helper's tier, asks it for whole requests (1 to 8 tokens, up to
// 80 routed entries, some experts repeated), and for every row it gets back it computes the same expert on the same
// activation on the CPU with the kernels the pool uses - native_quant_act, native_gu_rows, native_quant_h,
// native_down_rows on the same blob bytes FileExpertSource hands to the helper - and compares.
//
// The measure is the relative L1 difference (rel() of src/kernels/native_expert_parity.cpp) per row, under kBar
// below: "the same answer, with the rounding the two paths do differently", not a bit-for-bit claim.
#include "helper_tier.hpp"

#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/xhelper/protocol.hpp"

#include <cuda_runtime.h>

#include <sys/mman.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace xh = strata::xhelper;
namespace cpu = strata::kernels::cpu;

namespace {

constexpr int64_t H = cpu::H, FF = cpu::FF;

// The bar is native_expert_parity's: each path within 3e-2 of the float expert.  The plan said 5e-3 from that file's
// "a few 1e-3", which does not hold for the low-bit packs: on the IQ3_XXS model (RX 9060 XT, 2026-10-11)
// native_expert_parity itself measures CPU vs GPU 1.45e-2 .. 1.70e-2 on layers 0, 1, 24, 47, and this test measured
// the helper's rows 1.3e-2 .. 2.0e-2 from the CPU's, the same with STRATA_XHELPER_STAGED=1 (worst 0.01971 both ways).
constexpr double kBar = 3e-2;

double rel(const std::vector<float>& a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < a.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

const char* env_str(const char* name, const char* fallback) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? v : fallback;
}

int env_int(const char* name, int fallback) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? std::atoi(v) : fallback;
}

// One expert, one activation, the CPU pool's own path.
bool cpu_row(const cpu::NativeFmt& f, const uint8_t* blob, const float* x, std::vector<float>& out, std::string& err) {
    std::vector<uint8_t> act(cpu::kNativeActBytes), hq(cpu::kNativeHBytes);
    std::vector<float> ff((size_t) FF), one((size_t) H);
    cpu::native_quant_act(f, x, act.data());
    const void* a[1] = { act.data() };
    float* ffp[1] = { ff.data() };
    cpu::native_gu_rows(f, blob, a, 1, ffp, 0, (int) FF);
    cpu::native_quant_h(f, ff.data(), hq.data());
    const void* hp[1] = { hq.data() };
    float* op[1] = { one.data() };
    cpu::native_down_rows(f, blob, hp, 1, op, 0, (int) H);
    out = one;
    (void) err;
    return true;
}

}  // namespace

int main() {
    const char* pack = std::getenv("STRATA_TEST_PACK");
    if (!pack || !pack[0]) {
        std::printf("helper_tier_parity: STRATA_TEST_PACK is not set, so there is no expert pack to load.\n");
        std::printf("  STRATA_TEST_PACK=/path/to/pack STRATA_TEST_NATIVE=/path/to/model.gguf helper_tier_parity\n");
        return 77;   // ctest: skipped, not failed
    }
    const int64_t n_layers = env_int("STRATA_TEST_LAYERS", 48);
    const int64_t n_expert = env_int("STRATA_TEST_EXPERTS", 512);
    const int device = env_int("STRATA_TEST_HIP_DEVICE", 0);

    std::string err;
    if (!cpu::expert_layout_load(pack, n_layers, n_expert, err)) {
        std::fprintf(stderr, "helper_tier_parity: loading the expert layout of '%s' failed: %s\n", pack, err.c_str());
        return 1;
    }
    strata::core::FileExpertSource source;
    source.set_gguf(env_str("STRATA_TEST_NATIVE", ""));
    if (!source.open(pack, n_layers, n_expert, err)) {
        std::fprintf(stderr, "helper_tier_parity: opening '%s' as %lld x %lld failed: %s\n", pack, (long long) n_layers,
                     (long long) n_expert, err.c_str());
        return 1;
    }
    const auto& lay = cpu::expert_layout();
    if (!lay.native) {
        std::printf("helper_tier_parity: this pack is not a native one, and the CPU reference path here is the native "
                    "one.  Skipped rather than comparing against something else.\n");
        return 77;
    }

    // 32 pairs, spread over three layers and far apart within them: the point is that the tier holds what it was
    // given and answers for it layer by layer, not that one layer is right.
    const int32_t layers[3] = { 0, (int32_t) (n_layers / 2), (int32_t) (n_layers - 1) };
    std::vector<std::pair<int32_t, int32_t>> offered;
    for (int32_t e : { 0, 3, 11, 19, 23, 31, 37, 41, 47, 53, 61 })
        for (int32_t l : layers)
            if ((int64_t) l < n_layers && (int64_t) e < n_expert && offered.size() < 32) offered.push_back({ l, e });

    // A real mapping of the real protocol layout, so what gets registered with the device is what the helper
    // registers: the Request::x and Response::rows regions inside a block another process would have made.
    std::unique_ptr<void, void (*) (void*)> map(mmap(nullptr, xh::kMappingBytes, PROT_READ | PROT_WRITE,
                                                     MAP_SHARED | MAP_ANONYMOUS, -1, 0), [](void* p) { munmap(p, xh::kMappingBytes); });
    if (!map.get() || map.get() == MAP_FAILED) { std::fprintf(stderr, "helper_tier_parity: mmap failed\n"); return 1; }
    void* base = map.get();
    auto* hdr = reinterpret_cast<xh::Header*> (base);
    if (!xh::init_header(*hdr, (uint32_t) n_layers, (uint32_t) n_expert, xh::kMappingBytes, pack,
                         env_str("STRATA_TEST_NATIVE", ""))) {
        std::fprintf(stderr, "helper_tier_parity: the geometry does not fit the mapping\n");
        return 1;
    }
    auto* req = xh::request_at(base, hdr->request_off);
    auto* res = xh::response_at(base, hdr->response_off);

    xh::HelperTier tier;
    std::vector<std::pair<int32_t, int32_t>> loaded;
    if (!tier.open(device, 0, n_layers, n_expert, offered, source,
                   reinterpret_cast<char*> (req) + offsetof(xh::Request, x), sizeof(req->x),
                   reinterpret_cast<char*> (res) + offsetof(xh::Response, rows), sizeof(res->rows), loaded, err)) {
        std::fprintf(stderr, "helper_tier_parity: loading the tier failed: %s\n", err.c_str());
        return 1;
    }
    std::printf("helper_tier_parity: %zu of %zu pairs loaded (%.2f GiB), %s\n", loaded.size(), offered.size(),
                tier.gib(), tier.zero_copy() ? "kernels read and write the mapping directly"
                                             : "staged through private pinned buffers");
    if (loaded.size() < 8) {
        std::fprintf(stderr, "helper_tier_parity: too few experts fit on this card to test anything\n");
        return 1;
    }

    // Which layers actually got loaded, and which of their experts: a card that can only hold a prefix of the offer
    // must still be tested, on the experts it does hold.
    std::vector<int32_t> live_layers;
    std::vector<std::vector<int32_t>> live_experts;
    for (const auto& p : loaded) {
        size_t at = live_layers.size();
        for (size_t i = 0; i < live_layers.size(); ++i) if (live_layers[i] == p.first) { at = i; break; }
        if (at == live_layers.size()) { live_layers.push_back(p.first); live_experts.emplace_back(); }
        live_experts[at].push_back(p.second);
    }

    std::mt19937 rng(20260714);
    std::normal_distribution<float> nd(0.f, 1.f);
    int checks = 0, failures = 0;
    double worst = 0.0;
    std::vector<std::vector<float>> wanted;

    for (int round = 0; round < 6; ++round) {
        // A request shaped like the engine's: n_tok tokens, each routing kK experts, so n_tok * kK distinct routed
        // entries, entry j belonging to token j / kK.  Where the tier holds fewer experts than there are entries the
        // same expert is reached from two entries, which is what the grouped call is for.
        const int n_tok = 1 + round % cpu::MAXT;
        const size_t which_layer = (size_t) round % live_layers.size();
        const int32_t layer = live_layers[which_layer];
        const std::vector<int32_t>& ours = live_experts[which_layer];
        const int n = (int) std::min<int>({(int) ours.size() + 8, xh::kCap / 2, n_tok * xh::kK});
        std::vector<float> x((size_t) n_tok * H);
        for (auto& v : x) v = nd(rng);
        std::vector<int32_t> entry(n), expert(n);
        for (int j = 0; j < n; ++j) {
            entry[j] = j;
            expert[j] = ours[(size_t) j % ours.size()];
        }
        int32_t n_rows = 0;
        if (!tier.run(layer, x.data(), n_tok, entry.data(), expert.data(), n, res->rows, res->entry, n_rows, err)) {
            std::fprintf(stderr, "helper_tier_parity: layer %d, %d tokens, %d entries: %s\n", layer, n_tok, n, err.c_str());
            ++failures;
            break;
        }
        if (n_rows <= 0) { std::fprintf(stderr, "helper_tier_parity: the helper answered with no rows\n"); ++failures; continue; }

        // Every row it produced, against the same expert applied to the same activation on the CPU.
        wanted.assign((size_t) n_rows, {});
        for (int r = 0; r < n_rows; ++r) {
            const int32_t which = res->entry[r];
            if (which < 0 || which >= n) { std::fprintf(stderr, "helper_tier_parity: row %d names entry %d\n", r, which); ++failures; continue; }
            const int32_t e = expert[which];
            const uint8_t* blob = source.blob(layer, e);
            if (!blob) { std::fprintf(stderr, "helper_tier_parity: no blob for layer %d expert %d\n", layer, e); ++failures; continue; }
            const float* xrow = x.data() + (size_t) (which / xh::kK) * H;
            std::string ignored;
            cpu_row(lay.fmt[(size_t) layer], blob, xrow, wanted[(size_t) r], ignored);
            const std::vector<float> got(res->rows + (size_t) r * H, res->rows + (size_t) (r + 1) * H);
            const double d = rel(got, wanted[(size_t) r]);
            ++checks;
            if (d > worst) worst = d;
            if (!(d < kBar)) {
                std::fprintf(stderr, "helper_tier_parity: layer %d expert %d entry %d: rel L1 %.4g (bar %.0e)\n", layer, e,
                             which, d, kBar);
                ++failures;
            }
        }
    }
    std::printf("helper_tier_parity: %d rows checked, worst relative L1 difference %.4g (bar %.0e)\n", checks, worst, kBar);
    std::printf("%s\n", (failures || checks < 8) ? "FAIL" : "PASS");
    tier.close();
    source.close();
    return (failures || checks < 8) ? 1 : 0;
}
