// tests/xhelper/external_experts_test.cpp - Option A Phase 4: the engine's ExternalExperts against a real helper.
//
// CUDA build, Linux.  Needs STRATA_TEST_PACK (a native pack), STRATA_TEST_NATIVE (its GGUF, shard 1) and
// STRATA_TEST_HELPER (strata-expert-helper from a HIP build); exits 77, skipped, without them.
//
//   1. open() spawns the helper with 64 pairs (layers 0, middle, last) and waits for Ready.
//   2. 200 requests of 1..8 tokens whose routed experts are half the helper's and half not: begin() must own exactly
//      the entries whose expert the helper holds, finish() must write exactly those rows, and each one must match the
//      CPU pool's own kernels on the same blob and activation (the helper parity test's bar, 3e-2: the engine's own
//      CPU/GPU gap on the low-bit packs, see tests/xhelper/helper_tier_parity.cpp).
//   3. kill -9 the helper in the middle of a request: finish() fails once, alive() is false, and the next begin()
//      owns nothing.
#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/external_experts.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <csignal>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace cpu = strata::kernels::cpu;

namespace {

constexpr int64_t H = cpu::H, FF = cpu::FF, K = 10;
constexpr double kBar = 3e-2;

double rel(const float* a, const std::vector<float>& b) {
    double n = 0, d = 0;
    for (size_t i = 0; i < b.size(); ++i) { n += std::fabs((double) a[i] - b[i]); d += std::fabs((double) b[i]); }
    return n / (d + 1e-30);
}

const char* env_str(const char* name) {
    const char* v = std::getenv(name);
    return (v && v[0]) ? v : nullptr;
}

// One expert, one activation, the CPU pool's path (as tests/xhelper/helper_tier_parity.cpp).
void cpu_row(const cpu::NativeFmt& f, const uint8_t* blob, const float* x, std::vector<float>& out) {
    std::vector<uint8_t> act(cpu::kNativeActBytes), hq(cpu::kNativeHBytes);
    std::vector<float> ff((size_t) FF);
    out.assign((size_t) H, 0.f);
    cpu::native_quant_act(f, x, act.data());
    const void* a[1] = { act.data() };
    float* ffp[1] = { ff.data() };
    cpu::native_gu_rows(f, blob, a, 1, ffp, 0, (int) FF);
    cpu::native_quant_h(f, ff.data(), hq.data());
    const void* hp[1] = { hq.data() };
    float* op[1] = { out.data() };
    cpu::native_down_rows(f, blob, hp, 1, op, 0, (int) H);
}

}  // namespace

int main() {
    const char* pack = env_str("STRATA_TEST_PACK");
    const char* native = env_str("STRATA_TEST_NATIVE");
    const char* helper = env_str("STRATA_TEST_HELPER");
    if (!pack || !native || !helper) {
        std::printf("external_experts_test: needs STRATA_TEST_PACK, STRATA_TEST_NATIVE and STRATA_TEST_HELPER; skipped\n");
        return 77;
    }
    const int64_t n_layers = 48, n_expert = 512;
    std::string err;
    if (!cpu::expert_layout_load(pack, n_layers, n_expert, err)) {
        std::fprintf(stderr, "external_experts_test: %s\n", err.c_str());
        return 1;
    }
    const auto& lay = cpu::expert_layout();
    if (!lay.native) { std::printf("external_experts_test: not a native pack; skipped\n"); return 77; }
    strata::core::FileExpertSource source;
    source.set_gguf(native);
    if (!source.open(pack, n_layers, n_expert, err)) {
        std::fprintf(stderr, "external_experts_test: %s\n", err.c_str());
        return 1;
    }

    // 64 pairs: experts 0..21 of layer 0 and the middle one, 0..19 of the last.  Experts 22..43 of each are not offered.
    const int32_t layers[3] = { 0, (int32_t) n_layers / 2, (int32_t) n_layers - 1 };
    std::vector<std::pair<int32_t, int32_t>> ranked;
    for (int32_t l : layers)
        for (int32_t e = 0; e < 22 && ranked.size() < 64; ++e) ranked.emplace_back(l, e);

    strata::core::ExternalExperts ext;
    strata::core::ExternalExperts::Options opt;
    opt.helper_exe = helper;
    if (const char* d = env_str("STRATA_TEST_HIP_DEVICE")) opt.hip_device = std::atoi(d);
    const strata::core::ExpertCache no_primary;   // an unopened cache holds nothing
    if (!ext.open(opt, pack, native, n_layers, n_expert, ranked, no_primary, err)) {
        std::fprintf(stderr, "external_experts_test: %s\n", err.c_str());
        return 1;
    }
    std::printf("external_experts_test: helper pid %d, %zu of %zu pairs loaded (%.2f GiB) in %.1f s\n", ext.pid(),
                ext.loaded().size(), ranked.size(), ext.gib(), ext.load_seconds());
    if (ext.loaded().size() != ranked.size()) {
        std::fprintf(stderr, "external_experts_test: the helper loaded %zu of 64 pairs\n", ext.loaded().size());
        return 1;
    }

    std::mt19937 rng(20261011);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> x((size_t) cpu::MAXT * H), out((size_t) cpu::MAXT * K * H), want;
    std::vector<int32_t> ids((size_t) cpu::MAXT * K);
    int failures = 0;
    int64_t rows = 0, unowned = 0;
    double worst = 0;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (int r = 0; r < 200; ++r) {
        const int32_t layer = layers[r % 3];
        const int64_t n_tok = 1 + (int64_t) (rng() % cpu::MAXT), n = n_tok * K;
        for (auto& v : x) v = nd(rng);
        for (int64_t i = 0; i < n; ++i) ids[(size_t) i] = (int32_t) (rng() % 44);   // 0..21 the helper's (most of them)
        std::fill(out.begin(), out.end(), nan);
        if (!ext.begin(layer, x.data(), ids.data(), n_tok, K, nullptr, nullptr, err) || !ext.finish(out.data(), err)) {
            std::fprintf(stderr, "external_experts_test: request %d: %s\n", r, err.c_str());
            return 1;
        }
        for (int64_t i = 0; i < n; ++i) {
            const int32_t e = ids[(size_t) i];
            const bool held = ext.holds(layer, e);
            const float* row = out.data() + (size_t) i * H;
            if (ext.owns(i) != held) {
                std::fprintf(stderr, "external_experts_test: layer %d expert %d: owns %d, holds %d\n", layer, e, (int) ext.owns(i), (int) held);
                ++failures;
                continue;
            }
            if (!held) {
                ++unowned;
                if (!std::isnan(row[0])) { std::fprintf(stderr, "external_experts_test: a row the helper does not own was written\n"); ++failures; }
                continue;
            }
            cpu_row(lay.fmt[(size_t) layer], source.blob(layer, e), x.data() + (size_t) (i / K) * H, want);
            const double d = rel(row, want);
            worst = std::max(worst, d);
            ++rows;
            if (!(d < kBar)) {
                std::fprintf(stderr, "external_experts_test: layer %d expert %d entry %lld: rel L1 %.4g\n", layer, e, (long long) i, d);
                ++failures;
            }
        }
    }
    std::printf("external_experts_test: 200 requests, %lld helper rows checked (worst rel L1 %.4g, bar %.0e), %lld rows "
                "left to the CPU untouched; begin %.1f ms, wait %.1f ms\n", (long long) rows, worst, kBar,
                (long long) unowned, ext.ms_begin(), ext.ms_wait());

    // The helper dies with a request in flight.
    for (int64_t i = 0; i < K; ++i) ids[(size_t) i] = (int32_t) i;
    if (!ext.begin(layers[0], x.data(), ids.data(), 1, K, nullptr, nullptr, err)) { std::fprintf(stderr, "begin: %s\n", err.c_str()); return 1; }
    kill(ext.pid(), SIGKILL);
    const bool finished = ext.finish(out.data(), err);
    std::printf("external_experts_test: after kill -9: finish %s (%s), alive %d\n", finished ? "succeeded" : "failed",
                err.c_str(), (int) ext.alive());
    if (ext.alive()) {   // it answered before the signal landed: the next request is the one that finds it gone
        if (!ext.begin(layers[0], x.data(), ids.data(), 1, K, nullptr, nullptr, err) || ext.finish(out.data(), err)) ++failures;
    }
    if (ext.alive()) { std::fprintf(stderr, "external_experts_test: still alive after the helper was killed\n"); ++failures; }
    if (!ext.begin(layers[0], x.data(), ids.data(), 1, K, nullptr, nullptr, err)) ++failures;
    for (int64_t i = 0; i < K; ++i) if (ext.owns(i)) { std::fprintf(stderr, "external_experts_test: a dead helper still owns rows\n"); ++failures; break; }
    if (!ext.finish(out.data(), err)) ++failures;   // nothing pending: true
    ext.close();
    source.close();
    std::printf("%s\n", failures == 0 && rows > 0 ? "PASS" : "FAIL");
    return failures == 0 && rows > 0 ? 0 : 1;
}
