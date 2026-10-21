// src/xhelper/helper_main.cpp - strata-expert-helper: the AMD half of Option A.
//
// docs/plans/OPTION_A_EXCLUSIVE_RAM.md Phase 3.  A separate process whose only job is to hold experts in the VRAM of
// one AMD card and answer requests about them.  It is forked by the CUDA engine with the shared mapping's fd
// inherited, so everything it needs - the pack path, the native preset, the model geometry, the pair list it was
// offered, the VRAM budget - comes from `include/strata/xhelper/protocol.hpp` and cannot disagree with the engine.
// It builds only when STRATA_ENABLE_HIP is ON, and only on Linux: the mapping it lives in is a memfd.
//
// Startup, request loop and exit follow the plan:
//   mmap -> header_ok -> Loading -> FileExpertSource -> HelperTier::open -> publish the loaded pairs -> Ready
//   -> wait on Request::seq -> HelperTier::run -> publish Response -> ... -> Exited.
// Exit codes: 2 the mapping is not one this helper can use, 3 any other failure (model, VRAM, device), 0 a clean stop.
#include "helper_prompt.hpp"
#include "helper_tier.hpp"

#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/xhelper/protocol.hpp"

#include <cuda_runtime.h>

#if !defined(__linux__)
#error "strata-expert-helper is Linux-only: its shared mapping is a memfd or a shm object"
#endif

#include <atomic>
#include <cstdarg>
#include <cerrno>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

using namespace strata::xhelper;

std::atomic<int> g_stop{0};   // SIGINT/SIGTERM, or the engine going away

void on_signal(int) { g_stop.store(1, std::memory_order_relaxed); }

struct Options {
    int shm_fd = -1;
    uint32_t spin_us = kDefaultSpinUs;
    bool verbose = false;
};

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --shm-fd N [--spin-us N] [--verbose]\n"
                 "  N is an inherited descriptor of the engine's shared mapping; the model, the pair list and the\n"
                 "  VRAM budget are read from that mapping, not from the command line, so the two halves cannot\n"
                 "  disagree about which model is being served.\n",
                 argv0);
}

bool parse(int argc, char** argv, Options& o) {
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--help" || a == "-h") { usage(argv[0]); return false; }
        else if (a == "--verbose") o.verbose = true;
        else if (a == "--shm-fd" && i + 1 < argc) o.shm_fd = std::atoi(argv[++i]);
        else if (a == "--spin-us" && i + 1 < argc) o.spin_us = (uint32_t) std::atoi(argv[++i]);
        else { std::fprintf(stderr, "%s: unknown argument '%s'\n", argv[0], a.c_str()); usage(argv[0]); return false; }
    }
    if (o.shm_fd < 0) { std::fprintf(stderr, "%s: --shm-fd is required\n", argv[0]); usage(argv[0]); return false; }
    return true;
}

void log(const Options& o, const char* fmt, ...) {
    if (!o.verbose) return;
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "[helper pid %d] ", (int) getpid());
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
}

// STRATA_XH_STATS=<path> (the server sets it for the Monitor): the bytes this helper moves across its card's PCIe
// link, counted from each request's own sizes (the activations in, the rows or the per-token sums out, the experts
// uploaded at start), and the experts it holds.  The amdgpu driver has no PCIe traffic counter for every card
// (RDNA4's sysfs has no pcie_bw), so this is the Monitor's "PCIe to / from card" for the helper.  Written at most once
// a second, after the response is published, as one line of key=value through a rename (a reader never sees half).
struct Stats {
    std::string path, tmp;
    uint64_t to_card = 0, from_card = 0, requests = 0, prompt_requests = 0, experts = 0;
    uint64_t last_ns = 0;
    explicit Stats(const char* p) : path(p != nullptr ? p : "") { if (!path.empty()) tmp = path + ".tmp"; }
    void write(bool force = false) {
        if (path.empty()) return;
        const uint64_t now = monotonic_ns();
        if (!force && now - last_ns < 1000000000ull) return;
        last_ns = now;
        char line[256];
        const int n = std::snprintf(line, sizeof line,
                                    "to_card_bytes=%llu from_card_bytes=%llu experts=%llu requests=%llu "
                                    "prompt_requests=%llu t_ns=%llu\n",
                                    (unsigned long long) to_card, (unsigned long long) from_card,
                                    (unsigned long long) experts, (unsigned long long) requests,
                                    (unsigned long long) prompt_requests, (unsigned long long) now);
        const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
        if (fd < 0) return;
        const bool ok = n > 0 && ::write(fd, line, (size_t) n) == n;
        ::close(fd);
        if (ok) ::rename(tmp.c_str(), path.c_str());
    }
};

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    if (!parse(argc, argv, opt)) return 2;

    struct stat st{};
    if (fstat(opt.shm_fd, &st) != 0 || st.st_size < (off_t) sizeof(Header)) {
        std::fprintf(stderr, "helper: fd %d is not a shared expert mapping (size %lld)\n", opt.shm_fd,
                     (long long) st.st_size);
        return 2;
    }
    const size_t mapped = (size_t) st.st_size;
    void* base = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_SHARED, opt.shm_fd, 0);
    if (base == MAP_FAILED) {
        std::fprintf(stderr, "helper: mmap of fd %d failed: %s\n", opt.shm_fd, std::strerror(errno));
        return 2;
    }
    auto* hdr = reinterpret_cast<Header*> (base);
    Request* req = nullptr;
    Response* res = nullptr;
    if (!header_ok(*hdr, mapped)) {
        // Only now is it safe to write anything: the offsets and the fields this writes are the fixed ones.
        std::snprintf(hdr->error, sizeof(hdr->error), "%.511s",
                      "helper: the engine's mapping does not match this helper (magic, version, geometry or offsets)");
        hdr->helper_pid = (int32_t) getpid();
        hdr->helper_state.store((uint32_t) HelperState::Failed, std::memory_order_release);
        std::fprintf(stderr, "helper: %s\n", hdr->error);
        return 2;
    }
    req = request_at(base, hdr->request_off);
    res = response_at(base, hdr->response_off);
    const Pair* assigned = pairs_at(base, hdr->assigned_off);

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);
    const pid_t parent = getppid();

    hdr->helper_pid = (int32_t) getpid();
    hdr->helper_state.store((uint32_t) HelperState::Loading, std::memory_order_release);
    bump_heartbeat(*hdr);
    log(opt, "mapping %zu bytes, %u x %u experts, H %u, %u offered pairs, pack '%s', native '%s', budget %llu MiB",
        mapped, hdr->n_layers, hdr->n_expert, hdr->h, hdr->n_assigned, hdr->model_pack, hdr->native_preset,
        (unsigned long long) (hdr->vram_budget_bytes >> 20));

    if (hdr->model_pack[0] == '\0' || hdr->n_assigned == 0) {
        std::snprintf(hdr->error, sizeof(hdr->error), "%.511s", "helper: the engine offered no pack or no pairs");
        hdr->helper_state.store((uint32_t) HelperState::Failed, std::memory_order_release);
        return 3;
    }

    // The same path the engine takes (src/program/generate.cpp): load the pack's expert layout, which the kernels and
    // FileExpertSource::open both need, then open the source on it.
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(hdr->model_pack, hdr->n_layers, hdr->n_expert, err)) {
        std::snprintf(hdr->error, sizeof(hdr->error), "%.511s", err.c_str());
        hdr->helper_state.store((uint32_t) HelperState::Failed, std::memory_order_release);
        std::fprintf(stderr, "helper: loading the pack's expert layout failed: %s\n", err.c_str());
        return 3;
    }
    strata::core::FileExpertSource source;
    source.set_gguf(hdr->native_preset);
    if (!source.open(hdr->model_pack, hdr->n_layers, hdr->n_expert, err)) {
        std::snprintf(hdr->error, sizeof(hdr->error), "%.511s", err.c_str());
        hdr->helper_state.store((uint32_t) HelperState::Failed, std::memory_order_release);
        std::fprintf(stderr, "helper: opening the expert pack failed: %s\n", err.c_str());
        return 3;
    }

    std::vector<std::pair<int32_t, int32_t>> offered;
    offered.reserve(hdr->n_assigned);
    for (uint32_t i = 0; i < hdr->n_assigned; ++i) offered.push_back({ assigned[i].layer, assigned[i].expert });
    uint64_t free_bytes = 0, total_bytes = 0;
    if (!HelperTier::memory_info(0, free_bytes, total_bytes, err)) {
        std::snprintf(hdr->error, sizeof(hdr->error), "%.511s", err.c_str());
        hdr->helper_state.store((uint32_t) HelperState::Failed, std::memory_order_release);
        std::fprintf(stderr, "helper: %s\n", err.c_str());
        return 3;
    }
    log(opt, "device 0: %.2f GiB free of %.2f GiB", (double) free_bytes / 1073741824.0,
        (double) total_bytes / 1073741824.0);

    // Phase 7.1: with the bulk region the engine asks for prompt layers too; their buffers come off the expert budget
    const bool prompt_on = hdr->bulk_bytes != 0;
    const uint64_t prompt_reserve = prompt_on ? HelperPrompt::bytes_needed(hdr->bulk_max_tok) : 0;
    if (prompt_on)
        log(opt, "prompt kind on: %u tokens a request, %llu MiB of device buffers kept off the expert budget",
            hdr->bulk_max_tok, (unsigned long long) (prompt_reserve >> 20));

    HelperTier tier;
    std::vector<std::pair<int32_t, int32_t>> loaded;
    auto* req_x = reinterpret_cast<void*> (reinterpret_cast<char*> (req) + offsetof(Request, x));
    auto* res_rows = reinterpret_cast<void*> (reinterpret_cast<char*> (res) + offsetof(Response, rows));
    if (!tier.open(0, hdr->vram_budget_bytes, hdr->n_layers, hdr->n_expert, offered, source, req_x,
                   sizeof(req->x), res_rows, sizeof(res->rows), loaded, err, prompt_reserve)) {
        std::snprintf(hdr->error, sizeof(hdr->error), "%.511s", err.c_str());
        hdr->helper_state.store((uint32_t) HelperState::Failed, std::memory_order_release);
        std::fprintf(stderr, "helper: loading the expert tier failed: %s\n", err.c_str());
        return 3;
    }

    Pair* loaded_table = pairs_at(base, hdr->loaded_off);
    for (size_t i = 0; i < loaded.size(); ++i) loaded_table[i] = Pair{ loaded[i].first, loaded[i].second };
    hdr->n_loaded = (uint32_t) loaded.size();

    // A prompt kind that cannot start is not fatal: the engine sees prompt_ok 0 and reads these experts as before.
    HelperPrompt prompt;
    hdr->prompt_ok = 0;
    if (prompt_on) {
        std::string perr;
        if (prompt.open(0, hdr->bulk_max_tok, hdr->n_layers, hdr->n_expert, reinterpret_cast<char*> (base) + hdr->bulk_off,
                        hdr->bulk_bytes, hdr->prompt_layers, perr)) {
            hdr->prompt_ok = 1;
            uint32_t n = 0;
            for (uint32_t l = 0; l < hdr->n_layers; ++l) n += hdr->prompt_layers[l];
            log(opt, "prompt products ready for %u of %u layers", n, hdr->n_layers);
        } else {
            for (auto& v : hdr->prompt_layers) v = 0;
            std::fprintf(stderr, "helper: the prompt kind is off: %s\n", perr.c_str());
        }
    }
    bump_heartbeat(*hdr);
    hdr->helper_state.store((uint32_t) HelperState::Ready, std::memory_order_release);
    log(opt, "loaded %zu of %u offered experts (%.2f GiB of device slots), %s", loaded.size(), hdr->n_assigned,
        tier.gib(), tier.zero_copy() ? "reading the mapping directly" : "staged through private pinned buffers");

    Stats stats(std::getenv("STRATA_XH_STATS"));
    stats.experts = loaded.size();
    stats.to_card = (uint64_t) (tier.gib() * 1073741824.0);   // the experts' upload
    stats.write(true);

    uint32_t last = 0;
    int exit_code = 0;
    for (;;) {
        // A bounded wait, so that an engine which dies without saying Shutdown is noticed rather than waited on.
        const uint64_t deadline = monotonic_ns() + 250000000ull;
        if (!wait_until(&req->seq, &req->helper_sleeping, opt.spin_us, deadline,
                        [last] (uint32_t v) { return v != last; })) {
            const uint32_t engine = hdr->engine_state.load(std::memory_order_acquire);
            if (engine == (uint32_t) EngineState::Shutdown || g_stop.load() || getppid() != parent) {
                exit_code = 0;
                break;
            }
            bump_heartbeat(*hdr);   // idle: the engine's "older than 5 s is dead" rule needs a pulse
            stats.write();
            continue;
        }
        last = req->seq.load(std::memory_order_acquire);
        const bool shutting = hdr->engine_state.load(std::memory_order_acquire) == (uint32_t) EngineState::Shutdown ||
                              g_stop.load() || getppid() != parent;
        if (shutting) { exit_code = 0; break; }

        const int32_t layer = req->layer, n_tok = req->n_tok, n_entries = req->n_entries;
        int32_t n_rows = 0;
        bool ok = false;
        if (req->kind == kKindPrompt) {
            if (layer < 0 || (uint32_t) layer >= hdr->n_layers || !hdr->prompt_layers[layer]) {
                err = "helper: a prompt request for layer " + std::to_string(layer) + ", which it does not run";
            } else {
                ok = prompt.run(tier, layer, n_tok, n_entries, req->n_rows, err);
                n_rows = ok ? req->n_rows : 0;
            }
        } else if (req->kind != kKindRows) {
            err = "helper: a request of unknown kind " + std::to_string(req->kind);
        } else if (layer < 0 || (uint32_t) layer >= hdr->n_layers || n_tok <= 0 || n_tok > kMaxTok || n_entries <= 0 ||
            n_entries > kCap) {
            err = "helper: a request outside the model or the window (layer " + std::to_string(layer) +
                  ", n_tok " + std::to_string(n_tok) + ", n_entries " + std::to_string(n_entries) + ")";
        } else {
            ok = tier.run(layer, req->x, n_tok, req->entry, req->expert, n_entries, res->rows, res->entry, n_rows, err);
        }

        // The response is published before any decision to stop, so the engine is never left waiting on a helper
        // that has already decided to exit.  A failed request carries no rows: status says so, and nothing partial
        // was written under status ok.
        res->status = ok ? kStatusOk : kStatusFailed;
        res->n_rows = ok ? n_rows : 0;
        if (!ok) std::snprintf(hdr->error, sizeof(hdr->error), "%.511s", err.c_str());
        res->seq.store(last, std::memory_order_release);
        wake_if_parked(&res->seq, &res->engine_sleeping);
        bump_heartbeat(*hdr);
        if (ok) {
            // a prompt request: x and the per-row weights and pairs in, one sum per token out; decode: x in, rows out
            const uint64_t x_bytes = (uint64_t) n_tok * (uint64_t) kH * sizeof(float);
            if (req->kind == kKindPrompt) {
                stats.to_card += x_bytes + (uint64_t) n_rows * (sizeof(float) + sizeof(int32_t));
                stats.from_card += x_bytes;
                ++stats.prompt_requests;
            } else {
                stats.to_card += x_bytes;
                stats.from_card += (uint64_t) n_rows * (uint64_t) kH * sizeof(float);
                ++stats.requests;
            }
        }
        stats.write();
        if (ok && req->kind == kKindRows && opt.verbose && (tier.requests() % 512) == 0)
            log(opt, "answered %llu requests, %llu rows", (unsigned long long) tier.requests(),
                (unsigned long long) tier.computed());
        // A failed prompt request turns the prompt kind off (the engine reads those experts as before) and leaves
        // decode's rows to this helper; a failed decode request is a device error, and not a reason to keep pretending.
        if (!ok && req->kind == kKindPrompt) {
            std::fprintf(stderr, "helper: a prompt request failed, the prompt kind is off: %s\n", err.c_str());
            prompt.close();
            for (auto& v : hdr->prompt_layers) v = 0;
            continue;
        }
        if (!ok) { exit_code = 3; break; }
    }

    log(opt, "stopping after %llu requests (%.2f GiB held), %llu prompt requests (%llu rows)",
        (unsigned long long) tier.requests(), tier.gib(), (unsigned long long) prompt.requests(),
        (unsigned long long) prompt.rows());
    hdr->helper_state.store((uint32_t) HelperState::Exited, std::memory_order_release);
    bump_heartbeat(*hdr);
    if (!stats.path.empty()) ::unlink(stats.path.c_str());   // a stopped helper moves nothing: no stale reading
    prompt.close();
    tier.close();
    source.close();
    munmap(base, mapped);
    return exit_code;
}
