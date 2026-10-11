// tools/xhelper_spike/drive_helper.cpp - poke strata-expert-helper from a CPU-only process (Gate 3, second half).
//
// docs/plans/OPTION_A_EXCLUSIVE_RAM.md Phase 3: "the helper starts, loads ~12 GB, reports Ready, and exits cleanly on
// Shutdown, driven by a tiny test driver".  This is that driver.  It has no CUDA and no HIP in it on purpose: it is
// the engine's half of the handshake - make the mapping, describe the model, offer pairs, fork the helper, wait for
// Ready, send requests, then say Shutdown - and it can be built and run next to a CUDA engine without touching it.
//
//   g++ -O2 -std=c++20 -Iinclude tools/xhelper_spike/drive_helper.cpp -o build/drive_helper
//   build/drive_helper --helper build/strata-expert-helper --pack /models/qwen38 --native /models/qwen38/model.gguf
//                      --pairs 8000 --budget-gib 12 --requests 500
//
// It reports: whether the helper reached Ready, how many pairs it loaded, the round trip of each request, and the
// helper's exit code.  It does not check the numbers in the rows: that is tests/xhelper/helper_tier_parity.cpp.
#include "strata/xhelper/protocol.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <signal.h>
#include <string>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

namespace xh = strata::xhelper;

namespace {

struct Options {
    std::string helper, pack, native;
    uint32_t layers = 48, experts = 512;
    uint32_t pairs = 8000;
    uint64_t budget = 0;
    int requests = 500;
    uint32_t spin_us = xh::kDefaultSpinUs;
    int ready_timeout_s = 600;
    bool verbose = false;
};

double pct(std::vector<double> v, double p) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[(size_t) (p * (double) (v.size() - 1) + 0.5)];
}

}  // namespace

int main(int argc, char** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&](const char* what) -> long long {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", what); std::exit(2); }
            return std::atoll(argv[++i]);
        };
        if (a == "--helper") o.helper = argv[++i];
        else if (a == "--pack") o.pack = argv[++i];
        else if (a == "--native") o.native = argv[++i];
        else if (a == "--layers") o.layers = (uint32_t) next("layers");
        else if (a == "--experts") o.experts = (uint32_t) next("experts");
        else if (a == "--pairs") o.pairs = (uint32_t) next("pairs");
        else if (a == "--budget-gib") o.budget = (uint64_t) next("budget-gib") * xh::kMiB * 1024;
        else if (a == "--requests") o.requests = (int) next("requests");
        else if (a == "--spin-us") o.spin_us = (uint32_t) next("spin-us");
        else if (a == "--ready-timeout-s") o.ready_timeout_s = (int) next("ready-timeout-s");
        else if (a == "--verbose") o.verbose = true;
        else {
            std::fprintf(stderr,
                         "usage: %s --helper PATH --pack DIR [--native GGUF] [--pairs N] [--budget-gib N]\n"
                         "          [--layers N] [--experts N] [--requests N] [--spin-us N] [--ready-timeout-s N]\n",
                         argv[0]);
            return 2;
        }
    }
    if (o.helper.empty() || o.pack.empty()) { std::fprintf(stderr, "--helper and --pack are required\n"); return 2; }

    const int fd = (int) syscall(SYS_memfd_create, "strata-xhelper-drive", 0);
    if (fd < 0) { std::perror("memfd_create"); return 1; }
    if (ftruncate(fd, (off_t) xh::kMappingBytes) != 0) { std::perror("ftruncate"); return 1; }
    void* base = mmap(nullptr, xh::kMappingBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (base == MAP_FAILED) { std::perror("mmap"); return 1; }
    auto* hdr = reinterpret_cast<xh::Header*> (base);
    if (!xh::init_header(*hdr, o.layers, o.experts, xh::kMappingBytes, o.pack.c_str(), o.native.c_str())) {
        std::fprintf(stderr, "the geometry does not fit the mapping\n");
        return 1;
    }
    xh::Pair* assigned = xh::pairs_at(base, hdr->assigned_off);
    const uint32_t offered = std::min<uint32_t>(o.pairs, o.layers * o.experts);
    for (uint32_t i = 0; i < offered; ++i) {                       // hottest first; here, layer by layer
        assigned[i] = xh::Pair{ (int32_t) (i % o.layers), (int32_t) ((i / o.layers) % o.experts) };
    }
    hdr->n_assigned = offered;
    hdr->vram_budget_bytes = o.budget;
    hdr->engine_state.store((uint32_t) xh::EngineState::Assigned, std::memory_order_release);

    const pid_t child = fork();
    if (child < 0) { std::perror("fork"); return 1; }
    if (child == 0) {
        std::string fd_s = std::to_string(fd), spin = std::to_string(o.spin_us);
        std::vector<char*> args;
        args.push_back((char*) o.helper.c_str());
        args.push_back((char*) "--shm-fd");
        args.push_back(fd_s.data());
        args.push_back((char*) "--spin-us");
        args.push_back(spin.data());
        if (o.verbose) args.push_back((char*) "--verbose");
        args.push_back(nullptr);
        execv(o.helper.c_str(), args.data());
        std::fprintf(stderr, "execv %s: %s\n", o.helper.c_str(), std::strerror(errno));
        _exit(127);
    }
    std::printf("drive: forked helper pid %d, offered %u pairs of %u x %u, budget %llu MiB\n", (int) child, offered,
                o.layers, o.experts, (unsigned long long) (o.budget >> 20));
    hdr->engine_state.store((uint32_t) xh::EngineState::Running, std::memory_order_release);

    // Wait for Ready.  Loading 12 GiB of experts into the card is minutes on a spinning disk, so the patience is the
    // engine's problem too, and the heartbeat is what says the helper is still alive while it does.
    const uint64_t stop_ns = xh::monotonic_ns() + (uint64_t) o.ready_timeout_s * 1000000000ull;
    uint32_t state = 0;
    while (xh::monotonic_ns() < stop_ns) {
        state = hdr->helper_state.load(std::memory_order_acquire);
        if (state == (uint32_t) xh::HelperState::Ready) break;
        if (state == (uint32_t) xh::HelperState::Failed) break;
        int status = 0;
        if (waitpid(child, &status, WNOHANG) == child) {
            std::fprintf(stderr, "drive: the helper exited before it was ready (code %d, signal %d)\n",
                         WEXITSTATUS(status), WIFSIGNALED(status) ? WTERMSIG(status) : 0);
            if (hdr->error[0]) std::fprintf(stderr, "drive: helper said: %s\n", hdr->error);
            return 1;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (state != (uint32_t) xh::HelperState::Ready) {
        std::fprintf(stderr, "drive: the helper never reported Ready (state %u)%s%s\n", state,
                     hdr->error[0] ? ": " : "", hdr->error);
        kill(child, SIGTERM);
        int status = 0;
        waitpid(child, &status, 0);
        return 1;
    }
    const xh::Pair* loaded = xh::pairs_at(base, hdr->loaded_off);
    std::printf("drive: Ready after %.1f s, %u pairs loaded\n",
                (double) ((uint64_t) o.ready_timeout_s * 1000000000ull - (stop_ns - xh::monotonic_ns())) / 1e9,
                hdr->n_loaded);
    if (hdr->n_loaded == 0) { std::fprintf(stderr, "drive: the helper loaded nothing\n"); kill(child, SIGTERM); waitpid(child, nullptr, 0); return 1; }

    xh::Request* req = xh::request_at(base, hdr->request_off);
    xh::Response* res = xh::response_at(base, hdr->response_off);
    std::vector<double> us;
    uint32_t seq = 0, bad = 0;
    for (int r = 0; r < o.requests; ++r) {
        const uint32_t l = r % o.layers;
        if (!xh::pair_ok(*hdr, (int32_t) l, loaded[0].expert)) break;
        const int n_tok = 1 + r % xh::kMaxTok;
        const int n = std::min<int>(xh::kCap, n_tok * xh::kK);
        // Only pairs the helper said it holds.  An entry for something else is a bug in this driver, and the helper
        // answers it with an error rather than a wrong row.
        int picked = 0;
        for (uint32_t i = 0; i < hdr->n_loaded && picked < n; ++i)
            if (loaded[i].layer == (int32_t) l) { req->entry[picked] = picked; req->expert[picked] = loaded[i].expert; ++picked; }
        if (picked == 0) continue;   // this layer is not the helper's
        req->layer = (int32_t) l;
        req->n_tok = n_tok;
        req->n_entries = picked;
        for (int i = 0; i < n_tok; ++i)
            for (int c = 0; c < xh::kH; ++c) req->x[(size_t) i * xh::kH + c] = (float) ((i * 37 + c) % 101) * 0.01f - 0.5f;
        ++seq;
        const uint64_t t0 = xh::monotonic_ns();
        req->seq.store(seq, std::memory_order_release);
        xh::wake_if_parked(&req->seq, &req->helper_sleeping);
        if (!xh::wait_until(&res->seq, &res->engine_sleeping, o.spin_us, t0 + 2000000000ull,
                            [seq] (uint32_t v) { return v == seq; })) {
            std::fprintf(stderr, "drive: request %d was never answered%s%s\n", r, hdr->error[0] ? ": " : "", hdr->error);
            ++bad;
            break;
        }
        us.push_back((double) (xh::monotonic_ns() - t0) / 1000.0);
        if (res->status != xh::kStatusOk || res->n_rows != picked) {
            std::fprintf(stderr, "drive: request %d answered status %d n_rows %d (asked %d)%s%s\n", r, res->status,
                         res->n_rows, picked, hdr->error[0] ? ": " : "", hdr->error);
            ++bad;
            break;
        }
        if (o.verbose && (r % 100) == 0) std::printf("drive: %d/%d requests, last %.1f us\n", r, o.requests, us.back());
    }
    if (!us.empty())
        std::printf("drive: %zu requests answered, round trip p50 %.1f us  p99 %.1f us  p99.9 %.1f us\n", us.size(),
                    pct(us, 0.50), pct(us, 0.99), pct(us, 0.999));

    hdr->engine_state.store((uint32_t) xh::EngineState::Shutdown, std::memory_order_release);
    // The helper notices Shutdown on its own 250 ms poll; give it a second before insisting.
    int status = 0;
    pid_t done = 0;
    for (int i = 0; i < 40 && !done; ++i) {
        done = waitpid(child, &status, WNOHANG);
        if (!done) std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!done) {
        std::fprintf(stderr, "drive: the helper did not stop on Shutdown; sending SIGTERM\n");
        kill(child, SIGTERM);
        waitpid(child, &status, 0);
    }
    const int code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    std::printf("drive: helper exit %d%s, helper_state %u\n", code,
                WIFSIGNALED(status) ? " (killed by a signal)" : "", hdr->helper_state.load());
    const bool pass = !us.empty() && bad == 0 && code == 0 && hdr->helper_state.load() == (uint32_t) xh::HelperState::Exited;
    std::printf("drive: %s\n", pass ? "PASS" : "FAIL");
    munmap(base, xh::kMappingBytes);
    close(fd);
    return pass ? 0 : 1;
}
