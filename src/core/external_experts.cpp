// src/core/external_experts.cpp - see include/strata/core/external_experts.hpp.
//
// The classification in begin() is RemoteExperts::begin's, line for line, with "this tier's cache holds it" replaced by
// the helper's loaded list.  What differs is where the rows are computed: in another process, on another vendor's GPU,
// with the request and the answer in one shared mapping (include/strata/xhelper/protocol.hpp).
#include "strata/core/external_experts.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>

#if defined(__linux__) && !defined(STRATA_USE_HIP)
#define STRATA_EXTERNAL_EXPERTS 1
#include "strata/xhelper/protocol.hpp"

#include <csignal>
#include <spawn.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

extern char** environ;
#else
#define STRATA_EXTERNAL_EXPERTS 0
#endif

namespace strata::core {

ExternalExperts::~ExternalExperts() { close(); }

#if !STRATA_EXTERNAL_EXPERTS

bool ExternalExperts::open(const Options&, const std::string&, const std::string&, int64_t, int64_t,
                           const std::vector<std::pair<int32_t, int32_t>>&, const ExpertCache&, std::string& err) {
#if defined(STRATA_USE_HIP)
    err = "the external expert helper needs the CUDA engine (this is the HIP build)";
#else
    err = "the external expert helper runs on Linux only";
#endif
    return false;
}
void ExternalExperts::close() {}
bool ExternalExperts::begin(int64_t, const float*, const int32_t*, int64_t, int64_t, const int32_t*, const int32_t*,
                            std::string&) { return true; }
bool ExternalExperts::finish(float*, std::string&) { return true; }
bool ExternalExperts::take_round_trips(double&, double&, int64_t&) { return false; }
void ExternalExperts::fail(const std::string&) {}

#else

namespace xh = strata::xhelper;

namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;
static_assert(xh::kH == H, "the protocol's row width is the kernels' H");
static_assert(xh::kMaxTok == strata::kernels::cpu::MAXT, "the protocol's window is the CPU kernels' MAXT");
static_assert(xh::kCap == CAP, "the protocol's request width is the remote tier's CAP");

double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    return v[(size_t) (p * (double) (v.size() - 1) + 0.5)];
}
}  // namespace

bool ExternalExperts::open(const Options& opt, const std::string& pack, const std::string& native_preset,
                           int64_t layers, int64_t experts, const std::vector<std::pair<int32_t, int32_t>>& ranked,
                           const ExpertCache& primary, std::string& err) {
    close();
    opt_ = opt;
    inclusive_ = opt.inclusive_ram;
    n_layers_ = layers;
    n_expert_ = experts;
    if (!xh::geometry_ok((uint32_t) layers, (uint32_t) experts)) { err = "external experts: the model's geometry does not fit the protocol"; return false; }
    if (access(opt.helper_exe.c_str(), X_OK) != 0) {
        err = "external experts: '" + opt.helper_exe + "' is not an executable (--external-experts takes the path of strata-expert-helper)";
        return false;
    }

    fd_ = (int) syscall(SYS_memfd_create, "strata-xhelper", 0);   // no MFD_CLOEXEC: the helper inherits it
    if (fd_ < 0) { err = std::string("external experts: memfd_create: ") + std::strerror(errno); return false; }
    if (ftruncate(fd_, (off_t) xh::kMappingBytes) != 0) {
        err = std::string("external experts: ftruncate: ") + std::strerror(errno);
        close();
        return false;
    }
    base_ = mmap(nullptr, xh::kMappingBytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (base_ == MAP_FAILED) {
        base_ = nullptr;
        err = std::string("external experts: mmap: ") + std::strerror(errno);
        close();
        return false;
    }
    hdr_ = reinterpret_cast<xh::Header*> (base_);
    if (!xh::init_header(*hdr_, (uint32_t) layers, (uint32_t) experts, xh::kMappingBytes, pack.c_str(),
                         native_preset.c_str())) {
        err = "external experts: the pack path or the geometry does not fit the shared header";
        close();
        return false;
    }
    req_ = xh::request_at(base_, hdr_->request_off);
    res_ = xh::response_at(base_, hdr_->response_off);

    // The offer: the ranked pairs the primary does not hold, each once, in rank order.  The helper takes them in this
    // order while they fit its budget.
    xh::Pair* assigned = xh::pairs_at(base_, hdr_->assigned_off);
    std::vector<uint8_t> offered((size_t) layers * (size_t) experts, 0);
    uint32_t n = 0;
    for (const auto& pair : ranked) {
        if (pair.first < 0 || pair.first >= layers || pair.second < 0 || pair.second >= experts) continue;
        const size_t index = (size_t) pair.first * (size_t) experts + (size_t) pair.second;
        if (offered[index] || primary.slot_of(pair.first, pair.second) >= 0) continue;
        offered[index] = 1;
        assigned[n++] = xh::Pair{ pair.first, pair.second };
    }
    if (n == 0) { err = "external experts: every expert is already on the primary GPU; nothing to offer the helper"; close(); return false; }
    hdr_->n_assigned = n;
    hdr_->vram_budget_bytes = opt.vram_budget;
    hdr_->engine_state.store((uint32_t) xh::EngineState::Assigned, std::memory_order_release);

    // The helper sees only its AMD card: HIP_VISIBLE_DEVICES picks it and an empty CUDA_VISIBLE_DEVICES keeps any CUDA
    // runtime it links from opening the RTX.
    std::vector<std::string> env_s;
    for (char** e = environ; e && *e; ++e)
        if (std::strncmp(*e, "HIP_VISIBLE_DEVICES=", 20) != 0 && std::strncmp(*e, "CUDA_VISIBLE_DEVICES=", 21) != 0 &&
            std::strncmp(*e, "ROCR_VISIBLE_DEVICES=", 21) != 0)
            env_s.emplace_back(*e);
    env_s.push_back("HIP_VISIBLE_DEVICES=" + std::to_string(opt.hip_device));
    env_s.push_back("CUDA_VISIBLE_DEVICES=");
    std::vector<char*> envp;
    for (auto& s : env_s) envp.push_back(s.data());
    envp.push_back(nullptr);
    std::string fd_s = std::to_string(fd_), spin_s = std::to_string(opt.spin_us);
    std::vector<char*> argv = { (char*) opt.helper_exe.c_str(), (char*) "--shm-fd", fd_s.data(), (char*) "--spin-us",
                                spin_s.data() };
    if (opt.trace) argv.push_back((char*) "--verbose");   // its load lines and a count every 512 requests
    argv.push_back(nullptr);
    pid_t pid = -1;
    const int rc = posix_spawn(&pid, opt.helper_exe.c_str(), nullptr, nullptr, argv.data(), envp.data());
    if (rc != 0) {
        err = "external experts: starting '" + opt.helper_exe + "': " + std::strerror(rc);
        close();
        return false;
    }
    pid_ = pid;
    hdr_->engine_state.store((uint32_t) xh::EngineState::Running, std::memory_order_release);

    const auto t0 = std::chrono::steady_clock::now();
    auto seconds = [&] { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
    double next_note = 10.0;
    for (;;) {
        const uint32_t state = hdr_->helper_state.load(std::memory_order_acquire);
        if (state == (uint32_t) xh::HelperState::Ready) break;
        int status = 0;
        const bool exited = waitpid(pid_, &status, WNOHANG) == pid_;
        if (state == (uint32_t) xh::HelperState::Failed || exited) {
            err = "external experts: the helper " +
                  std::string(exited ? "exited (code " + std::to_string(WIFEXITED(status) ? WEXITSTATUS(status) : -1) + ")"
                                     : "failed") +
                  " before it was ready" + (hdr_->error[0] ? std::string(": ") + hdr_->error : std::string());
            if (exited) pid_ = -1;
            close();
            return false;
        }
        if (seconds() > (double) opt.ready_timeout_s) {
            err = "external experts: the helper was not ready after " + std::to_string(opt.ready_timeout_s) + " s";
            close();
            return false;
        }
        if (seconds() >= next_note) {
            std::fprintf(stderr, "strata generate: external helper: still loading after %.0f s\n", seconds());
            next_note += 10.0;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    load_s_ = seconds();

    const auto& lay = strata::kernels::cpu::expert_layout();
    const xh::Pair* loaded = xh::pairs_at(base_, hdr_->loaded_off);
    held_.assign((size_t) layers * (size_t) experts, 0);
    layers_present_.assign((size_t) layers, 0);
    loaded_.clear();
    loaded_bytes_ = 0;
    for (uint32_t i = 0; i < hdr_->n_loaded; ++i) {
        const xh::Pair p = loaded[i];
        if (!xh::pair_ok(*hdr_, p.layer, p.expert)) continue;
        held_[(size_t) p.layer * (size_t) experts + (size_t) p.expert] = 1;
        layers_present_[(size_t) p.layer] = 1;
        loaded_.emplace_back(p.layer, p.expert);
        loaded_bytes_ += (uint64_t) lay.blob_bytes(p.layer);
    }
    if (loaded_.empty()) { err = "external experts: the helper is ready but loaded nothing"; close(); return false; }
    owned_.assign((size_t) CAP, 0);
    alive_ = true;
    valid_ = true;
    seq_ = 0;
    return true;
}

void ExternalExperts::close() {
    alive_ = false;
    valid_ = false;
    pending_ = false;
    if (hdr_ != nullptr) {
        hdr_->engine_state.store((uint32_t) xh::EngineState::Shutdown, std::memory_order_release);
        xh::wake_always(&req_->seq);
    }
    if (pid_ > 0) {
        // The helper notices Shutdown within its 250 ms idle poll; two seconds, then it is not asked again.
        int status = 0;
        pid_t done = 0;
        for (int i = 0; i < 40 && done == 0; ++i) {
            done = waitpid(pid_, &status, WNOHANG);
            if (done == 0) std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (done == 0) {
            kill(pid_, SIGKILL);
            waitpid(pid_, &status, 0);
        }
        pid_ = -1;
    }
    if (base_ != nullptr) munmap(base_, xh::kMappingBytes);
    if (fd_ >= 0) ::close(fd_);
    base_ = nullptr;
    fd_ = -1;
    hdr_ = nullptr;
    req_ = nullptr;
    res_ = nullptr;
}

void ExternalExperts::fail(const std::string& why) {
    if (!alive_) return;
    alive_ = false;
    std::fprintf(stderr, "strata: external experts: the helper stopped (%s); its experts are read from the model files "
                         "from now on\n", why.c_str());
}

bool ExternalExperts::begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
                            const int32_t* kind, const int32_t* primary_res, std::string& err) {
    struct Timer { double& acc; std::chrono::steady_clock::time_point t0 = std::chrono::steady_clock::now();
        ~Timer() { acc += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); } } timer{ms_begin_};
    const int64_t n = n_tok * k;
    pending_ = false;
    if (n <= 0 || n > CAP || n_tok > strata::kernels::cpu::MAXT || k != xh::kK || layer < 0 || layer >= n_layers_) {
        err = "external experts: invalid layer, routing width or window size";
        return false;
    }
    std::fill(owned_.begin(), owned_.begin() + n, 0);
    if (!alive_ || !layers_present_[(size_t) layer]) return true;
    int32_t m = 0;
    for (int64_t i = 0; i < n; ++i) {
        const int32_t e = ids[i];
        if ((kind && kind[i] != -1) || e < 0 || e >= n_expert_ ||
            (primary_res && primary_res[(size_t) layer * (size_t) n_expert_ + (size_t) e] >= 0)) continue;
        if (!held_[(size_t) layer * (size_t) n_expert_ + (size_t) e]) continue;
        owned_[(size_t) i] = 1;
        req_->entry[m] = (int32_t) i;
        req_->expert[m] = e;
        ++m;
    }
    if (m == 0) return true;
    req_->layer = (int32_t) layer;
    req_->n_tok = (int32_t) n_tok;
    req_->n_entries = m;
    std::memcpy(req_->x, x, (size_t) n_tok * H * sizeof(float));
    ++seq_;
    sent_ns_ = xh::monotonic_ns();
    req_->seq.store(seq_, std::memory_order_release);
    xh::wake_if_parked(&req_->seq, &req_->helper_sleeping);
    pending_ = true;
    pending_layer_ = layer;
    pending_entries_ = m;
    computed_ += m;
    ++launched_layers_;
    return true;
}

bool ExternalExperts::finish(float* out, std::string& err) {
    if (!pending_) return true;
    pending_ = false;
    const auto w0 = std::chrono::steady_clock::now();
    const uint32_t want = seq_;
    if (res_->seq.load(std::memory_order_acquire) != want) ++waited_layers_;
    const uint64_t deadline = xh::monotonic_ns() + (uint64_t) opt_.timeout_ms * 1000000ull;
    bool answered = false;
    // In slices of 100 ms, so a helper that died (SIGKILL leaves no answer and no flag) is noticed before the deadline.
    while (!answered) {
        const uint64_t now = xh::monotonic_ns();
        if (now >= deadline) break;
        answered = xh::wait_until(&res_->seq, &res_->engine_sleeping, opt_.spin_us, std::min<uint64_t>(deadline, now + 100000000ull),
                                  [want] (uint32_t v) { return v == want; });
        if (!answered && pid_ > 0) {
            int status = 0;
            if (waitpid(pid_, &status, WNOHANG) == pid_) { pid_ = -1; break; }
        }
    }
    const uint64_t seen_ns = xh::monotonic_ns();
    ms_wait_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    const std::string at = " (layer " + std::to_string(pending_layer_) + ")";
    if (!answered) {
        ++timeouts_;
        const std::string why = pid_ < 0 ? "the helper process exited" : "no answer within " + std::to_string(opt_.timeout_ms) + " ms";
        fail(why);
        err = "external experts: " + why + at;
        return false;
    }
    if (res_->status != xh::kStatusOk || res_->n_rows != pending_entries_) {
        const std::string why = std::string("the helper answered with an error") + (hdr_->error[0] ? std::string(": ") + hdr_->error : std::string());
        fail(why);
        err = "external experts: " + why + at;
        return false;
    }
    for (int32_t r = 0; r < res_->n_rows; ++r)
        if (res_->entry[r] < 0 || res_->entry[r] >= (int32_t) CAP || !owned_[(size_t) res_->entry[r]]) {
            fail("the helper answered a row this layer did not ask for");
            err = "external experts: the helper answered a row this layer did not ask for" + at;
            return false;
        }
    if (opt_.trace) round_trip_us_.push_back((double) (seen_ns - sent_ns_) / 1000.0);
    for (int32_t r = 0; r < res_->n_rows; ++r)
        std::memcpy(out + (size_t) res_->entry[r] * H, res_->rows + (size_t) r * H, (size_t) H * sizeof(float));
    return true;
}

bool ExternalExperts::take_round_trips(double& p50_us, double& p99_us, int64_t& count) {
    count = (int64_t) round_trip_us_.size();
    if (round_trip_us_.empty()) return false;
    p50_us = pct(round_trip_us_, 0.50);
    p99_us = pct(round_trip_us_, 0.99);
    round_trip_us_.clear();
    return true;
}

#endif

}  // namespace strata::core
