#pragma once

#include "strata/core/expert_cache.hpp"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::xhelper {
struct Header;
struct Request;
struct Response;
}  // namespace strata::xhelper

namespace strata::core {

/// docs/plans/OPTION_A_EXCLUSIVE_RAM.md Phase 4: the engine's half of the cross-vendor helper.  It spawns
/// `strata-expert-helper` (a HIP build, on an AMD card next to this CUDA engine), offers it the ranked pairs the
/// primary cache does not hold, and per layer sends it the CPU candidates it holds through one shared mapping (memfd)
/// with two futex words.  The interface mirrors RemoteExperts: begin() claims rows, the CPU pool skips the rows
/// owns() names, finish() waits and scatters the helper's rows into `out`.
///
/// Linux and the CUDA engine only: elsewhere open() fails with a message and nothing else is reachable.
class ExternalExperts {
public:
    struct Options {
        std::string helper_exe;      ///< path to strata-expert-helper
        int hip_device = 0;          ///< HIP_VISIBLE_DEVICES for the helper
        uint64_t vram_budget = 0;    ///< bytes; 0 = the helper's free VRAM minus its headroom
        bool inclusive_ram = false;  ///< keep the helper's experts in the RAM copy too
        uint32_t spin_us = 200;      ///< STRATA_XH_SPIN_US
        int timeout_ms = 2000;       ///< STRATA_XH_TIMEOUT_MS: one layer's answer, before the CPU computes that layer
        int give_up_ms = 60000;      ///< STRATA_XH_GIVE_UP_MS: no answer at all for this long, and the helper counts as gone
        int ready_timeout_s = 120;   ///< loading the experts into the card
        bool trace = false;          ///< STRATA_XH_TRACE=1: per-request round-trip percentiles
    };

    ExternalExperts() = default;
    ~ExternalExperts();
    ExternalExperts(const ExternalExperts&) = delete;
    ExternalExperts& operator=(const ExternalExperts&) = delete;

    /// Spawns the helper and waits for Ready.  `ranked` is hottest first; pairs `primary` holds are not offered.
    bool open(const Options& opt, const std::string& pack, const std::string& native_preset, int64_t layers,
              int64_t experts, const std::vector<std::pair<int32_t, int32_t>>& ranked, const ExpertCache& primary,
              std::string& err);
    /// Shutdown, a bounded wait for the helper, then SIGKILL.  Safe to call twice.
    void close();

    /// As RemoteExperts::begin: `kind` is the primary's classification (-1 = CPU candidate) or null on the
    /// one-token path; `primary_res` the primary's residency table.  Writes one request when any row is ours.
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k, const int32_t* kind,
               const int32_t* primary_res, std::string& err);
    bool owns(int64_t index) const { return owned_[(size_t) index] != 0; }
    /// The helper loaded (layer, expert), and it is still answering.
    bool holds(int64_t layer, int32_t expert) const {
        return alive_ && layer >= 0 && layer < n_layers_ && expert >= 0 && expert < n_expert_ &&
               held_[(size_t) layer * (size_t) n_expert_ + (size_t) expert] != 0;
    }
    /// Waits for this layer's answer and copies each row to `out + entry * H`.  It returns false when the rows are not
    /// written; the caller computes the owned rows itself.  A timeout with the helper's process still there is
    /// transient: begin() owns nothing until the late answer arrives, then the helper is used again (after
    /// `give_up_ms` without it, it counts as gone).  An error answer or a dead helper stop it for good: from then on
    /// begin() owns nothing.
    bool finish(float* out, std::string& err);

    bool valid() const { return valid_; }
    /// The helper's process id while it runs (for a test that kills it), else -1.
    int pid() const { return pid_; }
    bool alive() const { return alive_; }
    bool inclusive_ram() const { return inclusive_; }
    const std::vector<std::pair<int32_t, int32_t>>& loaded() const { return loaded_; }
    /// The loaded pairs as a layer * n_expert + expert byte mask, fixed after open() (for the look-ahead).
    const uint8_t* held_mask() const { return held_.empty() ? nullptr : held_.data(); }
    double gib() const { return (double) loaded_bytes_ / 1073741824.0; }
    double load_seconds() const { return load_s_; }
    int64_t computed() const { return computed_; }
    int64_t launched_layers() const { return launched_layers_; }
    int64_t timeouts() const { return timeouts_; }
    /// Stalls: runs of timeouts, each ended by a late answer (or by the helper stopping).
    int64_t stalls() const { return stalls_; }
    /// A timed-out request is still unanswered.
    bool stalled() const { return stale_; }
    /// finish() calls that found the answer not there yet: the CPU pool finished its rows first.
    int64_t waited_layers() const { return waited_layers_; }
    double ms_begin() const { return ms_begin_; }
    double ms_wait() const { return ms_wait_; }
    /// STRATA_XH_TRACE: round trips since the last call (begin's publish to the answer seen), p50 and p99 in us.
    bool take_round_trips(double& p50_us, double& p99_us, int64_t& count);

private:
    void fail(const std::string& why);
    /// After a timeout: true once the late answer is there (the request area is free again).  Fails the helper when
    /// its process is gone or it has not answered for `give_up_ms`.
    bool settle_stale();

    Options opt_;
    int fd_ = -1;
    void* base_ = nullptr;
    int pid_ = -1;
    bool alive_ = false;
    bool valid_ = false;   ///< open() succeeded: stays true after the helper stops, so the engine keeps asking holds()
    bool inclusive_ = false;
    bool pending_ = false;
    bool stale_ = false;            ///< request seq_ timed out and has no answer yet
    uint64_t stale_since_ns_ = 0;   ///< when that request was sent
    uint32_t seq_ = 0;
    int64_t pending_layer_ = -1;
    int32_t pending_entries_ = 0;
    uint64_t sent_ns_ = 0;
    int64_t n_layers_ = 0, n_expert_ = 0;
    strata::xhelper::Header* hdr_ = nullptr;
    strata::xhelper::Request* req_ = nullptr;
    strata::xhelper::Response* res_ = nullptr;
    std::vector<uint8_t> held_;
    std::vector<uint8_t> owned_;
    std::vector<uint8_t> layers_present_;
    std::vector<std::pair<int32_t, int32_t>> loaded_;
    uint64_t loaded_bytes_ = 0;
    double load_s_ = 0;
    int64_t computed_ = 0, launched_layers_ = 0, timeouts_ = 0, waited_layers_ = 0, stalls_ = 0;
    double ms_begin_ = 0, ms_wait_ = 0;
    std::vector<double> round_trip_us_;
};

}  // namespace strata::core
