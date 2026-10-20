// src/xhelper/helper_tier.hpp - the expert helper's device side.
//
// docs/plans/OPTION_A_EXCLUSIVE_RAM.md Phase 3.  It is `strata::core::RemoteExperts` (src/core/remote_experts.cpp)
// adapted for a second vendor's card:
//   - this process exists to serve ONE device: device 0 of the helper, which is the whole AMD card the engine assigned
//     it with HIP_VISIBLE_DEVICES.  RemoteExperts refuses device 0 because in the engine device 0 is the primary.
//   - the pair list is handed to it (the engine's offer, hottest first, from the shared header) instead of being cut
//     out of an expert profile against the primary cache's claims.  It keeps the prefix that fits its own VRAM.
//   - there is no classification step: the engine already decided which routed entries belong to this helper and says
//     so with (entry, expert) pairs.  An entry the helper does not hold is an error, not a skip - a silently missing
//     row is exactly the kind of wrong answer the plan's Gate 5 is there to catch.
//   - no RemoteExpertOpt reduce path, and no lending: the helper's slots are its own.
//
// The kernels are the same ones the remote tier launches (quantize_q8_1_rows + native_expert_grouped, or
// quantize_q8_0_scaled + moe_grouped_s2), the same metadata upload, one non-blocking stream.
//
// CUDA API names throughout, as everywhere in this repository: the HIP build maps them through
// include/strata/hip_compat/cuda_runtime.h, and this file is only built when STRATA_ENABLE_HIP is ON.
#pragma once

#include "strata/core/expert_cache.hpp"
#include "strata/core/expert_source.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace strata::xhelper {

class HelperTier {
public:
    HelperTier() = default;
    ~HelperTier();
    HelperTier(const HelperTier&) = delete;
    HelperTier& operator=(const HelperTier&) = delete;

    /// Free and total VRAM on `device`, before anything is allocated.  A helper that cannot see its card has no
    /// reason to load a model, so the engine is told before the pack is opened.
    static bool memory_info(int device, uint64_t& free_bytes, uint64_t& total_bytes, std::string& err);

    /// Load the prefix of `assigned` this card can hold (`reserve_bytes` less: Phase 7.1's prompt buffers).  `loaded` comes back with the pairs that are resident, in
    /// the order they were taken; the helper publishes those and only those.
    ///
    /// `shared_x` and `shared_rows` are the Request::x and Response::rows regions of the engine's mapping as this
    /// process sees them.  When the driver accepts registering pages another process mapped (Phase 1 of the plan is
    /// the measurement of exactly that) the kernels read the activations straight out of the mapping and write their
    /// rows straight into it.  When it refuses, private pinned staging and one copy each way per layer.
    bool open(int device, uint64_t vram_budget_bytes, int64_t n_layers, int64_t n_expert,
              const std::vector<std::pair<int32_t, int32_t>>& assigned, strata::core::ExpertSource& source,
              void* shared_x, uint64_t shared_x_bytes, void* shared_rows, uint64_t shared_rows_bytes,
              std::vector<std::pair<int32_t, int32_t>>& loaded, std::string& err, uint64_t reserve_bytes = 0);
    void close();

    /// One request: `n_entries` routed entries of `layer`, entry j belonging to expert[j], activations for n_tok
    /// tokens.  Rows are written compactly to `rows` and out_entry[r] says which routed entry row r belongs to.
    /// On failure nothing is written and err says why: the plan's rule is never a partial answer with status ok.
    bool run(int32_t layer, const float* x, int32_t n_tok, const int32_t* entry, const int32_t* expert,
             int32_t n_entries, float* rows, int32_t* out_entry, int32_t& n_rows, std::string& err);

    int64_t loaded() const { return (int64_t) loaded_.size(); }
    int64_t computed() const { return computed_; }
    int64_t requests() const { return requests_; }
    bool holds(int32_t layer, int32_t expert) const { return cache_.slot_of(layer, expert) >= 0; }
    /// The device address of (layer, expert)'s blob in this card's cache, or null when it is not held.
    const uint8_t* slot_ptr(int32_t layer, int32_t expert) const {
        const int32_t s = cache_.slot_of(layer, expert);
        return s < 0 ? nullptr : cache_.device_slot(s);
    }
    double gib() const { return cache_.gib(); }
    uint64_t vram_free_bytes() const { return vram_free_; }
    uint64_t vram_total_bytes() const { return vram_total_; }
    uint64_t budget_bytes() const { return budget_; }
    bool zero_copy() const { return zero_copy_; }

private:
    int device_ = -1;
    int64_t n_expert_ = 0;
    int64_t computed_ = 0;
    int64_t requests_ = 0;
    uint64_t vram_free_ = 0, vram_total_ = 0, budget_ = 0;
    strata::core::ExpertCache cache_;
    std::vector<std::pair<int32_t, int32_t>> loaded_;
    std::vector<uint8_t> layers_present_;

    cudaStream_t stream_ = nullptr;
    void* h_meta_ = nullptr;              ///< the one metadata upload, pinned here
    void* d_meta_ = nullptr;              ///< ...and its device image, one contiguous copy per request
    float* d_x_ = nullptr;                ///< staging input, when the shared mapping cannot be registered
    float* d_out_ = nullptr;              ///< staging output
    float* h_out_ = nullptr;              ///< pinned host output for the staged path
    float* shared_x_ = nullptr;           ///< Request::x in this process's mapping
    float* shared_rows_ = nullptr;        ///< Response::rows in this process's mapping
    float* z_x_ = nullptr;                ///< Request::x as this GPU sees it
    float* z_rows_ = nullptr;             ///< Response::rows as this GPU sees it
    bool zero_copy_ = false;
    bool registered_x_ = false, registered_rows_ = false;
    uint8_t* d_q8_ = nullptr;
    float* d_scales_ = nullptr;
    void* d_scratch_ = nullptr;
    int32_t* d_start_ = nullptr;
    int32_t* d_dst_ = nullptr;
    int32_t* d_tok_ = nullptr;
    int32_t* d_count_ = nullptr;
    unsigned long long* d_ptr_ = nullptr;

    std::vector<int32_t> group_of_, group_id_;
    std::vector<int32_t> start_, dst_, tok_, entry_of_row_;
    std::vector<unsigned long long> ptr_;
};

}  // namespace strata::xhelper
