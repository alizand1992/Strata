// src/xhelper/helper_prompt.hpp - the expert helper's prompt products (docs/plans/OPTION_A_EXCLUSIVE_RAM.md Phase 7.1).
//
// Decode asks the helper for at most kMaxTok tokens' rows (HelperTier).  A prompt chunk routes hundreds to thousands
// of tokens to the experts this card holds; in exclusive mode the engine has no other copy of them than the model
// files, and before this it read them from there (the SSD) for every chunk.  Here the card computes them itself, with
// the prompt path's own kernels (src/prefill/moe_mmq.cu: llama.cpp's MMQ, built for HIP), and sends back one weighted
// sum per token instead of a row per routed pair - the --peer-device "sums" route (prefill.cpp, PeerPrefill::sums).
//
// Per request (one layer, up to max_tok tokens; see protocol.hpp's bulk layout):
//   x, w, pair -> the device; the sums zeroed
//   per group of up to 16 experts and at most G rows: gather the experts' blobs from their cache slots into one
//   buffer, q8_1 the group's rows of x, gate/up, swiglu, q8_1 of H, down, then each expert's rows weighted into the
//   sums (peer_scatter_add: one launch per expert, so a token is never added twice in one launch)
//   the sums -> the mapping
#pragma once

#include "helper_tier.hpp"

#include "strata/prefill/moe_mmq.hpp"
#include "strata/xhelper/protocol.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::xhelper {

class HelperPrompt {
public:
    HelperPrompt() = default;
    ~HelperPrompt();
    HelperPrompt(const HelperPrompt&) = delete;
    HelperPrompt& operator=(const HelperPrompt&) = delete;

    /// Device bytes open() allocates for requests of up to `max_tok` tokens (the helper takes them off its expert
    /// budget before it loads, so the prompt buffers never meet a full card).  0 when no layer can run them.
    static uint64_t bytes_needed(uint32_t max_tok);

    /// Allocate the buffers on `device` and mark in `layer_ok` (n_layers entries) the layers whose two formats the
    /// prompt products cover on this card.  `bulk` is the mapping's bulk region in this process.
    bool open(int device, uint32_t max_tok, int64_t n_layers, int64_t n_expert, void* bulk, uint64_t bulk_bytes,
              uint8_t* layer_ok, std::string& err);
    void close();

    /// One request: `n_tok` tokens of `layer`, `n_exp` experts (the bulk's expert/count arrays), `n_rows` rows.  The
    /// sums are in the bulk region when it returns true.  Nothing is published on false: err says why.
    bool run(const HelperTier& tier, int32_t layer, int32_t n_tok, int32_t n_exp, int32_t n_rows, std::string& err);

    bool ready() const { return device_ >= 0; }
    int64_t requests() const { return requests_; }
    int64_t rows() const { return rows_; }

private:
    int device_ = -1;
    uint32_t max_tok_ = 0;
    int64_t n_expert_ = 0, G_ = 0;
    BulkLayout lay_{};
    char* bulk_ = nullptr;
    uint64_t bulk_bytes_ = 0;
    bool registered_ = false;
    cudaStream_t s_ = nullptr;
    std::unique_ptr<strata::prefill::mmq::Context> ctx_;
    std::vector<void*> owned_;
    float *d_x_ = nullptr, *d_sum_ = nullptr, *d_w_ = nullptr, *d_gu_ = nullptr, *d_h_ = nullptr, *d_d_ = nullptr;
    int32_t *d_pair_ = nullptr, *d_src_ = nullptr, *d_bounds_ = nullptr, *d_ident_ = nullptr;
    void *d_xq_ = nullptr, *d_hq_ = nullptr;
    uint8_t *d_grp_gu_ = nullptr, *d_grp_d_ = nullptr;
    size_t gub_max_ = 0, db_max_ = 0;
    std::vector<int32_t> src_, bounds_;
    int64_t requests_ = 0, rows_ = 0;
};

}  // namespace strata::xhelper
