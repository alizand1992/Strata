// src/xhelper/helper_prompt.cpp - see src/xhelper/helper_prompt.hpp.
//
// The products are the prompt path's MMQ group (src/prefill/prefill.cpp, `compute`'s group branch) with the group's
// rows quantized on their own instead of a layer-wide Xq: the same gather, the same two products and swiglu, the same
// q8_1 rounding of the input and of H.  The weighting and the per-token sum are PeerPrefill::sums' (peer_scatter_add).
#include "helper_prompt.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <algorithm>
#include <cstring>

namespace strata::xhelper {
namespace {
namespace mmq = strata::prefill::mmq;
constexpr int64_t N = kH;
constexpr int64_t FF2 = 1280, FF = 640;   // gate|up rows, n_ff (the prefill path's fixed shapes)
constexpr int MMQ_GROUP = 16;             // prefill.cpp's MMQ_GROUP: experts per launch
constexpr size_t MMQ_TAIL = 4096;         // prefill.cpp's MMQ_TAIL: zeroed after a group's last expert

inline void ignore(cudaError_t) {}

bool check(cudaError_t result, const char* what, std::string& err) {
    if (result == cudaSuccess) return true;
    err = std::string("helper HIP prompt: ") + what + ": " + cudaGetErrorString(result);
    return false;
}

// A layer the prompt products take: MMQ has both formats on this card, and no NVFP4 tail (its scales need the
// prompt path's tail buffers, which this does not carry).
bool layer_covered(const strata::kernels::cpu::NativeFmt& f) {
    return mmq::built() && f.tail_off == 0 && f.n_embd == N && f.n_ff == FF && mmq::fits(f.gu_type, FF2) &&
           mmq::fits(f.d_type, N) && !mmq::fp4_activations(f.gu_type) && !mmq::fp4_activations(f.d_type);
}

void max_matrices(size_t& gub, size_t& db) {
    gub = db = 0;
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || !mmq::built()) return;
    for (const auto& f : lay.fmt) {
        if (!mmq::supported(f.gu_type) || !mmq::supported(f.d_type)) continue;
        gub = std::max(gub, mmq::matrix_bytes(f.gu_type, FF2, N));
        db = std::max(db, mmq::matrix_bytes(f.d_type, N, FF));
    }
}
}  // namespace

HelperPrompt::~HelperPrompt() { close(); }

uint64_t HelperPrompt::bytes_needed(uint32_t max_tok) {
    size_t gub = 0, db = 0;
    max_matrices(gub, db);
    if (gub == 0 || db == 0 || max_tok == 0) return 0;
    const uint64_t T = max_tok, G = max_tok, E = T * (uint64_t) kK;
    uint64_t b = 0;
    b += 2 * T * N * sizeof(float);                              // x, sums
    b += E * sizeof(float) + 2 * E * sizeof(int32_t);            // w, pair, src
    b += ((uint64_t) kMaxExpert + kMaxExpert / MMQ_GROUP + 2) * (MMQ_GROUP + 1) * sizeof(int32_t);   // bounds
    b += G * sizeof(int32_t);                                    // the identity row map
    b += mmq::q8_bytes((int64_t) G, N) + mmq::q8_bytes((int64_t) G, FF);
    b += G * (FF2 + FF + N) * sizeof(float);                     // GU, H, D
    b += (uint64_t) MMQ_GROUP * (gub + db) + 2 * MMQ_TAIL;       // the group's gathered weights
    return b + (64ull << 20);                                    // MMQ's stream-k scratch and the allocator's rounding
}

void HelperPrompt::close() {
    if (device_ < 0) return;
    int prev = -1;
    ignore(cudaGetDevice(&prev));
    ignore(cudaSetDevice(device_));
    if (s_) ignore(cudaStreamSynchronize(s_));
    ctx_.reset();
    for (void* p : owned_) ignore(cudaFree(p));
    owned_.clear();
    if (registered_ && bulk_) ignore(cudaHostUnregister(bulk_));
    if (s_) ignore(cudaStreamDestroy(s_));
    if (prev >= 0) ignore(cudaSetDevice(prev));
    s_ = nullptr;
    registered_ = false;
    bulk_ = nullptr;
    device_ = -1;
}

bool HelperPrompt::open(int device, uint32_t max_tok, int64_t n_layers, int64_t n_expert, void* bulk,
                        uint64_t bulk_bytes, uint8_t* layer_ok, std::string& err) {
    close();
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native || (int64_t) lay.fmt.size() < n_layers) { err = "helper HIP prompt: needs a native pack's layout"; return false; }
    if (max_tok == 0 || max_tok > kMaxPromptTok || n_expert <= 0 || n_expert > (int64_t) kMaxExpert) {
        err = "helper HIP prompt: request size or geometry outside the protocol";
        return false;
    }
    int64_t covered = 0;
    for (int64_t l = 0; l < n_layers; ++l) {
        layer_ok[l] = layer_covered(lay.fmt[(size_t) l]) ? 1 : 0;
        covered += layer_ok[l];
    }
    if (covered == 0) { err = "helper HIP prompt: MMQ covers no layer's formats on this card"; return false; }
    max_matrices(gub_max_, db_max_);
    max_tok_ = max_tok;
    n_expert_ = n_expert;
    G_ = (int64_t) max_tok;   // an expert has at most one row per token, so its rows always fit one group
    lay_ = bulk_layout(max_tok, (uint32_t) n_expert);
    if (lay_.total_bytes > bulk_bytes) { err = "helper HIP prompt: the bulk region is smaller than its layout"; return false; }
    bulk_ = reinterpret_cast<char*> (bulk);
    bulk_bytes_ = bulk_bytes;
    if (!check(cudaSetDevice(device), "cudaSetDevice", err)) return false;
    device_ = device;
    // Registered, the copies to and from the mapping are DMA; refused, they are staged by the runtime (slower, the same)
    registered_ = cudaHostRegister(bulk_, bulk_bytes_, cudaHostRegisterPortable) == cudaSuccess;
    if (!registered_) ignore(cudaGetLastError());
    auto alloc = [&](void** p, size_t bytes, const char* what) {
        if (!check(cudaMalloc(p, bytes), what, err)) return false;
        owned_.push_back(*p);
        return true;
    };
    const size_t T = max_tok, G = (size_t) G_, E = T * kK;
    const size_t nb = ((size_t) kMaxExpert + kMaxExpert / MMQ_GROUP + 2) * (MMQ_GROUP + 1);
    const bool ok = check(cudaStreamCreateWithFlags(&s_, cudaStreamNonBlocking), "stream", err) &&
                    alloc((void**) &d_x_, T * N * sizeof(float), "input") &&
                    alloc((void**) &d_sum_, T * N * sizeof(float), "sums") &&
                    alloc((void**) &d_w_, E * sizeof(float), "weights") &&
                    alloc((void**) &d_pair_, E * sizeof(int32_t), "pairs") &&
                    alloc((void**) &d_src_, E * sizeof(int32_t), "row tokens") &&
                    alloc((void**) &d_bounds_, nb * sizeof(int32_t), "bounds") &&
                    alloc((void**) &d_ident_, G * sizeof(int32_t), "row map") &&
                    alloc(&d_xq_, mmq::q8_bytes((int64_t) G, N), "q8 input") &&
                    alloc(&d_hq_, mmq::q8_bytes((int64_t) G, FF), "q8 hidden") &&
                    alloc((void**) &d_gu_, G * FF2 * sizeof(float), "gate/up") &&
                    alloc((void**) &d_h_, G * FF * sizeof(float), "hidden") &&
                    alloc((void**) &d_d_, G * N * sizeof(float), "down") &&
                    alloc((void**) &d_grp_gu_, MMQ_GROUP * gub_max_ + MMQ_TAIL, "group gate/up") &&
                    alloc((void**) &d_grp_d_, MMQ_GROUP * db_max_ + MMQ_TAIL, "group down");
    if (!ok) { close(); return false; }
    ctx_ = std::make_unique<mmq::Context>();
    mmq::iota(d_ident_, (int64_t) G, s_);
    if (!check(cudaStreamSynchronize(s_), "setup", err)) { close(); return false; }
    src_.reserve(E);
    bounds_.reserve(nb);
    return true;
}

bool HelperPrompt::run(const HelperTier& tier, int32_t layer, int32_t n_tok, int32_t n_exp, int32_t n_rows,
                       std::string& err) {
    if (device_ < 0) { err = "helper HIP prompt: the prompt kind is off"; return false; }
    const auto& lay = strata::kernels::cpu::expert_layout();
    if (layer < 0 || (size_t) layer >= lay.fmt.size() || n_tok <= 0 || (uint32_t) n_tok > max_tok_ || n_exp <= 0 ||
        n_exp > n_expert_ || n_rows <= 0 || n_rows > n_tok * kK) {
        err = "helper HIP prompt: a request outside the model or the bulk region (layer " + std::to_string(layer) +
              ", tokens " + std::to_string(n_tok) + ", experts " + std::to_string(n_exp) + ", rows " +
              std::to_string(n_rows) + ")";
        return false;
    }
    const auto& f = lay.fmt[(size_t) layer];
    if (!layer_covered(f)) { err = "helper HIP prompt: layer " + std::to_string(layer) + " is not one it runs"; return false; }
    const float* x = reinterpret_cast<const float*> (bulk_ + lay_.x_off);
    float* sum = reinterpret_cast<float*> (bulk_ + lay_.sum_off);
    const float* w = reinterpret_cast<const float*> (bulk_ + lay_.w_off);
    const int32_t* pair = reinterpret_cast<const int32_t*> (bulk_ + lay_.pair_off);
    const int32_t* expert = reinterpret_cast<const int32_t*> (bulk_ + lay_.expert_off);
    const int32_t* count = reinterpret_cast<const int32_t*> (bulk_ + lay_.count_off);

    // the request checked before anything runs: every expert held, ascending, every row a pair of this request
    int64_t total = 0;
    std::vector<const uint8_t*> blob((size_t) n_exp);
    for (int32_t i = 0; i < n_exp; ++i) {
        const int32_t e = expert[i], c = count[i];
        if (e < 0 || e >= n_expert_ || (i > 0 && e <= expert[i - 1]) || c <= 0 || c > n_tok) {
            err = "helper HIP prompt: expert entry " + std::to_string(i) + " is outside the request";
            return false;
        }
        blob[(size_t) i] = tier.slot_ptr(layer, e);
        if (blob[(size_t) i] == nullptr) {
            err = "helper HIP prompt: the engine sent layer " + std::to_string(layer) + " expert " + std::to_string(e) +
                  ", which this helper does not hold";
            return false;
        }
        total += c;
    }
    if (total != n_rows) { err = "helper HIP prompt: the experts' row counts do not add up to the request's rows"; return false; }
    src_.resize((size_t) n_rows);
    for (int32_t r = 0; r < n_rows; ++r) {
        const int32_t p = pair[r];
        if (p < 0 || p >= n_tok * kK) { err = "helper HIP prompt: row " + std::to_string(r) + " names no pair of the request"; return false; }
        src_[(size_t) r] = p / kK;
    }

    // groups: up to MMQ_GROUP experts and at most G_ rows (each expert alone fits: at most n_tok <= G_ rows)
    struct Group { int32_t e0, e1, r0, r1, max_rows; size_t b; };
    std::vector<Group> groups;
    bounds_.clear();
    for (int32_t i = 0, r = 0; i < n_exp;) {
        Group g{ i, i, r, r, 0, bounds_.size() };
        bounds_.push_back(0);
        while (g.e1 < n_exp && g.e1 - g.e0 < MMQ_GROUP && (int64_t) (g.r1 - g.r0) + count[g.e1] <= G_) {
            g.r1 += count[g.e1];
            g.max_rows = std::max(g.max_rows, count[g.e1]);
            bounds_.push_back(g.r1 - g.r0);
            ++g.e1;
        }
        groups.push_back(g);
        i = g.e1;
        r = g.r1;
    }

    const cudaStream_t s = s_;
    const size_t gub = mmq::matrix_bytes(f.gu_type, FF2, N), db = mmq::matrix_bytes(f.d_type, N, FF);
    const bool up =
        check(cudaMemcpyAsync(d_x_, x, (size_t) n_tok * N * sizeof(float), cudaMemcpyHostToDevice, s), "copy input", err) &&
        check(cudaMemcpyAsync(d_w_, w, (size_t) n_tok * kK * sizeof(float), cudaMemcpyHostToDevice, s), "copy weights", err) &&
        check(cudaMemcpyAsync(d_pair_, pair, (size_t) n_rows * sizeof(int32_t), cudaMemcpyHostToDevice, s), "copy pairs", err) &&
        check(cudaMemcpyAsync(d_src_, src_.data(), (size_t) n_rows * sizeof(int32_t), cudaMemcpyHostToDevice, s), "copy row tokens", err) &&
        check(cudaMemcpyAsync(d_bounds_, bounds_.data(), bounds_.size() * sizeof(int32_t), cudaMemcpyHostToDevice, s), "copy bounds", err) &&
        check(cudaMemsetAsync(d_sum_, 0, (size_t) n_tok * N * sizeof(float), s), "clear sums", err);
    if (!up) return false;
    for (const Group& g : groups) {
        const int ng = g.e1 - g.e0;
        const int64_t nr = g.r1 - g.r0;
        mmq::GatherGroup gg;
        for (int q = 0; q < ng; ++q) gg.blob[q] = blob[(size_t) (g.e0 + q)];
        gg.n = ng;
        if (!mmq::gather_native_group(gg, f.up_off, gub / 2, f.down_off, db, d_grp_gu_, gub, d_grp_d_, db, s, 0, nullptr,
                                      MMQ_TAIL)) {
            for (int q = 0; q < ng; ++q) {   // not 16-byte aligned: one at a time
                const uint8_t* b = gg.blob[q];
                mmq::gather_native(b, b + f.up_off, gub / 2, b + f.down_off, db, d_grp_gu_ + (size_t) q * gub,
                                   d_grp_d_ + (size_t) q * db, s, nullptr, nullptr, q + 1 == ng ? MMQ_TAIL : 0);
            }
        }
        mmq::quantize(d_x_, d_src_ + g.r0, d_xq_, f.gu_type, N, N, nr, s);
        mmq::Product gu;
        gu.w = d_grp_gu_; gu.type = f.gu_type; gu.w_rows = FF2; gu.w_cols = N; gu.expert_bytes = gub;
        gu.n = ng; gu.xq = d_xq_; gu.bounds = d_bounds_ + g.b; gu.ids = d_ident_;
        gu.total_rows = nr; gu.max_rows = g.max_rows; gu.dst = d_gu_; gu.ld_dst = FF2;
        ctx_->run(gu, s);
        mmq::swiglu(d_gu_, d_h_, nr, FF, false, s);
        mmq::quantize(d_h_, nullptr, d_hq_, f.d_type, FF, FF, nr, s);
        mmq::Product dn;
        dn.w = d_grp_d_; dn.type = f.d_type; dn.w_rows = N; dn.w_cols = FF; dn.expert_bytes = db;
        dn.n = ng; dn.xq = d_hq_; dn.bounds = d_bounds_ + g.b; dn.ids = d_ident_;
        dn.total_rows = nr; dn.max_rows = g.max_rows; dn.dst = d_d_; dn.ld_dst = N;
        ctx_->run(dn, s);
        for (int32_t i = g.e0, r = g.r0; i < g.e1; r += count[i], ++i)
            strata::prefill::peer_scatter_add(d_sum_, d_d_ + (size_t) (r - g.r0) * N, d_w_, d_pair_ + r, count[i], s);
    }
    if (!check(cudaMemcpyAsync(sum, d_sum_, (size_t) n_tok * N * sizeof(float), cudaMemcpyDeviceToHost, s), "copy sums", err) ||
        !check(cudaStreamSynchronize(s), "request", err))
        return false;
    ++requests_;
    rows_ += n_rows;
    return true;
}

}  // namespace strata::xhelper
