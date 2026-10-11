// src/xhelper/helper_tier.cpp - see src/xhelper/helper_tier.hpp.
//
// Structurally this is RemoteExperts::open / begin / finish with the parts the helper does not have (a profile, a
// primary cache to compare against, a classification array, the reduce path) taken out and the pair list put in.  The
// staging sizes, the metadata upload, the kernel calls and the scratch sizing are the same on purpose: the parity
// test (tests/xhelper/helper_tier_parity.cpp) is the claim that the rows are the same rows.
#include "helper_tier.hpp"

#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_expert_grouped.hpp"
#include "strata/xhelper/protocol.hpp"

#include <algorithm>
#include <cstring>

namespace strata::xhelper {
namespace {
constexpr int64_t H = strata::kernels::cpu::H;
constexpr int64_t FF = strata::kernels::cpu::FF;
constexpr int64_t CAP = strata::kernels::cpu::MAXT * 10;   // = kCap, asserted below

static_assert(kCap == CAP, "the protocol's request width is this tier's CAP");
static_assert(kH == H, "the protocol's row width is the kernels' H");

// Same as RemoteMeta in src/core/remote_experts.cpp: one contiguous upload instead of five small copies.
struct HelperMeta {
    unsigned long long ptr[CAP];
    int32_t start[CAP + 1];
    int32_t dst[CAP];
    int32_t tok[CAP];
    int32_t count;
};

// HIP declares the whole runtime [[nodiscard]]; a cleanup path that cannot act on a failure says so.
inline void ignore(cudaError_t) {}

struct DeviceScope {
    int previous = -1;
    bool ok = false;
    cudaError_t status = cudaSuccess;
    const char* failed_step = nullptr;
    explicit DeviceScope(int device) : device_(device) {
        status = cudaGetDevice(&previous);
        if (status != cudaSuccess) { previous = -1; failed_step = "cudaGetDevice"; return; }
        status = cudaSetDevice(device);
        if (status != cudaSuccess) { failed_step = "cudaSetDevice"; return; }
        ok = true;
    }
    ~DeviceScope() { if (previous >= 0) ignore(cudaSetDevice(previous)); }
    std::string error() const {
        return "helper HIP: " + std::string(failed_step ? failed_step : "device switch") + " on device " +
               std::to_string(device_) + ": " + cudaGetErrorString(status);
    }
    int device_ = 0;   // only for the message
};


bool check(cudaError_t result, const char* what, std::string& err) {
    if (result == cudaSuccess) return true;
    err = std::string("helper HIP: ") + what + ": " + cudaGetErrorString(result);
    return false;
}
}  // namespace

HelperTier::~HelperTier() { close(); }

bool HelperTier::memory_info(int device, uint64_t& free_bytes, uint64_t& total_bytes, std::string& err) {
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err)) return false;
    if (device < 0 || device >= count) { err = "helper HIP: device " + std::to_string(device) + " is not visible"; return false; }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(); return false; }
    if (!check(cudaMemGetInfo(&free_bytes, &total_bytes), "cudaMemGetInfo", err)) return false;
    return true;
}

void HelperTier::close() {
    if (device_ < 0) return;
    DeviceScope scope(device_);
    if (scope.ok) {
        if (stream_) ignore(cudaStreamSynchronize(stream_));
        if (registered_x_ && shared_x_) ignore(cudaHostUnregister(shared_x_));
        if (registered_rows_ && shared_rows_) ignore(cudaHostUnregister(shared_rows_));
        cache_.close();
        if (d_x_) ignore(cudaFree(d_x_));
        if (d_out_) ignore(cudaFree(d_out_));
        if (d_q8_) ignore(cudaFree(d_q8_));
        if (d_scales_) ignore(cudaFree(d_scales_));
        if (d_scratch_) ignore(cudaFree(d_scratch_));
        if (d_meta_) ignore(cudaFree(d_meta_));
        if (h_out_) ignore(cudaFreeHost(h_out_));
        if (h_meta_) ignore(cudaFreeHost(h_meta_));
        if (stream_) ignore(cudaStreamDestroy(stream_));
    }
    device_ = -1;
    stream_ = nullptr;
    h_meta_ = d_meta_ = d_scratch_ = nullptr;
    h_out_ = d_x_ = d_out_ = z_x_ = z_rows_ = nullptr;
    d_q8_ = nullptr;
    d_scales_ = nullptr;
    d_start_ = d_dst_ = d_tok_ = d_count_ = d_ptr_ = nullptr;
    shared_x_ = shared_rows_ = nullptr;
    registered_x_ = registered_rows_ = false;
    zero_copy_ = false;
    loaded_.clear();
    layers_present_.clear();
}

bool HelperTier::open(int device, uint64_t vram_budget_bytes, int64_t n_layers, int64_t n_expert,
                      const std::vector<std::pair<int32_t, int32_t>>& assigned, strata::core::ExpertSource& source,
                      void* shared_x, uint64_t shared_x_bytes, void* shared_rows, uint64_t shared_rows_bytes,
                      std::vector<std::pair<int32_t, int32_t>>& loaded, std::string& err) {
    close();
    loaded.clear();
    if (n_layers <= 0 || n_expert <= 0 || assigned.empty()) {
        err = "helper HIP: need the model geometry and a non-empty pair list";
        return false;
    }
    int count = 0;
    if (!check(cudaGetDeviceCount(&count), "cudaGetDeviceCount", err)) return false;
    if (device < 0 || device >= count) { err = "helper HIP: device " + std::to_string(device) + " is not visible"; return false; }
    DeviceScope scope(device);
    if (!scope.ok) { err = scope.error(); return false; }
    device_ = device;
    n_expert_ = n_expert;
    shared_x_ = reinterpret_cast<float*> (shared_x);
    shared_rows_ = reinterpret_cast<float*> (shared_rows);

    const auto& lay = strata::kernels::cpu::expert_layout();
    if (!lay.native && lay.max_blob <= 0) { err = "helper HIP: the expert layout is not initialised"; close(); return false; }

    // The helper is the only process on this card, so its free VRAM is its budget.  vram_budget_bytes lets the
    // engine cap it; the 512 MiB headroom is the remote tier's, kept because a driver that runs out mid-request
    // fails a generation rather than a benchmark.
    if (!check(cudaMemGetInfo(&vram_free_, &vram_total_), "cudaMemGetInfo", err)) { close(); return false; }
    const uint64_t headroom = 512ull << 20;
    const uint64_t usable = vram_free_ > headroom ? vram_free_ - headroom : 0;
    budget_ = vram_budget_bytes ? std::min<uint64_t>(vram_budget_bytes, usable) : usable;
    if (budget_ < (uint64_t) lay.max_blob) {
        err = "helper HIP: " + std::to_string(vram_free_ >> 20) + " MiB free leaves no room for one " +
              std::to_string(lay.max_blob >> 20) + " MiB expert after the " + std::to_string(headroom >> 20) +
              " MiB headroom";
        close();
        return false;
    }

    // Take the offer in the order the engine ranked it, stop when the next expert does not fit.  This is the
    // exclusive-RAM rule: what is loaded here is reported back and the engine keeps no RAM copy of it.
    std::vector<uint8_t> picked((size_t) n_layers * (size_t) n_expert, 0);
    uint64_t needed = 0;
    for (const auto& pair : assigned) {
        if (pair.first < 0 || pair.first >= n_layers || pair.second < 0 || pair.second >= n_expert) continue;
        const size_t index = (size_t) pair.first * (size_t) n_expert + (size_t) pair.second;
        if (picked[index]) continue;
        const uint64_t bytes = lay.native ? (lay.blob_bytes(pair.first) + 255) / 256 * 256 : (uint64_t) lay.max_blob;
        if (needed + bytes > budget_) break;
        loaded.push_back(pair);
        needed += bytes;
        picked[index] = 1;
    }
    if (loaded.empty()) {
        err = "helper HIP: no offered expert fits in " + std::to_string(budget_ >> 20) + " MiB";
        close();
        return false;
    }

    std::vector<int64_t> sizes;
    if (lay.native) {
        sizes.reserve(loaded.size());
        for (const auto& pair : loaded) sizes.push_back((int64_t) lay.blob_bytes(pair.first));
    }
    const bool cache_ok = lay.native ? cache_.open_sized(sizes, n_layers, n_expert, err)
                                     : cache_.open((int64_t) loaded.size(), n_layers, n_expert, (int64_t) lay.max_blob, err);
    if (!cache_ok) { err = "helper HIP: " + (err.empty() ? "cache open failed" : err); close(); return false; }

    for (const auto& pair : loaded) {
        const int32_t slot = cache_.admit(pair.first, pair.second);
        const uint8_t* blob = source.blob(pair.first, pair.second);
        if (slot < 0 || !blob || !cache_.fill_slot_blocking(slot, blob, err, (int64_t) lay.blob_bytes(pair.first))) {
            err = "helper HIP: " + (err.empty() ? "cache fill failed" : err);
            close();
            return false;
        }
    }
    const auto& first = loaded.front();
    if (!cache_.verify_slot(cache_.slot_of(first.first, first.second), source.blob(first.first, first.second), err,
                            (int64_t) lay.blob_bytes(first.first))) {
        err = "helper HIP: " + err;
        close();
        return false;
    }

    const size_t scratch = std::max<size_t>(
        (size_t) strata::kernels::moe_hit_grouped_scratch_bytes(CAP, H, FF),
        strata::kernels::native_expert_scratch_bytes(CAP, FF));
    const size_t meta_bytes = sizeof(HelperMeta);
    // Phase 1 of the plan is the measurement of registering pages another process mapped.  If the driver says no,
    // the staged path below is the same tier with two copies per layer instead of none.
    registered_x_ = cudaHostRegister(shared_x, shared_x_bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess;
    registered_rows_ = cudaHostRegister(shared_rows, shared_rows_bytes, cudaHostRegisterPortable | cudaHostRegisterMapped) == cudaSuccess;
    if (!registered_x_ || !registered_rows_) {
        ignore(cudaGetLastError());
        if (registered_x_) { ignore(cudaHostUnregister(shared_x)); registered_x_ = false; }
        if (registered_rows_) { ignore(cudaHostUnregister(shared_rows)); registered_rows_ = false; }
    }
    const bool allocated =
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream", err) &&
        check(cudaMalloc((void**) &d_x_, (size_t) CAP * H * sizeof(float)), "input", err) &&
        check(cudaMalloc((void**) &d_out_, (size_t) CAP * H * sizeof(float)), "output", err) &&
        check(cudaMalloc((void**) &d_q8_, (size_t) CAP * (H / 32) * 36), "activation", err) &&
        check(cudaMalloc((void**) &d_scales_, (size_t) CAP * (H / 32) * sizeof(float)), "activation scales", err) &&
        check(cudaMalloc(&d_scratch_, scratch), "scratch", err) &&
        check(cudaMalloc(&d_meta_, meta_bytes), "group metadata", err) &&
        check(cudaHostAlloc(&h_meta_, meta_bytes, cudaHostAllocPortable), "metadata staging", err);
    if (!allocated) { close(); return false; }

    zero_copy_ = registered_x_ && registered_rows_ &&
                 cudaHostGetDevicePointer((void**) &z_x_, shared_x_, 0) == cudaSuccess &&
                 cudaHostGetDevicePointer((void**) &z_rows_, shared_rows_, 0) == cudaSuccess;
    ignore(cudaGetLastError());
    if (!zero_copy_) {
        // The staged path needs its own pinned result buffer: a D2H copy into pageable shared memory is not a
        // thing to rely on, so it lands here and is memcpy-ed into the mapping.
        if (!check(cudaHostAlloc((void**) &h_out_, (size_t) CAP * H * sizeof(float), cudaHostAllocPortable | cudaHostAllocMapped),
                   "result staging", err)) { close(); return false; }
    }
    auto* meta = (HelperMeta*) d_meta_;
    d_ptr_ = meta->ptr;
    d_start_ = meta->start;
    d_dst_ = meta->dst;
    d_tok_ = meta->tok;
    d_count_ = &meta->count;
    group_of_.resize(CAP);
    group_id_.reserve(CAP);
    ptr_.reserve(CAP);
    start_.reserve(CAP + 1);
    dst_.reserve(CAP);
    tok_.reserve(CAP);
    entry_of_row_.reserve(CAP);
    layers_present_.assign((size_t) n_layers, 0);
    for (const auto& pair : loaded) layers_present_[(size_t) pair.first] = 1;
    return true;
}

bool HelperTier::run(int32_t layer, const float* x, int32_t n_tok, const int32_t* entry, const int32_t* expert,
                     int32_t n_entries, float* rows, int32_t* out_entry, int32_t& n_rows, std::string& err) {
    n_rows = 0;
    if (device_ < 0) { err = "helper HIP: the tier is closed"; return false; }
    if (layer < 0 || (size_t) layer >= layers_present_.size() || n_tok <= 0 || n_tok > strata::kernels::cpu::MAXT ||
        n_entries <= 0 || n_entries > CAP) {
        err = "helper HIP: request outside the model, the window, or the row budget";
        return false;
    }
    if (!layers_present_[(size_t) layer]) { err = "helper HIP: no expert of layer " + std::to_string(layer) + " is loaded"; return false; }

    group_id_.clear(); ptr_.clear(); start_.clear(); dst_.clear(); tok_.clear(); entry_of_row_.clear();
    std::fill(group_of_.begin(), group_of_.begin() + n_entries, -1);
    for (int32_t j = 0; j < n_entries; ++j) {
        const int32_t e = expert[j];
        const int32_t which = entry[j];
        if (e < 0 || e >= n_expert_ || which < 0 || which >= n_tok * kK) {
            err = "helper HIP: entry " + std::to_string(j) + " is outside the request";
            return false;
        }
        const int32_t slot = cache_.slot_of(layer, e);
        if (slot < 0) {
            err = "helper HIP: the engine sent layer " + std::to_string(layer) + " expert " + std::to_string(e) +
                  ", which this helper does not hold";
            return false;
        }
        int32_t group = -1;
        for (size_t g = 0; g < group_id_.size(); ++g) if (group_id_[g] == e) { group = (int32_t) g; break; }
        if (group < 0) {
            group = (int32_t) group_id_.size();
            group_id_.push_back(e);
            ptr_.push_back((unsigned long long) cache_.device_slot(slot));
        }
        group_of_[(size_t) j] = group;
    }
    if (group_id_.empty()) { err = "helper HIP: the request carried no entry"; return false; }

    // Rows come back packed, and entry_of_row_ says which routed entry each one belongs to - the same trick the
    // remote tier uses for a USB4 return, here it is what keeps the PCIe trip at n_entries rows and not n_tok * k.
    for (size_t g = 0; g < group_id_.size(); ++g) {
        start_.push_back((int32_t) dst_.size());
        for (int32_t j = 0; j < n_entries; ++j)
            if (group_of_[(size_t) j] == (int32_t) g) {
                entry_of_row_.push_back(entry[j]);
                dst_.push_back((int32_t) dst_.size());
                tok_.push_back(entry[j] / kK);   // the activation row this routed entry belongs to
            }
    }
    start_.push_back((int32_t) dst_.size());

    auto* meta = (HelperMeta*) h_meta_;
    std::memcpy(meta->ptr, ptr_.data(), ptr_.size() * sizeof(ptr_[0]));
    std::memcpy(meta->start, start_.data(), start_.size() * sizeof(start_[0]));
    std::memcpy(meta->dst, dst_.data(), dst_.size() * sizeof(dst_[0]));
    std::memcpy(meta->tok, tok_.data(), tok_.size() * sizeof(tok_[0]));
    meta->count = (int32_t) group_id_.size();

    DeviceScope scope(device_);
    if (!scope.ok) { err = scope.error(); return false; }
    const cudaStream_t s = stream_;
    const float* src = zero_copy_ ? z_x_ : d_x_;
    float* out = zero_copy_ ? z_rows_ : d_out_;
    // Zero-copy kernels read Request::x and write Response::rows of the mapping, not the caller's buffers.  The helper
    // passes exactly those, so these copies only run for a caller (the parity test) that hands in its own.
    if (zero_copy_ && x != shared_x_) std::memcpy(shared_x_, x, (size_t) n_tok * H * sizeof(float));
    const bool staged =
        (zero_copy_ || check(cudaMemcpyAsync(d_x_, x, (size_t) n_tok * H * sizeof(float), cudaMemcpyHostToDevice, s), "copy input", err)) &&
        check(cudaMemcpyAsync(d_meta_, h_meta_, sizeof(HelperMeta), cudaMemcpyHostToDevice, s), "copy group metadata", err);
    if (!staged) return false;

    const auto& lay = strata::kernels::cpu::expert_layout();
    const int32_t groups = (int32_t) group_id_.size();
    if (lay.native) {
        strata::kernels::quantize_q8_1_rows(src, n_tok, H, d_q8_, s);
        const auto& fmt = lay.fmt[(size_t) layer];
        auto L = strata::kernels::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
        strata::kernels::native_expert_grouped(L, d_ptr_, d_start_, d_count_, d_dst_, d_tok_, groups,
                                               (int64_t) dst_.size(), d_q8_, d_scratch_, out, s);
    } else {
        strata::kernels::quantize_q8_0_scaled(src, d_q8_, d_scales_, (int64_t) n_tok * H, s);
        strata::kernels::moe_grouped_s2(d_ptr_, d_start_, d_count_, d_dst_, d_tok_, groups, (int64_t) dst_.size(),
                                        d_q8_, d_scales_, d_scratch_, out, s);
    }
    if (!check(cudaStreamSynchronize(stream_), "request", err)) return false;
    if (zero_copy_ && rows != shared_rows_) {
        std::memcpy(rows, shared_rows_, (size_t) dst_.size() * H * sizeof(float));
    } else if (!zero_copy_) {
        // The kernel wrote d_out_, so bring it home and then into the mapping.
        if (!check(cudaMemcpy(h_out_, d_out_, (size_t) dst_.size() * H * sizeof(float), cudaMemcpyDeviceToHost), "copy results", err))
            return false;
        std::memcpy(rows, h_out_, (size_t) dst_.size() * H * sizeof(float));
    }
    for (size_t r = 0; r < entry_of_row_.size(); ++r) out_entry[r] = entry_of_row_[r];
    n_rows = (int32_t) dst_.size();
    computed_ += (int64_t) dst_.size();
    ++requests_;
    return true;
}

}  // namespace strata::xhelper
