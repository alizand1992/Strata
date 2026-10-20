# Plan A: a cross-vendor expert helper (AMD card next to an NVIDIA primary)

**Status:** design approved by the owner; not started. This document is written to be executed by an agent working
in this repository on the owner's machine. Read it top to bottom once before starting. Every phase ends with a
**gate**: do not start the next phase until the gate passes, and stop and report if a **stop condition** fires.

Line numbers below are from commit `fb58e0d` (upstream main when this was written). If they have drifted, find the
same code by the quoted text or function name; do not guess.

---

## 1. Goal

Let the Strata engine (CUDA build, primary GPU = RTX 5060 Ti 16 GB) use the second card in the same PC, an
**RX 9060 XT 16 GB (AMD, gfx1200)**, as an additional expert tier, by running a small **HIP helper process** on
the AMD card and talking to it through shared host memory.

Two wins, in this order of importance:

1. **Fit a bigger model in 32 GB of RAM.** In the resident low-RAM mode the RAM copy holds only the experts no GPU
   holds. With the helper's ~12 GB of experts also left out of RAM, IQ3_XXS (~43 GB of experts) needs roughly
   43 - ~10 (RTX) - ~12 (helper) = **~21 GB of RAM** instead of ~33 GB. Today it does not fit on this box.
2. **Less CPU work per token** on any box: every expert the helper computes is one the CPU does not.

### Non-goals (for this plan)

- No change to the default engine behaviour. Everything is opt-in behind new flags (AGENTS.md).
- No layer split, no `--peer-device`, no `--expert-cache-device1..3`, no `--batch`, no `--pipeline-windows 2`
  together with the helper. The engine refuses those combinations with a clear message.
- No Windows support (Linux `memfd` + `futex`). No HIP *primary* with a CUDA helper.
- No upstream pull request until Phase 6's numbers exist. Work lands on the owner's fork (`alizand1992/strata`).

### Success criteria (the whole plan)

- IQ3_XXS starts in `--resident-experts` mode on the 32 GB box with the helper and **without** the
  "whole resident RAM mode does not fit" warning, and decodes with **0 file reads per request** after warm-up
  (the engine's `resident RAM: ... blob reads from the file` line).
- Decode tok/s with the helper is reported against the same-day RTX-only baseline (any value; the number is the
  deliverable, not a target).
- Killing the helper mid-request fails that request with a clear error; the engine stays up and serves the next
  request without the helper.
- The default build and default flags produce byte-identical output to the parent commit (checked, see Phase 6).

---

## 2. The machine

| Part | Value |
|---|---|
| CPU | Ryzen 9 3900X (12C/24T, AVX2, no AVX-512) |
| RAM | 32 GB DDR4 |
| Board | MSI B550 Tomahawk MAX |
| GPU 0 (primary) | RTX 5060 Ti 16 GB, CPU slot x16 Gen4 (the card itself is x8) |
| GPU 1 (helper) | RX 9060 XT 16 GB, gfx1200, **chipset slot x4 Gen3 (~3.9 GB/s)** |
| Disk | NVMe on CPU lanes |
| OS | Ubuntu (Linux), NVIDIA driver 580.x, CUDA 13.x |

The owner may move both cards to an X570 board (Ryzen 9 5900X, 64 GB, x8/x8 Gen4). Everything here works on either;
record which box each number came from.

### Pre-flight checks (run and paste the output into `docs/plans/results/00-preflight.md`)

```sh
nvidia-smi --query-gpu=name,pci.bus_id,memory.total,driver_version --format=csv
ls /dev/kfd /dev/dri/renderD*                     # the AMD card's compute node must exist
cat /sys/class/kfd/kfd/topology/nodes/*/name 2>/dev/null
sudo lspci -vv | grep -E "VGA|LnkSta:"            # link speed/width of both cards
nvcc --version; ls /opt/rocm 2>/dev/null && /opt/rocm/bin/hipcc --version
free -g; uname -r
```

If there is no system ROCm 7, Strata's setup can install one into `.venv` (docs/AMD_HIP.md, "ROCm"). The helper
needs `hipcc`, HIP runtime and hipBLAS headers (the HIP engine build needs the same).

---

## 3. Why the engine cannot do this today (read once)

- Strata compiles the same `.cu` sources for either CUDA or HIP. The HIP build maps every `cuda*` name onto HIP
  through `include/strata/platform/hip_compat/` (`cuda_runtime.h`, `cublas_v2.h`, ...). Two such copies cannot be
  linked into one binary; `CMakeLists.txt:88` refuses `STRATA_ENABLE_CUDA` with `STRATA_ENABLE_HIP`.
- docs/AMD_HIP.md:9-10 and docs/MULTI_GPU.md:53: "mixed AMD/NVIDIA execution in one run is not supported".
- The engine's existing helper-GPU tier, `RemoteExperts` (`include/strata/core/remote_experts.hpp`,
  `src/core/remote_experts.cpp`), is CUDA-device-only, and `src/program/generate.cpp:2046` refuses it together
  with the resident RAM mode (`--resident-cpu-experts does not support remote expert caches`).

So the helper is a **separate process** built with the HIP backend, and the engine gets a **new** tier type for it.
`RemoteExperts` is the template to copy from: it already does exactly the per-layer work the helper must do, on a
device instead of over shared memory.

---

## 4. Architecture

```
            strata (CUDA build, RTX)                       strata-expert-helper (HIP build, RX 9060 XT)
 ┌──────────────────────────────────────────┐        ┌──────────────────────────────────────────────┐
 │ dense layers, KV, MTP, router, RTX cache │        │ only expert blobs it was assigned (VRAM)      │
 │ CPU expert pool (RAM copy, resident mode)│        │ the same grouped expert kernel, compiled HIP  │
 │ ExternalExperts adapter  ◄──────────┐    │        │ request loop                                  │
 └─────────────────────────────────────│────┘        └──────────────────────▲───────────────────────┘
                                       │   shared memory (memfd, mmap'd)     │
                                       └──── header | assignment | request | response ──┘
                                             + two futex words for wake-ups
```

### Per layer, decode and verify windows

1. The pool thread classifies each routed entry `i` (token x 10 experts) as RTX-held, PCIe share, or CPU candidate
   (existing code, `src/core/expert_source.cpp` ~3000-3060 for one token, ~3235-3265 for a window).
2. `ExternalExperts::begin(...)` (new) marks the CPU candidates whose expert the helper holds as **owned**, writes
   one request (layer, activations, the owned entries' expert ids), and wakes the helper.
3. The CPU pool computes the remaining CPU rows; the RTX computes its rows. The pool already skips rows an
   external tier owns (it zeroes them; same as `RemoteExperts::owns`, `expert_source.cpp:3067`).
4. The helper quantizes the activations to q8_1, runs `native_expert_grouped` for its entries, writes compact rows
   and their entry indices into the response, and wakes the engine.
5. `ExternalExperts::finish(out)` waits for the response and copies each row into `out + i * H`
   (same as `RemoteExperts::finish`, `remote_experts.cpp`, last function).

### Data per layer (why x4 Gen3 is enough)

- In: activations `n_tok x 2560 x 4 B` (n_tok <= `MAXT` = 8) = at most 80 KiB, plus ids.
- Out: `rows x 2560 x 4 B`; at most `CAP = MAXT x 10 = 80` rows = 800 KiB, typically a few rows (~40-200 KiB).
- Expert weights cross the link once, at fill time.

### Prompt processing (important limitation)

The prompt path (`src/prefill/prefill.cpp`, the `Stager`) streams every expert the RTX does not hold from host
memory to the RTX in chunks. It has a path for `--peer-device` (`PeerPrefill`) but **none for remote tiers**. So:

- In **exclusive** mode (helper experts left out of RAM), the prompt path reads the helper's experts **from the
  model files on the SSD** (through `copy_blobs` on the mapped fallback). Correct, slower prompts.
- In **inclusive** mode (helper experts also kept in RAM), prompts are unaffected; only decode gains. Use this on
  the 64 GB box.

Helper participation in prompt processing is Phase 7 (optional).

---

## 5. Ground rules for the agent

- Branch per phase from the fork's `main`: `xhelper/p1-shm-spike`, `xhelper/p2-protocol`, ... Push each branch to
  `origin` (the fork). Open pull requests **against the fork's main**, never upstream.
- AGENTS.md applies: opt-in only; the default path stays byte-identical; build every backend a file touches
  (a file compiled into both the CUDA and the HIP build must build in both); measured numbers with what they ran on.
- Every new file starts with a one-paragraph header comment in the repo's style (what it is, the contract, the
  issue/plan it belongs to: "docs/plans/OPTION_A_EXCLUSIVE_RAM.md").
- Never `cudaSetDevice`/`hipSetDevice` to a device the process does not own. The engine sees only NVIDIA devices; the
  helper is started with `HIP_VISIBLE_DEVICES=<n>` so it sees only the AMD card as device 0.
- Results go in `docs/plans/results/` as Markdown, one file per phase, with the exact commands, the commit, the box,
  and raw numbers.

---

## Phase 0 - baselines on unchanged code

Build the CUDA engine as setup does (or `./setup.sh` with the CUDA backend, RTX only). Use the server
(`./setup.sh` then start it) or the engine directly. Record, for each run: commit, box, model, flags, decode tok/s,
prompt tok/s at a 4K and a 32K prompt, the startup lines about the expert cache and the RAM copy, and per request
`resident RAM: ... blob reads from the file`.

| Run | Model | Mode |
|---|---|---|
| 0a | Coder (IQ1_M) | `--resident-experts`, RTX only (setup's default on 32 GB) |
| 0b | IQ3_XXS | `--resident-experts` (expect the "does not fit" warning and the soft fallback; record it) |
| 0c | IQ3_XXS | `--mmap-experts --resident-budget-gib 20` |
| 0d | Coder | HIP engine alone on the 9060 XT (`./setup.sh --backend hip`), for the card's own number |

Use one fixed prompt set for every phase: save 5 prompts in `docs/plans/results/prompts/` (a 4K and a 32K
code-reading prompt, two short chat prompts, one tool-use style prompt). Greedy (`temperature 0`) where the engine
allows, so runs are comparable.

**Gate 0:** the four runs are recorded in `docs/plans/results/00-baseline.md`.

---

## Phase 1 - shared-memory spike (standalone, no engine change)

Purpose: prove the transport before touching the engine. This phase decides the transfer design.

### Build

New directory `tools/xhelper_spike/` with two programs and a script. Do not add them to the main CMake build; build
with `nvcc` and `hipcc` directly from the script.

- `spike_cuda.cu` (nvcc): the parent.
- `spike_hip.cpp` (hipcc, `--offload-arch=gfx1200`): the child.
- `run.sh`: builds both, runs the parent, prints a results table.

### What the parent does

1. `memfd_create("strata-xh-spike", 0)` (no `MFD_CLOEXEC`), `ftruncate` to 256 MiB, `mmap(PROT_READ|PROT_WRITE,
   MAP_SHARED)`.
2. `cudaHostRegister(p, size, cudaHostRegisterPortable | cudaHostRegisterMapped)`; record success/error.
3. `fork` + `execv` the child with the fd number as an argument, and `HIP_VISIBLE_DEVICES` set to the AMD card.
4. Runs the tests below, coordinating through two `std::atomic<uint32_t>` words at the start of the mapping, using
   `syscall(SYS_futex, addr, FUTEX_WAIT/FUTEX_WAKE, ...)` **without** `FUTEX_PRIVATE_FLAG` (cross-process).

### What the child does

`mmap` the inherited fd, `hipHostRegister(p, size, hipHostRegisterMapped | hipHostRegisterPortable)`;
`hipHostGetDevicePointer` for the zero-copy alias. Record success/error.

### Tests (each prints PASS/FAIL and numbers)

1. **Dual registration:** both registrations succeed on the same pages.
2. **Integrity:** the RTX writes a pattern into region A with `cudaMemcpyAsync` D2H from device memory; the child
   reads region A with `hipMemcpyAsync` H2D into AMD VRAM and checks it on the GPU (a small kernel computing a
   checksum). Then the reverse direction (AMD writes region B, RTX checks). Also once with zero-copy kernels reading
   and writing through the mapped device pointers on both sides.
3. **Bandwidth:** H2D and D2H on each card from/to the shared mapping for 160 KiB, 800 KiB and 64 MiB, 50 repetitions,
   median and p99. Compare with the same sizes from a private `cudaHostAlloc` / `hipHostMalloc` buffer.
4. **Round-trip latency:** 10,000 ping-pongs where the parent writes a sequence number, the child sees it and answers.
   Three variants: both spin (`_mm_pause`), child futex-waits / parent spins, both futex-wait. Median, p99, p99.9.
5. **Round trip with work:** parent writes 80 KiB, child H2D copies it to VRAM, runs a trivial kernel, D2H copies
   200 KiB back into the mapping, signals; parent checks. Median and p99.

### Gate 1 (record in `docs/plans/results/01-spike.md`)

- Integrity passes in both directions.
- Round trip with work: **p50 < 150 us, p99 < 500 us** with the spin variant on this box.

### Stop conditions and fallbacks

- `hipHostRegister` or `cudaHostRegister` fails on the memfd mapping: try `shm_open` + `mmap`, then an anonymous
  `MAP_SHARED` mapping created before `fork`. If dual registration still fails, use the **staged design**: the child
  registers its own `hipHostMalloc` buffers and `memcpy`s between them and the shared mapping (~80-800 KiB per
  layer). Measure its round trip; continue only if Gate 1's latency still holds.
- p99 above 500 us with spinning: stop and report. The design costs 48 layers x this per window; above that it
  cannot beat the CPU.

---

## Phase 2 - the protocol header

New header `include/strata/xhelper/protocol.hpp`, compiled into both binaries. **No CUDA/HIP includes**; plain C++20.
Everything lives in one shared mapping, cache-line aligned.

```cpp
namespace strata::xhelper {

inline constexpr uint64_t kMagic   = 0x31504C484B525453ull;   // "STRKHLP1" little-endian
inline constexpr uint32_t kVersion = 1;
inline constexpr int kH = 2560, kMaxTok = 8, kK = 10, kCap = kMaxTok * kK;   // must equal kernels/cpu H, MAXT, k

enum class EngineState : uint32_t { Init = 0, Assigned = 1, Running = 2, Shutdown = 3 };
enum class HelperState : uint32_t { Starting = 0, Loading = 1, Ready = 2, Failed = 3, Exited = 4 };

struct alignas(64) Header {
    uint64_t magic, total_bytes;
    uint32_t version, header_bytes;
    uint32_t n_layers, n_expert, h, cap, max_tok;
    std::atomic<uint32_t> engine_state, helper_state;
    std::atomic<uint64_t> helper_heartbeat_ns;     // CLOCK_MONOTONIC, the helper bumps it every 100 ms while idle
    int32_t helper_pid;
    uint64_t vram_budget_bytes;                    // 0 = helper decides: free VRAM - 1 GiB
    uint32_t n_assigned, n_loaded;                 // pairs offered by the engine / actually loaded by the helper
    uint64_t assigned_off, loaded_off;             // byte offsets of the Pair arrays below
    uint64_t request_off, response_off;
    char model_pack[1024];                         // the engine's --pack path (o.pack)
    char native_preset[1024];                      // the engine's --native argument (o.native_preset; "" if none)
    char error[512];                               // helper's last fatal error, NUL-terminated
};
struct Pair { int32_t layer, expert; };

struct alignas(64) Request {                       // engine -> helper; exactly one in flight
    std::atomic<uint32_t> seq;                     // futex word; engine increments after writing the payload
    std::atomic<uint32_t> helper_sleeping;         // helper sets 1 before FUTEX_WAIT, 0 after
    int32_t layer, n_tok, n_entries;               // n_entries = number of owned entries
    int32_t entry[kCap];                           // index i into the n_tok*k routed list
    int32_t expert[kCap];                          // expert id of entry[j]
    alignas(64) float x[kMaxTok * kH];             // activations of the n_tok tokens (as the pool sees x_f)
};

struct alignas(64) Response {                      // helper -> engine
    std::atomic<uint32_t> seq;                     // futex word; helper sets = Request::seq when done
    std::atomic<uint32_t> engine_sleeping;
    int32_t status;                                // 0 ok, else an error code; error text in Header::error
    int32_t n_rows;                                // == Request::n_entries on success
    int32_t entry[kCap];                           // row r belongs to routed entry entry[r]
    alignas(64) float rows[kCap * kH];
};

}  // namespace strata::xhelper
```

Rules:

- Writer stores the payload, then `seq.store(n, std::memory_order_release)`; reader `seq.load(acquire)`. Wake with
  `FUTEX_WAKE` only if the other side's `*_sleeping` flag is 1 (avoid a syscall on the fast path).
- `static_assert` that `kH`, `kMaxTok`, `kCap` match `strata::kernels::cpu::H`, `MAXT` and the remote tier's `CAP`
  in the engine-side translation unit (`src/core/remote_experts.cpp` defines `CAP = MAXT * 10`).
- Total mapping size: header + 2 x `n_layers*n_expert` pairs + request + response (~1.2 MiB). Allocate 4 MiB.
- A unit test (`tests/xhelper/protocol_test.cpp`, CPU only, in ctest) checks sizes, alignment, offsets, and a
  two-thread ping-pong over a heap buffer using the same atomics and futex helpers.

**Gate 2:** the test passes under ctest in the CUDA build and in the HIP build.

---

## Phase 3 - the helper binary `strata-expert-helper`

### Files

- `src/xhelper/helper_main.cpp` - argument parsing, shared mapping, lifecycle, request loop.
- `src/xhelper/helper_tier.hpp/.cpp` - the device side: an adaptation of `RemoteExperts` for **device 0** that
  loads an explicit pair list and computes one request.
- CMake: inside the existing GPU-targets block, next to `add_executable(strata src/program/generate.cpp)`
  (`CMakeLists.txt:549`), add `strata-expert-helper`, **only when `STRATA_ENABLE_HIP` is ON**:
  ```cmake
  if(STRATA_ENABLE_HIP)
    add_executable(strata-expert-helper src/xhelper/helper_main.cpp src/xhelper/helper_tier.cpp)
    target_link_libraries(strata-expert-helper PRIVATE strata_engine)
  endif()
  ```
  `strata_engine` already brings `FileExpertSource`, `ExpertCache` and the kernels in the HIP build.

### Command line

```
strata-expert-helper --shm-fd N            # inherited fd of the shared mapping (required)
                     [--spin-us 200]        # spin this long before FUTEX_WAIT
                     [--verbose]
```

Everything else (model path, pairs, budget) comes from the shared header, so engine and helper cannot disagree.

### Startup (`helper_main.cpp`)

1. `mmap` the fd; check `magic`, `version`, sizes; on mismatch set `helper_state = Failed` with a message and exit 2.
2. `helper_state = Loading`; `helper_pid = getpid()`.
3. Open the expert source **exactly like the engine's mmap path** (`src/program/generate.cpp:4151-4159`):
   `FileExpertSource src; src.set_gguf(native_preset); src.open(model_pack, n_layers, n_expert, err)`.
   This also initializes `strata::kernels::cpu::expert_layout()`, which the kernels need.
4. `HelperTier::open(budget, assigned pairs)`:
   - `hipMemGetInfo`; budget = `vram_budget_bytes` or free - 1 GiB.
   - Walk the assigned pairs in order; take each while it fits (blob size per layer:
     `lay.native ? (lay.blob_bytes(layer) + 255) / 256 * 256 : lay.max_blob`, as in `RemoteExperts::open`).
   - `ExpertCache::open_sized(sizes, n_layers, n_expert, err)` (native) or `open(...)`; `admit` + `fill_slot_blocking`
     from `src.blob(layer, expert)`; `verify_slot` on the first (copy the code from `RemoteExperts::open`).
   - Allocate the same scratch/staging as `RemoteExperts::open`: `d_x`, `d_q8`, `d_scales`, `d_scratch`, `d_meta`,
     `d_out`. Register the request/response regions with `hipHostRegister` (Phase 1 decided whether this works; if
     not, allocate `hipHostMalloc` staging and `memcpy`).
5. Write the loaded pairs to `loaded_off`, set `n_loaded`, then `helper_state = Ready`.

### Request loop

```
last = 0
loop:
  wait until request.seq != last   (spin --spin-us with _mm_pause, then helper_sleeping=1, FUTEX_WAIT, helper_sleeping=0)
  if engine_state == Shutdown: break
  last = request.seq
  ok = tier.run(request.layer, request.x, request.n_tok, request.entry, request.expert, request.n_entries,
                response.rows, response.entry, &n_rows, err)
  response.status = ok ? 0 : 1; if !ok copy err to header.error
  response.n_rows = n_rows
  response.seq.store(last, release); if response.engine_sleeping: FUTEX_WAKE
  bump heartbeat
```

`HelperTier::run` is the body of `RemoteExperts::begin` + `finish` with the classification removed (the engine
already decided ownership): group entries by expert, build `ptr/start/dst/tok` metadata, upload x and metadata,
then for native layouts `quantize_q8_1_rows` + `native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff)`
+ `native_expert_grouped(...)`, otherwise `quantize_q8_0_scaled` + `moe_grouped_s2(...)` - **the exact calls in
`remote_experts.cpp` `begin()`** - then `hipStreamSynchronize` and copy compact rows plus their entry indices into
the response. Keep the device work on one non-blocking stream. No `RemoteExpertOpt` (reduce) support.

### Shutdown and failure

- `engine_state == Shutdown` or `getppid()` changed (engine died): free everything, `helper_state = Exited`, exit 0.
- Any HIP error: `helper_state = Failed`, message in `header.error`, exit 3. Never write partial rows with status 0.

### Tests

- `tests/xhelper/helper_tier_parity.cpp` (HIP build, ctest, needs a model; skipped with a message when
  `STRATA_TEST_PACK` is unset): load 32 pairs from layers {0, 23, 47}, build a request with random activations,
  run `HelperTier::run`, and compare each row with the CPU path's row for the same blob and activation (call the
  CPU kernel the pool uses: `strata::kernels::cpu` native expert functions, as `src/kernels/native_expert_parity.cpp`
  does). Pass: relative L1 difference (`sum|a-b| / sum|b|`, the `rel()` of native_expert_parity.cpp) **< 5e-3** per
  row. That file documents a few 1e-3 between CPU and GPU from activation rounding alone.

**Gate 3:** the parity test passes on the 9060 XT; the helper starts, loads ~12 GB, reports Ready, and exits cleanly
on Shutdown, driven by a tiny test driver (`tools/xhelper_spike/drive_helper.cpp`, CPU-only, no CUDA).

---

## Phase 4 - the engine side (`ExternalExperts`)

### New files (CUDA build only; refuse in a HIP engine build)

- `include/strata/core/external_experts.hpp`, `src/core/external_experts.cpp`. Add the `.cpp` to `strata_engine`'s
  source list (`CMakeLists.txt:446`), guarded so it compiles in the HIP build as a stub that always fails `open`
  ("the external expert helper needs the CUDA engine").

### Interface (mirror `RemoteExperts`)

```cpp
class ExternalExperts {
public:
    struct Options { std::string helper_exe; int hip_device = 0; uint64_t vram_budget = 0;
                     bool inclusive_ram = false; int spin_us = 200; int timeout_ms = 2000; };
    bool open(const Options&, const std::string& pack, const std::string& native_preset,
              int64_t layers, int64_t experts,
              const std::vector<std::pair<int32_t,int32_t>>& ranked,   // hottest first
              const ExpertCache& primary, std::string& err);          // spawns the helper, waits for Ready
    void close();                                                     // Shutdown, waitpid with timeout, SIGKILL
    bool begin(int64_t layer, const float* x, const int32_t* ids, int64_t n_tok, int64_t k,
               const int32_t* kind, const int32_t* primary_res, std::string& err);
    bool owns(int64_t entry) const;
    bool holds(int64_t layer, int32_t expert) const;                  // from the helper's loaded list
    bool finish(float* out, std::string& err);
    const std::vector<std::pair<int32_t,int32_t>>& loaded() const;
    bool alive() const;
    // stats, as RemoteExperts: computed(), launched_layers(), ms_begin(), ms_wait(), timeouts()
};
```

`begin` copies `RemoteExperts::begin`'s classification loop verbatim (skip `kind[i] != -1`, skip out-of-range ids,
skip `primary_res[...] >= 0`, own the entry when `holds(layer, e)`), then writes the request instead of launching
kernels. `finish` waits on `Response::seq` (spin up to `timeout_ms`, with `engine_sleeping` + futex after the first
`spin_us`) and scatters `rows[r]` into `out + entry[r] * H`.

On timeout, a `status != 0`, or a dead helper (`waitpid(WNOHANG)` / heartbeat older than 5 s):
`finish` returns false with an error naming the layer; set `alive_ = false`; from then on `begin` owns nothing, so
the CPU computes those experts from the mapped files. The current request fails (the pool's `d.failed` path); the
engine stays up. Log once: `external experts: the helper stopped (<reason>); its experts are read from the model
files from now on`.

### Spawning

`memfd_create` + `ftruncate(4 MiB)` + `mmap`, fill the header (pack path, native preset, assigned pairs,
budget), `cudaHostRegister` the request/response region (Phase 1 result), then `posix_spawn` the helper with
`--shm-fd N`, environment `HIP_VISIBLE_DEVICES=<hip_device>` and `CUDA_VISIBLE_DEVICES=` (empty, so the helper never
touches the RTX). Wait up to 120 s for `Ready` or `Failed`, printing progress every 10 s.

**Assignment:** like the remote path (`generate.cpp:4832-4895`): the profile's ranked pairs, extended with every
unranked pair in expert-then-layer order, skipping pairs the primary cache holds (`xcache.slot_of(l, e) >= 0`). Offer
all of them; the helper loads what fits its budget, in order.

### Wiring in `src/program/generate.cpp`

1. **Options** (near `expert_cache_remote`, line ~445) and parsing (near line ~1757):
   - `--external-experts PATH` (path to `strata-expert-helper`; enables the feature)
   - `--external-experts-device N` (HIP device index, default 0)
   - `--external-experts-gib X` (VRAM budget on the helper; default 0 = free - 1 GiB)
   - `--external-experts-ram exclusive|inclusive` (default exclusive)
   - env `STRATA_XH_SPIN_US` (default 200), `STRATA_XH_TIMEOUT_MS` (default 2000), `STRATA_XH_TRACE=1` (per-layer
     timings to stderr).
   Add all four to the usage text (the `--expert-cache-remote-placement` lines, ~824, show the format).
2. **Validation** (next to the check at ~2046), refuse with one clear line each: with `--expert-cache-device1..3`,
   `--peer-device`, a layer split (`multi_gpu`), `--batch`, `--pipeline-windows 2`, `--vram-elastic`, `--kv-grow` (allowed
   since branch xhelper/kv-grow, see results/09-kv-grow.md),
   `--remote-expert-opt`; without `--expert-profile`; without the expert pool; in a HIP build; on non-Linux.
3. **Open** the helper right after the remote-experts block (after line ~4914, before `Drive drive;`), so `xcache`
   is filled and the helper's pairs exist before the RAM copy is built. Print:
   `strata generate: external helper (HIP device N): K experts, X GiB, loaded in T s; with CUDA0's M that is ...`.
4. **Pool hook:** add `ExternalExperts* external = nullptr;` to `struct ExpertDispatch`
   (`include/strata/core/expert_source.hpp:296`, next to `remote[3]`) and set `drive.d.external` next to
   `drive.d.remote[...]` (~4946). In `src/core/expert_source.cpp`, at every place the remote tiers are used, do the
   same for `external` (after the remote loop, so a helper GPU and the external helper never both own a row):
   - one-token path: `begin` at ~3029, `owns` at ~3067, `finish` at ~3089;
   - window path: `helper_holds` lambda at ~3150 (keep helper experts out of the PCIe share), `begin` at ~3261
     including the `kind[i] = 2` marking at ~3264, `finish` at ~3356.
   With `external == nullptr` the code path must be unchanged (byte-identical output; Phase 6 checks it).
5. **RAM copy:** at the resident-mode block (~6091-6106), in exclusive mode append the helper's loaded pairs to
   `stage_pairs` before `src.pin_cache_complement(...)`. That function already leaves out every pair it is given
   (`make_cache_complement_plan`'s `additional_gpu_pairs`, `expert_source.cpp:117`). Note: a non-empty list turns
   the prompt path's slot lending off inside `pin_cache_complement` (see its `lend` condition, ~2294: `additional_gpu_pairs.empty()`); record the
   prompt-speed effect in Phase 6.
6. **Adaptive tier:** the primary's adaptive swaps must never promote an expert the helper holds. Today that filter
   is opt-in (`STRATA_DISJOINT_ADAPT`, the `helper_holds` lambda at ~4938). Make it unconditional for `external`:
   `if (external && external->holds(l, e)) return true;` before the env check. Without this, an exchange is staged
   with an `in` that is not in the RAM copy, the swap becomes a non-exchange (`generate.cpp:7669`), and the evicted
   expert drops to file-only, breaking "every expert is on a GPU or in RAM".
7. **Look-ahead:** `RouterLookahead` warms the pages of predicted experts that no cache holds. Check
   `RouterLookahead::submit` / its filter and make it skip helper-held experts in exclusive mode; otherwise it reads
   the helper's experts from the SSD for nothing. Verify with `STRATA_IO_STATS=1`: helper-held experts cause no file
   reads during decode.
8. **Stats:** next to the remote tiers' per-request line (~11115), print
   `external: N rows, L layers, begin X ms, wait Y ms, timeouts T` per request.
9. **Shutdown:** `external.close()` on every exit path of the engine, including signals the engine already handles.

### Tests

- `tests/xhelper/external_experts_test.cpp` (CUDA build, ctest, needs `STRATA_TEST_PACK` + a HIP helper path in
  `STRATA_TEST_HELPER`; skipped otherwise): spawn the helper with 64 pairs, run 200 random requests through
  `begin/finish`, compare rows with the CPU pool's (rel < 5e-3), then `kill -9` the helper and check `finish` fails
  cleanly and `alive()` is false.

**Gate 4:** the engine runs the Coder with `--resident-experts --external-experts ... --external-experts-ram
inclusive` for 20 requests of the prompt set without errors; the stats line shows helper rows on every request.

---

## Phase 5 - config and setup

Minimum (required): document a manual config. The server takes engine flags from the config's `"args"` list and
environment from `"env"` (`serve/server.py:2165`, `:2361`). Write `docs/plans/results/05-config.md` with a working
`strata-*.json` fragment:

```json
"args": [ "...existing...", "--resident-experts",
          "--external-experts", "/path/to/build-hip/strata-expert-helper",
          "--external-experts-device", "0", "--external-experts-ram", "exclusive" ],
"env": { "STRATA_XH_SPIN_US": "200" }
```

Optional: teach `setup.py` to detect "one usable NVIDIA + one supported AMD card", build both engines (CUDA into the
usual build dir, HIP into `build-hip` with `-DCMAKE_HIP_ARCHITECTURES=gfx1200`), and offer the helper. The HIP build
command is in docs/AMD_HIP.md "Build". Setup has offline tests (`python tools/test_setup_<name>.py`); add one.

---

## Phase 6 - measurement and acceptance

Same day, same build, same prompt set, greedy. Report in `docs/plans/results/06-results.md`:

| Arm | Model | RAM mode | Decode tok/s | Prompt 4K tok/s | Prompt 32K tok/s | file reads/request | RAM committed |
|---|---|---|---|---|---|---|---|
| A | Coder | resident, RTX only | | | | | |
| B | Coder | resident + helper, inclusive | | | | | |
| C | Coder | resident + helper, exclusive | | | | | |
| D | IQ3_XXS | resident (soft fallback), RTX only | | | | | |
| E | IQ3_XXS | resident + helper, exclusive | | | | | |

Also report: helper round trip per layer (p50/p99 from `STRATA_XH_TRACE=1`), share of routed entries the helper took,
how often a layer waited on the helper longer than the CPU pool's time for that layer.

**Byte-identity check of the default path:** build the parent commit and this branch; run the Coder with the default
flags on two prompts with `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0` (docs/DETAILS.md lists these as the
switches for byte-identical repeats); outputs must match token for token.

**Robustness:** 100 requests in a row with arm E; then `kill -9` the helper during a request: that request errors,
the next succeeds (CPU from files), the engine stays up.

**Acceptance:** arm E starts without the "does not fit" warning, shows 0 file reads per request during decode after
warm-up, and completes the robustness run. Speed numbers are reported whatever they are.

---

## Phase 7 - optional follow-ups (only after Phase 6)

1. **Helper in prompt processing** (exclusive mode's slow prompts). Model it on `PeerPrefill`
   (`src/prefill/prefill.cpp:555-600`): per MoE layer of a chunk, send the chunk's activations once, the helper
   computes its experts' rows and returns the **weighted sum per token** (needs the router weights; the
   `RemoteExpertOpt` "reduce" mode in `remote_expert_opt.hpp` is the existing pattern) so the return is
   `n_tok x H` floats, not `rows x H`. At 8,192 tokens that is ~84 MB each way per layer: ~43 ms per layer on x4 Gen3,
   ~11 ms on x8 Gen4. Compare with reading the same experts from the SSD before building.

   **Status (2026-10-20, branch `xhelper/p7-prompt`): built, not measured yet.** Opt-in, `--external-experts-prompt N`
   (tokens per request, e.g. 2048; default 0 keeps the prompt path as it was). Protocol version 2: a prompt request kind,
   and a bulk region after the fixed 4 MiB of the mapping (x and the sums, N x 2560 floats each: 40 MiB at N = 2048).
   The helper runs the prompt path's MMQ products on its own slots (`src/xhelper/helper_prompt.cpp`) and keeps their
   buffers off its expert budget. The engine (`Prefill::set_external`) holds a lease on the helper for each chunk,
   leaves the helper's experts out of a streamed chunk's stream and out of a staged chunk's staging and CPU share,
   lays their rows out last, sends the requests from a thread while the RTX runs its own experts, and adds the sums in
   the combine (`moe_combine_peer`). A request the helper does not answer is computed on the CPU pool from the files.
   Measured motive, from the live server's log on the B550 box (IQ3_S, 256K, exclusive): agent turns of 100-1400 new
   tokens took 13-34 s, and one 361-token turn read 21.8 GB from the SSD for 19.1 GB of expert reads.
2. **Adaptive helper tier**, batched: every `--adapt-every` windows, swap the most-routed CPU experts into the
   helper's least-used slots (same hysteresis and decay rules as the primary, `generate.cpp:7640-7660`), at most
   `--external-adapt-swaps` per round, one batched request carrying the blobs, applied between windows. In exclusive
   mode the evicted blob must come back into the RAM slot of the one promoted (as the primary's exchange does), so
   the RAM copy stays exclusive.
3. **Two requests in flight** (a ring of 2) to support `--pipeline-windows 2`.

---

## Risks and decisions (keep this list current)

| Risk | Where it shows | Decision |
|---|---|---|
| Dual pinning of one mapping fails | Phase 1 | Staged copy fallback; continue only if latency gate holds |
| Wake-up latency too high | Phase 1, 6 | Spin first (`STRATA_XH_SPIN_US`); helper share capped if it becomes the slowest |
| Helper rounding differs from CUDA | Phase 3 | Accept within rel < 5e-3; outputs not byte-identical to RTX-only (as CPU vs GPU today) |
| Exclusive mode slows prompts | Phase 6 | Inclusive mode on 64 GB boxes; Phase 7.1 if needed |
| Adaptive tier breaks exclusivity | Phase 4 step 6 | Unconditional filter for helper-held experts |
| Look-ahead reads helper experts from SSD | Phase 4 step 7 | Filter; verify with `STRATA_IO_STATS=1` |
| x4 Gen3 link | All | Fine for activations/rows; fill takes seconds; X570 box gives x8 Gen4 |
| Helper crash | Phase 4, 6 | Request fails, engine continues on CPU + files |

## Done means

All gates passed; `docs/plans/results/` holds phases 0-6; the fork has one branch per phase merged into its main;
the default path's byte-identity is shown; the owner has the numbers to decide on an upstream issue.
