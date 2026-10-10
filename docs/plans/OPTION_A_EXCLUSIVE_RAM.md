# Plan A: experts in VRAM *or* RAM, never both ("on the fly")

Goal: run a bigger model (IQ3_XXS, later IQ3_S) on a 32 GB RAM box by keeping in RAM only the experts no GPU holds,
and by using a second, **different-vendor** GPU (RTX 5060 Ti 16 GB + RX 9060 XT 16 GB) as a second expert tier.

Target box: Ryzen 9 3900X, 32 GB DDR4, B550 (RTX on CPU x16 Gen4, 9060 XT on chipset x4 Gen3), Linux, NVMe.

## What already exists (read before writing code)

The exclusive layout is already in the engine as the **resident low-RAM mode** (`--resident-experts`, engine 0.1.30;
on a layer split since 0.1.40, #642/#848):

- At start the RAM copy holds exactly the experts no GPU cache holds:
  `detail::make_cache_complement_plan` (`src/core/expert_source.cpp:117`), filled by
  `FileExpertSource::pin_cache_complement` (`:2182`). Its `additional_gpu_pairs` argument already excludes a second
  GPU's experts.
- Adaptive swaps keep it exclusive: the evicted expert is copied back from VRAM into the RAM slot of the one that
  replaces it (`stage_exchange` / `commit_exchanges`, `:2811-2850`; the decision is in `src/program/generate.cpp:7640-7670`).
- Swaps are **already batched**: each adaptation round (`--adapt-every` windows) picks up to `--adapt-swaps` swaps,
  only when the gain clears a hysteresis of 1.5 routed uses (`generate.cpp:7657`), with usage decayed by
  `--adapt-decay` (0.7). `--adapt-async 1` moves a round's copies between decode windows on a helper thread, and
  `STRATA_EXCHANGE_ROTATE=1` swaps buffer ownership instead of copying (docs/EXCHANGE_ROTATION.md).
- Measured on a split (docs/MULTI_GPU.md): RTX 4060 Ti + RTX 5080, 32 GB RAM, Swift 1.5 IQ3_XXS, 22 GiB of experts
  resident in RAM: 71 tok/s, vs 25 tok/s on the 5080 alone.

What blocks this box: **mixed NVIDIA + AMD in one run is not supported** (docs/AMD_HIP.md:9-10, docs/MULTI_GPU.md:53;
`CMakeLists.txt:88` refuses CUDA and HIP in one build). So today Strata here uses one 16 GB card, and with one 16 GB
card on 32 GB RAM only the Coder fits resident; IQ3_XXS "on a 32 GB PC stays mapped" (docs/DETAILS.md, low-RAM mode).

## Step 0 - baseline on today's code (no changes, ~1 hour)

1. `./setup.sh` (CUDA, RTX only) with the **Coder**, low-RAM resident mode (setup picks it on 32 GB).
2. Record: decode tok/s, prompt tok/s at 4K and 32K, `resident RAM: ... blob reads from the file` per request (should
   be 0), and the startup line with the complement's GiB.
3. Same with `--low-ram mmap` and IQ3_XXS + `--resident-budget-gib` (Plan B's baseline too).
4. Optional: the 9060 XT alone with `./setup.sh --backend hip` and the Coder, to know each card's single-card number.

These numbers decide whether the rest is worth building.

## The gap worth building: a cross-vendor expert helper (opt-in)

The engine already has a second-GPU expert tier, `RemoteExperts` (`include/strata/core/remote_experts.hpp`): a static,
profile-filled cache on another CUDA device; per layer it computes the rows of the experts it owns while the CPU pool
works, and returns them through pinned host rows. The helper below is the same contract, served by a **separate HIP
process** instead of a CUDA device in the same process.

Why a separate process: one process cannot load both the CUDA and the HIP runtime for this codebase (the HIP build
maps `cuda*` names onto HIP through `include/strata/platform/hip_compat/`). Two processes avoid that entirely.

### Design

- **Helper binary** (`strata-expert-helper`, built with `STRATA_ENABLE_HIP=ON`): loads only the expert blobs of the
  `(layer, expert)` pairs it is assigned, into its VRAM, and runs the existing HIP grouped expert kernel
  (`native_expert_grouped` path). No dense weights, no KV, no MTP.
- **Shared memory**: one `memfd` mapping, registered with `cudaHostRegister` in the engine and `hipHostRegister` in
  the helper, so both GPUs DMA it directly. Per layer it holds: the q8_1 activations, the routed ids and the
  helper-row mask (input), and the output rows (`k x n_embd` floats).
- **Batching (one exchange per layer, not per expert):** the engine writes the whole layer's request in one go
  (all tokens of the verify window x their routed ids), signals once (eventfd or a futex word), and the helper launches
  one grouped kernel for all of its rows in that layer, then signals back once. 48 layers x 2 wakeups per window;
  at ~10-20 us per wakeup that is ~1-2 ms per window, small next to a 20-70 ms window.
- **PCIe traffic is small by construction:** only activations in and rows out cross the x4 Gen3 link
  (2560 x 4 bytes per row; a 4-token window with ~4 helper rows per layer is ~160 KB per layer, ~8 MB per window,
  ~2 ms at ~3.9 GB/s). Expert weights cross the link **once**, at fill time. If this is too slow on x4 Gen3, the X570
  box (x8 Gen4) quadruples it.
- **Static tier first:** fill the helper from the shipped expert profile, like `RemoteExperts` (the ranked pairs after
  CUDA0's cache). Adaptive helper swaps come later and should reuse the round-based, hysteresis-gated batching above.
- **RAM saving:** pass the helper's pairs as `additional_gpu_pairs` to `make_cache_complement_plan`, so the RAM copy
  leaves them out. That is the whole point for this box:
  IQ3_XXS ~43 GB of experts - ~10 GB on the RTX - ~12 GB on the 9060 XT = **~21 GB in RAM**, which fits 32 GB with
  the engine's 4 GB headroom. Today it does not fit at all.

### Work, in pull-request-sized steps

1. **Shared-memory transport + parity test** (no engine change): a test that runs one layer's rows through the helper
   and through the CPU pool from the same blobs and compares them within the same tolerance the repo uses between
   CPU and GPU rounding. Build: HIP for the helper, CUDA for the test driver.
2. **Helper process**: blob loading for an assigned pair list, the grouped kernel, the request loop, clean shutdown
   when the engine exits (the engine owns the helper's lifetime; a dead helper is detected by a timeout and its rows
   fall back to the CPU pool, never to stale rows).
3. **Engine side, behind an opt-in flag** (`--external-helper <cmd>`): a `RemoteExperts`-shaped adapter that owns no
   CUDA device. Hook it where the pool calls `RemoteExperts::begin` (`src/core/expert_source.cpp:3029` for one token,
   `:3261` for a verify window); reuse `owns()` / `holds()` so the CPU pool skips its rows. Default path untouched (AGENTS.md: "A new feature is opt-in, and the default path stays byte-identical").
4. **Complement planning**: include the helper's pairs in `additional_gpu_pairs`; log the RAM saved.
5. **Setup**: detect "one NVIDIA + one AMD card", offer the helper, build both engines.
6. **Adaptive helper swaps** (later): reuse the round/hysteresis/decay rules; each swap is one blob copied RAM -> helper
   VRAM and the evicted blob copied back into RAM (to keep the RAM copy exclusive), batched per round and applied
   between windows like `--adapt-async`.

### Measurements to report (repo style: numbers with what they ran on)

- Decode and prompt tok/s, RTX alone vs RTX + helper, Coder and IQ3_XXS, same day, same build.
- Per-layer helper round-trip time (p50/p99) and how often a layer waited on the helper.
- RAM committed at start, and file reads per request (should stay 0 in the resident mode).

### Risks

- **Synchronization stalls**: a layer cannot finish until the helper answers. Mitigation: the helper only takes rows
  it can finish within the CPU pool's time for that layer (measure, then cap its share per layer).
- **Rounding**: HIP and CUDA kernels round differently; answers will not be byte-identical to a single-card run.
  The repo already accepts this between CPU and GPU rows; document it.
- **x4 Gen3 link**: fine for activations and rows; the initial fill of ~12 GB takes ~3-4 s.
- **Upstream fit**: maintainers have said mixed vendors are unsupported; open an issue with step 0's numbers before
  sending a pull request.

## Effort and expected result

High: a new process, IPC, and engine integration (weeks, not days). Expected: IQ3_XXS fits and runs on this 32 GB box
at roughly the single-RTX resident speed or better. That is an estimate, not a measurement. Moving the hardware to the
X570 box (64 GB, x8/x8 Gen4) gets IQ3_XXS/IQ3_S running with no code at all, so build this only if that move is off
the table.
