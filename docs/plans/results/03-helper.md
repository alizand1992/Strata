# Phase 3 result - the helper binary `strata-expert-helper` (Gate 3)

Plan: [../OPTION_A_EXCLUSIVE_RAM.md](../OPTION_A_EXCLUSIVE_RAM.md), Phase 3. Branch `xhelper/p3-helper`.

**Gate 3: PASS on the B550 box (2026-10-11),** with the parity bar changed from 5e-3 to 3e-2 (why below). The
results from the box are in "Run on the box" at the end. The sections before it describe the code as it was written,
on a Mac, before its first compile.

## What is in it

| File | What |
|---|---|
| `src/xhelper/helper_main.cpp` | `--shm-fd N [--spin-us N] [--verbose]`; mmap, `header_ok`, Loading, `FileExpertSource`, `HelperTier::open`, publish loaded pairs, Ready, request loop, Exited. Exit 2 bad mapping, 3 model/device failure, 0 clean stop |
| `src/xhelper/helper_tier.hpp/.cpp` | `RemoteExperts` open/begin/finish for device 0, with an explicit pair list and no classification. Zero-copy when `cudaHostRegister` accepts the shared pages, staged pinned copies when it does not |
| `tests/xhelper/helper_tier_parity.cpp` | ctest `xhelper_helper_tier_parity` (HIP build, Linux): 32 pairs over layers {0, mid, last}, every returned row against the CPU pool's native kernels, rel L1 < 5e-3. Exit 77 (skipped) without `STRATA_TEST_PACK` or with a non-native pack |
| `tools/xhelper_spike/drive_helper.cpp` | CPU-only engine stand-in: memfd, offer pairs, fork the helper, wait for Ready, N requests with round-trip p50/p99, Shutdown, check exit 0 + `Exited` |
| `CMakeLists.txt` | `strata-expert-helper` and the parity test, only with `STRATA_ENABLE_HIP` on Linux |

## What was checked here

- `drive_helper.cpp`: built clean with `g++ 13.3 -std=c++20 -O2 -Wall -Wextra` (ubuntu:24.04 arm64 container), and
  with `--helper /bin/false` reports "the helper exited before it was ready" and exits 1.
- Every engine API the helper calls exists with the signature used (`ExpertCache::open_sized/admit/fill_slot_blocking/
  verify_slot/device_slot`, `native_expert_grouped`, `moe_grouped_s2`, the scratch-size functions, the CPU
  `native_*` kernels), checked by reading the headers, not by a compile.

Fixed before commit (found by review, not by a run): `ignore()` was used before its declaration (a compile error);
the zero-copy branch of `HelperTier::run` returned false on every request; zero-copy ignored a caller's own `x`/`rows`
buffers (the parity test's); `drive_helper` gave the helper no time to stop on Shutdown before SIGTERM.

## To run on the box (after Gates 1 and 2)

```sh
cmake -S . -B build-hip -DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1200
cmake --build build-hip --target strata-expert-helper xhelper_helper_tier_parity -j
HIP_VISIBLE_DEVICES=<amd> STRATA_TEST_PACK=<pack> STRATA_TEST_NATIVE=<gguf> \
    ctest --test-dir build-hip -R xhelper_helper_tier_parity --output-on-failure
g++ -O2 -std=c++20 -Iinclude tools/xhelper_spike/drive_helper.cpp -o build-hip/drive_helper
HIP_VISIBLE_DEVICES=<amd> build-hip/drive_helper --helper build-hip/strata-expert-helper \
    --pack <pack> --native <gguf> --pairs 8000 --budget-gib 12 --requests 500 --verbose
```

Gate 3 passes when the parity test prints PASS on the 9060 XT, and `drive_helper` shows ~12 GB loaded, Ready, all
requests answered, and PASS (helper exit 0, state Exited). Record the load time, the round trip p50/p99, and whether
the helper ran zero-copy or staged.

## Run on the box (2026-10-11)

B550 box: RX 9060 XT 16 GB (gfx1200) on the chipset's x4 Gen3 link, setup's ROCm 7.10 from `.venv` (the build flags
are in [00-baseline.md](00-baseline.md), 0d). Model: the IQ3_XXS GGUF (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS`, shard 1 as
`--native`), pack `~/Strata-data/packs/iq3_xxs` (`tools/iq_pack.py`, experts read from the GGUF in place). Gate 1 was
decided for zero-copy ([01-spike.md](01-spike.md)).

### Fixes needed to run

- `helper_main.cpp` and the parity test opened `FileExpertSource` without loading the pack's expert layout first, and
  `open` refused: "the requested geometry does not match the loaded expert layout". Both now call
  `expert_layout_load(pack, n_layers, n_expert)` first, as the engine does (`src/program/generate.cpp`, the "Plan v0.3
  P6" block). The plan's Startup step 3 said `open` initialises the layout; it does not.
- The parity bar. With 5e-3 every row failed at 1.3e-2 .. 2.0e-2. The repo's own `native_expert_parity` on the same
  model measures the engine's CPU path vs its GPU path at 1.45e-2 (layer 0, iq2_xs/iq4_nl), 1.56e-2 (1, iq2_xxs/q2_0),
  1.51e-2 (24, iq3_s/q2_0) and 1.70e-2 (47, iq3_xxs/iq4_nl), and accepts each path within 3e-2 of the float expert.
  The "few 1e-3" the plan quoted holds for other formats, not these low-bit ones. The test now uses 3e-2. (That tool
  also reports 2 failures on this model, both in the prompt path's fp16 dequant check on the q2_0 layers 1 and 24;
  they have nothing to do with the helper.)
- `STRATA_XHELPER_STAGED=1` (new) forces the staged path where zero-copy works, for the A/B below.
- `drive_helper` prints the round trip per request size (1..8 tokens), not only overall.

### Parity

`ctest -R xhelper`: `xhelper_protocol_test` and `xhelper_helper_tier_parity` pass. 32 of 32 pairs loaded, 103 rows
checked, worst relative L1 0.01971 (bar 3e-2). The worst is the same with `STRATA_XHELPER_STAGED=1`, so zero-copy
changes nothing in the rows.

### Load, requests, shutdown

`drive_helper --pairs 9000 --budget-gib 12 --requests 2000`: the helper loaded **7380 of 9000** offered experts
(12.00 GiB of device slots; 15.86 GiB free at start) in 6.7-7.3 s (24.8 s on the first run, when the shard was not in
the file cache), reported Ready, answered all 2000 requests, and exited 0 with state Exited on Shutdown: **PASS**.

The driver's requests use distinct experts for each entry, so a request with n tokens computes 10 x n rows and reads
10 x n experts (1.3-2.2 MB each) from VRAM. That is the worst case for a window. In the engine the helper gets only
the CPU candidates it holds.

Round trip p50 (p99) in us, per request size. The zero-copy column is the final build. The staged column is one run
with `STRATA_XHELPER_STAGED=1`. The last column is zero-copy with the group metadata read by the kernels from mapped
pinned memory instead of copied (two runs, tried and reverted):

| Tokens / rows | Zero-copy (final) | Staged copies | Zero-copy, metadata mapped |
|---|---|---|---|
| 1 / 10 | 187.9 (207.7) | 232.1 (266.3) | 198.0-199.0 |
| 2 / 20 | 269.2 (289.5) | 347.3 (385.3) | 285.5-287.0 |
| 4 / 40 | 412.4 (481.2) | 580.4 (687.1) | 440.5-442.2 |
| 8 / 80 | 775.2 (845.8) | 1125.7 (1222.7) | 824.0-827.4 |
| all | 494.7 (841.3) | 724.3 (1197.0) | 547.0-548.3 |

Two more zero-copy runs with the metadata copied matched the final build within 2 us at every size.

### What it says

- Zero-copy saves 44 us on a one-token request and 350 us on a full window, mostly the D2H of the rows: 800 KiB at
  ~2.4 GB/s.
- With zero-copy, each extra 10 rows costs about 75-90 us. That is about 7.5-9 us per row, each reading a 1.3-2.2 MB
  expert. This is the 9060 XT's own memory bandwidth, not the link. The fixed cost of a request (wake, input, one
  metadata copy, launches, synchronize) is about 110 us.
- Reading the 1.6 KB of group metadata across the link is slower than one copy of it: every block of every kernel reads
  it. The copy stays.
- For the engine, the question is whether the helper's rows finish before the CPU pool's do. On this box one AVX-2
  thread takes ~530 us per expert row for one token (`native_expert_parity`, layer 47: gate+up 336 us, down 197 us),
  so the helper's ~8 us per row is much faster per row. The fixed ~110-190 us per layer is what Phase 6 will show.
