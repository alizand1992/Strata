# Phase 3 result - the helper binary `strata-expert-helper` (Gate 3)

Plan: [../OPTION_A_EXCLUSIVE_RAM.md](../OPTION_A_EXCLUSIVE_RAM.md), Phase 3. Branch `xhelper/p3-helper`.

**Gate 3 is not closed, and nothing here has run on a GPU.** Gates 1 and 2 are still open too. This code was written
on a Mac with no NVIDIA or AMD toolkit, so `helper_tier.cpp`, `helper_main.cpp` and the parity test have **not been
compiled**: the first HIP build on the box will be their first compile.

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
