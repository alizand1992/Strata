# Phase 2 result - the protocol header (Gate 2)

`include/strata/xhelper/protocol.hpp`, `tests/xhelper/protocol_test.cpp`, registered in ctest in `CMakeLists.txt`.
Branch `xhelper/p2-protocol`.

Gate 2 says: **the test passes under ctest in the CUDA build and in the HIP build.** Half of that is measured here,
half needs the box the plan targets. What follows is what was actually run, on the machines available while writing.

## What is in the header

| Region | Size | Note |
|---|---|---|
| `Header` | 2688 B | magic `STRKHLP1`, version 1, model geometry, both states, the helper's heartbeat, `vram_budget_bytes`, `model_pack`, `native_preset`, `error`, and the four offsets below |
| `Pair[n_layers*n_expert]` assigned | 196608 B for 48 x 512 | the engine's offer, hottest first |
| `Pair[n_layers*n_expert]` loaded | 196608 B | what the helper actually holds, and it is the answer |
| `Request` | 82624 B | `seq` + `helper_sleeping` + layer/n_tok/n_entries + `entry[80]` + `expert[80]` + `x[8*2560]` |
| `Response` | 819584 B | `seq` + `engine_sleeping` + status/n_rows + `entry[80]` + `rows[80*2560]` |
| **whole mapping** | **1298112 B = 1.24 MiB** | both sides allocate `kMappingBytes` = 4 MiB |

`Header`, `Request` and `Response` are each a whole number of 64-byte lines and each region starts on one. `kH` 2560,
`kMaxTok` 8, `kK` 10, `kCap` 80 are asserted against `strata::kernels::cpu::H` and `MAXT` in the test, which is the
engine-side TU the plan names; `kCap == MAXT * 10` is the remote tier's `CAP` in `src/core/remote_experts.cpp:18`.

Publish and wait are the plan's: payload first, then `seq.store(release)`; the reader spins `spin_us` (200 default) and
only then `FUTEX_WAIT`/`FUTEX_WAKE` - never the `_PRIVATE` variants, which are keyed to one process and would wake
nobody - and `FUTEX_WAKE` is skipped unless the other side's `*_sleeping` flag says it is parked. `header_ok` is the
helper's first act after `mmap`; `helper_alive` is the heartbeat rule the engine will use (older than 5 s is dead).

## Gate 2: what was run

**CUDA build.** `nvidia/cuda:12.8.1-devel-ubuntu24.04` (arm64) on an Apple M3 Max - CMake 3.28.3, `nvcc 12.8.1`,
`-DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_CUDA_ARCHITECTURES=120` (sm_120, the RTX 5060 Ti the plan
targets) `-DSTRATA_NATIVE_EXPERTS=OFF` (the ggml checkout would otherwise be fetched from GitHub and the container has
no git; the protocol test does not use it). No NVIDIA driver in the container, so nothing GPU-related was compiled or
run for this test - it has no CUDA in it.

```
cmake --build /tmp/bcuda --target xhelper_protocol_test -j4   # built clean
ctest -R xhelper_protocol_test --output-on-failure
    Start 8: xhelper_protocol_test
1/1 Test #8: xhelper_protocol_test ............   Passed    0.51 sec
100% tests passed, 0 tests failed out of 1
```

**HIP toolchain.** `rocm/rocm-terminal:latest` (ROCm 6.4.0, amd64 under emulation on the same Mac):

```
hipcc -std=c++20 -O2 -Wall -Wextra -Iinclude -c tests/xhelper/protocol_test.cpp -o /tmp/proto_test_hip.o   # rc=0, no diagnostics
/opt/rocm-6.4.0/lib/llvm/bin/clang++ -std=c++20 -O2 -Iinclude tests/xhelper/protocol_test.cpp -o /tmp/proto_test_run -pthread
/tmp/proto_test_run   # PASS (0 failed)
```

The full `hipcc` **link** was tried and died in `amdgcn-link` with a bus error - that is the emulated x86_64 container,
not the code: the object compiled first, clean. **What has not been done is a configure through `cmake/hip_backend.cmake`**
(that path wants `rocminfo` and a gfx arch, so it needs the 9060 XT) and the test run inside it.

## Numbers the test produced

Round trip measured by the engine side, both halves in one process on one mapping (arm64 container, `g++ 13.3.0 -O2`;
the parenthesised row is the same test compiled by ROCm 6.4's clang++ on emulated x86_64):

| Case | p50 | p99 | p99.9 |
|---|---|---|---|
| transport alone: one row each way, 20000 rounds | 1.17 µs | 1.62 µs | 6.46 µs |
| full request: up to 80 rows of 2560 floats answered, 20000 rounds | 28.04 µs | 55.21 µs | 63.58 µs (7.79 / 18.00 / 21.54) |
| same, both sides parked in the futex with no spin budget, 2000 rounds | 60.42 µs | 99.50 µs | 129.08 µs (40.58 / 54.42 / 59.12) |

- 0 timeouts, 0 mismatches in all three: every row the engine asked for came back with the entry index it was asked
  under.
- The middle row is the honest shape of one layer's helper call: the answer costs the arithmetic (80 x 2560 multiply
  and add on the responder), not the hand-off. The first row is the hand-off alone.
- The last row is the cost of parking: about 26 µs per hand-off, which matches
  `tools/xhelper_spike/spike_doorbell.cpp` measured on the same machine, and is why `spin_us` exists.
- With no partner at all the wait returned false after 152.8 ms of a 150 ms window instead of hanging.

## Gate 2

**Not closed.** The test is in ctest and green in a CUDA configure on the plan's arch, and the same file compiles clean
under `hipcc 6.4` and runs green when linked by that toolchain's host compiler. What is missing is the HIP **build**
(`-DSTRATA_ENABLE_HIP=ON` through `cmake/hip_backend.cmake`) on the box with the 9060 XT. Run there:

```sh
cmake -S . -B build-hip -DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_TESTS=ON -DCMAKE_HIP_ARCHITECTURES=gfx1200
cmake --build build-hip --target xhelper_protocol_test -j
ctest --test-dir build-hip -R xhelper_protocol_test --output-on-failure
```

## Not tested

Nothing device-side: no memfd, no `hipHostRegister`, no VRAM, no helper process. The two-process form of this exact
doorbell is `tools/xhelper_spike/spike_doorbell.cpp` (branch `xhelper/p1-shm-spike`), and the GPU half of Gate 1 is
still open.
