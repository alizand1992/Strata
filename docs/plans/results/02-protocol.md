# Phase 2 result - the protocol header (Gate 2)

`include/strata/xhelper/protocol.hpp`, `tests/xhelper/protocol_test.cpp`, registered in ctest in `CMakeLists.txt`.
Branch `xhelper/p2-protocol`.

Gate 2 says: **the test passes under ctest in the CUDA build and in the HIP build.** Half of that is measured here,
half needs the box the plan targets. The first sections are from the machines available while writing; "Gate 2"
below is the run on the box.

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

**Closed on 2026-10-10**, on the plan's box: B550 Tomahawk MAX, Ryzen 9 3900X, 32 GB, Ubuntu 24.04, kernel
7.0.0-38. Branch `xhelper/p3-helper` at `93f9376` (the test and header are unchanged from `xhelper/p2-protocol`).

**CUDA build** (CUDA 13.0.88, g++ 13.3.0, setup's flags plus tests):

```sh
cmake -G Ninja -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES=120 -DCMAKE_CUDA_COMPILER=/usr/local/cuda-13.0/bin/nvcc \
      -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
cmake --build build-cuda --target xhelper_protocol_test
ctest --test-dir build-cuda -R xhelper_protocol_test --output-on-failure
    1/1 Test #8: xhelper_protocol_test ............   Passed    1.07 sec
```

**HIP build** (ROCm 7.10.0a20251120 from TheRock wheels in `.venv`, as setup installs it; AMD clang 22.0.0git; setup's
`build_engine_hip` flags plus tests):

```sh
cmake -G Ninja -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
      -DSTRATA_BUILD_TESTS=ON -DSTRATA_PREFILL_MMQ=ON -DCMAKE_HIP_ARCHITECTURES=gfx1200 \
      -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R -DCMAKE_PREFIX_PATH="$R;$L" \
      -DCMAKE_HIP_FLAGS="--rocm-path=$R --rocm-device-lib-path=$R/lib/llvm/amdgcn/bitcode \
                         --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13" \
      -DSTRATA_GGML_DIR=$PWD/third_party/llama.cpp
#   R = .venv/lib/python3.12/site-packages/_rocm_sdk_devel, L = ..._rocm_sdk_libraries_gfx120X_all
cmake --build build-hip --target xhelper_protocol_test
ctest --test-dir build-hip -R xhelper_protocol_test --output-on-failure
    1/1 Test #6: xhelper_protocol_test ............   Passed    0.94 sec
```

`--gcc-install-dir` is needed on this box and is **not** something setup passes: clang 22 picks
`/usr/lib/gcc/x86_64-linux-gnu/14`, which has no C++ headers here (only `libstdc++-13-dev` is installed), and CMake's HIP
compiler check then fails with "Could not find standard C++ header 'cmath'". So `./setup.sh --backend hip` would fail
the same way on this box. The other fix is `sudo apt install libstdc++-14-dev`. This is a setup issue, separate from
this plan.

Numbers from the test binary on this box (3900X; the protocol test has no GPU in it):

| Case | Build | p50 | p99 | p99.9 |
|---|---|---|---|---|
| transport alone, one row each way, 20000 rounds | CUDA | 1.94 µs | 2.54 µs | 5.79 µs |
| | HIP | 0.88 µs | 2.54 µs | 5.71 µs |
| full request, up to 80 rows of 2560 floats, 20000 rounds | CUDA (3 runs) | 16.0-21.5 µs | 68.3-79.9 µs | 80.6-89.5 µs |
| | HIP | 15.55 µs | 48.56 µs | 79.80 µs |
| same, both sides parked in the futex, no spin, 2000 rounds | CUDA (3 runs) | 22.8-26.5 µs | 66.9-85.2 µs | 85.0-89.2 µs |
| | HIP | 51.56 µs | 90.59 µs | 105.13 µs |

0 timeouts and 0 mismatches in every row. With no partner the wait gave up after 151.3-151.4 ms of the 150 ms window.
Both builds compile this file with the same host compiler (g++ 13.3, the HIP build's `CMAKE_CXX_COMPILER` is
`/usr/bin/c++`), so the CUDA and HIP rows differ only by run-to-run noise. The llama-server described in
`00-preflight.md` was running on ~6 cores during these runs.

## Not tested

Nothing device-side: no memfd, no `hipHostRegister`, no VRAM, no helper process. The two-process form of this exact
doorbell is `tools/xhelper_spike/spike_doorbell.cpp` (branch `xhelper/p1-shm-spike`), and the GPU half of Gate 1 is
still open.
