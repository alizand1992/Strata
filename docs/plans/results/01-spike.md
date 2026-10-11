# Phase 1 result - shared-memory spike (Gate 1)

Plan: [../OPTION_A_EXCLUSIVE_RAM.md](../OPTION_A_EXCLUSIVE_RAM.md), Phase 1. Tool: `tools/xhelper_spike/run.sh`.

Run on the plan's box, 2026-10-11. The spike as committed in `ea35974` did not get past test 1 here; three fixes,
in the commit with this sheet, are described under "Changes to the spike" below.

## Run

Box: the B550 Tomahawk MAX box (Ryzen 9 3900X, 32 GB DDR4, Ubuntu 24.04, kernel 7.0.0-38). RTX 5060 Ti 16 GB (driver
580.178.04, `nvcc` 13.0.88, sm_120) on the CPU's lanes
(an x8 card; B550 runs that slot at Gen4 at most); RX 9060 XT 16 GB (gfx1200) behind the B550 chipset, whose uplink is x4 Gen3 (sysfs link table in [00-preflight.md](00-preflight.md)). ROCm 7.10.0a20251120 from Strata's
setup in `.venv` (TheRock `gfx120X-all`), no system ROCm. Nothing else on either GPU.
Commit: `ea35974` plus the fixes below. Date: 2026-10-11.

```sh
R=~/Strata/.venv/lib/python3.12/site-packages/_rocm_sdk_devel
L=~/Strata/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx120X_all/lib
# hipcc from .venv needs the ROCm paths (setup's HIP build passes the same two)
printf '#!/bin/sh\nexec ~/Strata/.venv/bin/hipcc --rocm-path=%s --rocm-device-lib-path=%s/lib/llvm/amdgcn/bitcode "$@"\n' \
  "$R" "$R" > hipcc-venv && chmod +x hipcc-venv
export PATH=/usr/local/cuda/bin:~/Strata/.venv/bin:$PATH ROCM_PATH=$R HIP_PATH=$R LD_LIBRARY_PATH=$R/lib:$L
HIPCC=$PWD/hipcc-venv tools/xhelper_spike/run.sh                    # three times
SPIKE_STAGES=1 HIPCC=$PWD/hipcc-venv tools/xhelper_spike/run.sh     # once more, test 5's stages timed
```

The tables are run 1 of the three. Runs 2 and 3 are in the test 5 table; elsewhere their p50s are within 3 us of
run 1 up to 800 KiB and within 0.3% at 64 MiB.

## Host-only pre-check (measured, on a different machine, so not Gate 1)

`tools/xhelper_spike/spike_doorbell.cpp` - the same memfd, the same mapping layout, the same two doorbell words, a
CPU-only child instead of the HIP half. Box: `ubuntu:24.04` arm64 container on an Apple M3 Max (14 CPUs), `g++ 13.3.0
-O2`, command `./run.sh --host-only --rounds 20000 --spin-us 200`.

| Variant | p50 | p99 | p99.9 | timeouts |
|---|---|---|---|---|
| both spin | 0.25 µs | 0.33 µs | 0.38 µs | 0 |
| child futex, parent spin | 0.25 µs | 0.33 µs | 0.38 µs | 0 |
| both futex (200 µs spin first) | 0.25 µs | 0.29 µs | 0.33 µs | 0 |
| both sleep immediately, no spin | 25.67 µs | 34.25 µs | 38.33 µs | 0 |

- Cross-process wake works: 26 µs, not the 2 ms the waiter's own futex timeout would give. That is the proof that the
  doorbell uses `FUTEX_WAIT`/`FUTEX_WAKE` and not the `_PRIVATE` variants, which cannot wake another process.
- A hand-off that goes through sleep costs ~26 µs on this CPU, which is what the spin-then-sleep waits in Phase 2 are
  there to avoid. No card is in this loop, so it says nothing about the 150 µs target.
- With the child `SIGKILL`ed, the parent's wait returned after 20 ms instead of hanging.

Compile checks of the spike itself, on the same container: `spike_layout.hpp` and `spike_doorbell.cpp` with `g++ 13.3.0
-std=c++20 -Wall -Wextra` (clean); `spike_cuda.cu` with `nvcc 12.8.1 -O3 -std=c++20 -Xcompiler '-Wall,-Wextra'`
(compiles and links, clean, never executed - no driver in the container); `spike_hip.cpp` was **not** built with hipcc
(no ROCm on this machine, no arm64 ROCm image) but builds clean through a throwaway hip-to-cuda shim under `nvcc -x cu`,
which checks syntax and types only.

## 1. Dual registration

| Registration | Result |
|---|---|
| `cudaHostRegister` on the 256 MiB memfd mapping | ok |
| `hipHostRegister` on the same pages | ok |
| `cudaHostGetDevicePointer` alias | ok |
| `hipHostGetDevicePointer` alias | ok |

`memfd_create`, as the plan's first choice; no fallback needed.

If either registration fails: the staged design from the plan's Phase 1 fallback applies - re-run with the child using
`hipHostMalloc` staging buffers and memcpy into the mapping, and measure its round trip too. Record which of
`memfd_create`, `shm_open`, and anonymous `MAP_SHARED` before `fork` was used.

## 2. Integrity

| Direction | GPU checksum | Bad words on the GPU | Bad words on the host |
|---|---|---|---|
| RTX writes region A, RX reads it | matches | 0 | 0 |
| RX writes region B, RTX reads it | matches the HIP checksum | 0 | 0 |
| zero-copy, no memcpy, both aliases | matches both ways | 0 | 0 |

**Gate: 0 bad words in all three rows.** PASS (64 MiB each way, all four runs).

## 3. Bandwidth (50 reps, median and p99)

| Size | Card | Through the shared mapping | Private pinned buffer |
|---|---|---|---|
| 160 KiB H2D | RTX 5060 Ti | 16 / 45 us, 9.17 GiB/s | 16 / 18 us, 9.50 GiB/s |
| 160 KiB D2H | RTX 5060 Ti | 15 / 22 us, 9.81 GiB/s | 15 / 16 us, 10.01 GiB/s |
| 800 KiB H2D | RTX 5060 Ti | 63 / 64 us, 12.11 GiB/s | 63 / 65 us, 12.10 GiB/s |
| 800 KiB D2H | RTX 5060 Ti | 61 / 62 us, 12.47 GiB/s | 61 / 62 us, 12.51 GiB/s |
| 64 MiB H2D | RTX 5060 Ti | 4756 / 4759 us, 13.14 GiB/s | 4756 / 4760 us, 13.14 GiB/s |
| 64 MiB D2H | RTX 5060 Ti | 4687 / 4688 us, 13.33 GiB/s | 4687 / 4697 us, 13.33 GiB/s |
| 160 KiB H2D | RX 9060 XT | 67 / 188 us, 2.19 GiB/s | 67 / 68 us, 2.28 GiB/s |
| 160 KiB D2H | RX 9060 XT | 73 / 80 us, 2.09 GiB/s | 73 / 74 us, 2.09 GiB/s |
| 800 KiB H2D | RX 9060 XT | 278 / 283 us, 2.75 GiB/s | 277 / 277 us, 2.76 GiB/s |
| 800 KiB D2H | RX 9060 XT | 340 / 349 us, 2.24 GiB/s | 342 / 347 us, 2.23 GiB/s |
| 64 MiB H2D | RX 9060 XT | 21591 / 21610 us, 2.89 GiB/s | 21568 / 21581 us, 2.90 GiB/s |
| 64 MiB D2H | RX 9060 XT | 27901 / 28437 us, 2.23 GiB/s | 27972 / 28539 us, 2.23 GiB/s |

p50 / p99. The shared mapping is as fast as a private pinned buffer on both cards (the one outlier, the RX's first
160 KiB H2D p99, is a single slow rep). The RX's ceiling is the chipset link: 2.2-2.9 GiB/s, ~4.5x below the RTX.

## 4. Round-trip latency (10 000 pings)

| Variant | p50 | p99 | p99.9 | timeouts |
|---|---|---|---|---|
| both spin | 0.42 us | 0.52 us | 0.54 us | 0 |
| child futex, parent spin | 0.44 us | 0.53 us | 0.56 us | 0 |
| both futex | 0.44 us | 0.45 us | 0.55 us | 0 |

With the 200 us spin budget the "futex" variants never reach the sleep (each answer comes within a microsecond), so
the three rows are the same; the doorbell costs nothing next to the transfers.

## 5. Round trip with work (80 KiB in, kernel, 200 KiB back)

500 rounds each, 0 wrong results and 0 timeouts in every row.

| Design | Run 1 p50 / p99 | Run 2 | Run 3 |
|---|---|---|---|
| The plan's: `hipMemcpyAsync` H2D, kernel, D2H, sync, checksum read back | 242 / 250 us | 243 / 258 us | 243 / 251 us |
| Zero-copy (`SPIKE_WORK_ZERO`, added here): the kernel reads and writes the mapping through the HIP alias | 149 / 160 us | 149 / 160 us | 149 / 159 us |

Where the plan's round goes (`SPIKE_STAGES=1`, HIP events on the stream, which themselves add ~35 us: that run's p50 was
277 us): memset + 80 KiB H2D 61.5 us, kernel 14.6 us, 200 KiB D2H 114.5 us, the 16-byte checksum read-back (a
synchronous `hipMemcpy`) 50.3 us; the doorbell, 0.4 us. The two copies alone are ~176 us, above the target before any
work. The checksum read-back is the spike's, not the design's, but without it the round would still be ~190 us.

## Gate 1

- Integrity both directions: **PASS**
- Round trip with work p50 < 150 µs: **FAIL** with the plan's copies (242-243 us); zero-copy 149 us, a pass by 1 us
- Round trip with work p99 < 500 µs: **PASS** (250-258 us; zero-copy 159-160 us)
- Decision: **for the owner.** On this slot the copy design cannot meet p50 < 150 us: the x4 Gen3 link moves 280 KiB
  in ~176 us. Zero-copy meets it with no margin, and a real expert kernel reading weights from VRAM while its
  activations come over PCIe would sit at the same ~150 us floor. The plan's stop rule (p99 above 500 us) does not
  fire. Options: continue on this box with zero-copy as the transfer design and accept ~150 us x 48 layers per window;
  run the spike on the X570 box (x8/x8 Gen4, ~4x the link) before deciding; or stop Option A on the B550.
- Owner's decision (2026-10-11): **continue on the B550 with zero-copy** as the transfer design, accepting the
  missing margin. Phase 3's helper already took zero-copy when registration works; its numbers with real experts are
  in [03-helper.md](03-helper.md).

## Changes to the spike

- `spike_cuda.cu`: the parent waited for the HIP child 100 us a step, so 0.12 s in all instead of the 120 s the loop
  counts; HIP takes longer than that to start, and test 1 failed with "the HIP half never got ready". Now 100 ms a step.
- `spike_cuda.cu`: test 4 timed pings in whole microseconds; every ping here is under 1 us, so all rows read 0.0 and the
  verdict (which wants p50 > 0) said FAIL. Now nanoseconds, printed to 0.01 us.
- `spike_cuda.cu`: test 5's input fill copied device to host with `cudaMemcpyHostToDevice`; now
  `cudaMemcpyDeviceToHost`. It did not change a result (the check compares the child's checksum with the output).
- New: `SPIKE_WORK_ZERO`, test 5's zero-copy row, and `SPIKE_STAGES=1` for test 5's per-stage times
  (`Report::work_stage_ns`).
- Building on this box: `hipcc` from `.venv` needs `--rocm-path` and `--rocm-device-lib-path` (the wrapper above).
  Before `libstdc++-14-dev` was installed, ROCm's clang also needed `--gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13`
  (it picks GCC 14, whose C++ headers were missing); the numbers above are from the build after the install.
