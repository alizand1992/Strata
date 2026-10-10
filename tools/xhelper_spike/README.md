# `tools/xhelper_spike` - Option A Phase 1: does the transport work at all?

Phase 1 of [docs/plans/OPTION_A_EXCLUSIVE_RAM.md](../../docs/plans/OPTION_A_EXCLUSIVE_RAM.md). Nothing in the engine
is touched and nothing here is used by the engine. Two throwaway programs prove the shared-memory channel between a
CUDA process on the RTX and a HIP process on the RX 9060 XT, because if that channel is slow or `hipHostRegister`
refuses pages `cudaHostRegister` already took, the rest of the plan should not be built.

| File | Built by | Role |
|---|---|---|
| `spike_layout.hpp` | both | the mapping both sides agree on: the control block, the two 64 MiB regions, the 80 KiB / 200 KiB work regions, the child's report block, and the two doorbell words with the spin/futex wait |
| `spike_cuda.cu` | `nvcc` | the parent: makes the memfd, maps 256 MiB, registers it with CUDA, forks the child with the fd, runs the five tests, prints the tables |
| `spike_hip.cpp` | `hipcc --offload-arch=gfx1200` | the child: maps the same pages, registers them with HIP, answers commands, measures its own card |
| `spike_doorbell.cpp` | plain `g++` | the same mapping and the same two doorbell words with no GPU in the loop: memfd, MAP_SHARED, cross-process futex, and a wait that must give up when its partner is dead |
| `run.sh` | - | builds and runs them |

## Run it

On the box the plan names (Ubuntu, RTX 5060 Ti as primary, RX 9060 XT as the helper card), with `nvcc` and `hipcc`
both installed:

```sh
cd tools/xhelper_spike
./run.sh                       # defaults: 10000 pings, 500 work rounds, gfx1200
./run.sh --hip-arch=gfx1201    # the other AMD card (rocminfo is read for the arch when it is present)
./run.sh --rounds 30000        # a better tail
./run.sh --parent-only         # build and run just the CUDA half, to check this script before fighting hipcc
./run.sh --host-only           # spike_doorbell.cpp only: no nvcc, no hipcc, no GPU
NVCC=/usr/local/cuda/bin/nvcc HIPCC=/opt/rocm/bin/hipcc ./run.sh
```

`--cuda-arch` and `--hip-arch` override the arch; without them `run.sh` asks `nvidia-smi --query-gpu=compute_cap` and
`rocminfo` what the cards in the box are.

The tables go to stdout and to `results-<date>.txt` (gitignored). Paste them into
`docs/plans/results/01-spike.md` with the commit, the box and the exact command.

`run.sh` builds outside CMake on purpose: the plan says the spike must not enter the main build, and the two halves
need two different compilers on one box.

## What each test answers

1. **Dual registration** - `cudaHostRegister` and `hipHostRegister` on the *same* 256 MiB memfd mapping. If either
   fails, the whole "kernels on both cards read one mapping" idea is replaced by the plan's staged design (the child
   registers its own `hipHostMalloc` buffers and memcpy's 80-800 KiB per layer), so this line is read first.
2. **Integrity** - 64 MiB written by the NVIDIA card into region A, read by the HIP card with `hipMemcpyAsync` and
   checked on its GPU, then the reverse in region B, then once more with both sides touching the mapping through
   `cudaHostGetDevicePointer` / `hipHostGetDevicePointer` and no memcpy at all. The payload is an exact integer
   pattern, so a mismatch count of 0 means byte-equal; the parent also re-checks each region on the host, which is
   independent of both cards.
3. **Bandwidth** - 160 KiB, 800 KiB and 64 MiB, 50 reps each way, p50/p99 and GiB/s, from each card, each against its
   own private pinned buffer. This is the number that says whether expert fill traffic through the mapping is
   respectable or pathological on the x4 Gen3 link.
4. **Round-trip latency** - 10 000 pings on two `std::atomic<uint32_t>` words in the mapping, woken with
   `syscall(SYS_futex, ..., FUTEX_WAIT/FUTEX_WAKE)` and **no `FUTEX_PRIVATE_FLAG`** (a private futex is keyed to one
   process's address space and would wake nobody in the other). Three variants: both spin, child futex + parent spin,
   both futex. Median, p99, p99.9, plus timeouts.
5. **Round trip with work** - the real shape of one layer: parent writes 80 KiB, the child copies it to VRAM, runs a
   kernel, copies 200 KiB back, signals; the parent re-reads the 200 KiB and compares the sum the child computed on
   its GPU. Median and p99.

## Gate 1

- Integrity passes in both directions (0 bad words, matching GPU checksums).
- Round trip with work: **p50 < 150 µs, p99 < 500 µs** with the spin variant.

`spike_cuda.cu` prints a PASS/FAIL line under each. If latency fails, the plan stops here: 48 layers x that cost per
window cannot beat the CPU pool.

## The host-only row, and what it showed

`spike_doorbell.cpp` proves the plumbing that does not need a card. Run first, because if the futex word does not wake
across the process boundary the GPU half cannot work either.

Measured on the machine this was written on - `ubuntu:24.04` (arm64) under Docker on an Apple M3 Max, 14 CPUs, `g++
13.3.0 -O2`,
20 000 pings, `--spin-us 200`, no GPU in the loop (`./run.sh --host-only --rounds 20000`):

| Variant | p50 | p99 | p99.9 | timeouts |
|---|---|---|---|---|
| both spin | 0.25 µs | 0.33 µs | 0.38 µs | 0 |
| child futex, parent spin | 0.25 µs | 0.33 µs | 0.38 µs | 0 |
| both futex (200 µs spin first) | 0.25 µs | 0.29 µs | 0.33 µs | 0 |
| both sleep immediately, no spin | 25.67 µs | 34.25 µs | 38.33 µs | 0 |

- The last row is the one that matters: 26 µs, not 2 ms, so `FUTEX_WAKE` really crossed the process boundary. With
  `FUTEX_PRIVATE_FLAG` it would have been about 2000 µs, which is the mistake this row is for.
- It is also why the waits in this design spin first: sleeping costs about 26 µs per hand-off on this CPU, spinning is
  free, and a real helper's device work is slower than either.
- After `SIGKILL`ing the child, the parent's wait gave up after 20 ms instead of hanging.
- These are not Gate 1 numbers: no card is in the loop. Gate 1's p50 < 150 µs has to come from the real box.

## What has been checked, and where

- `spike_layout.hpp`: compiles clean with `g++ 13.3.0 -std=c++20 -Wall -Wextra` in `ubuntu:24.04` (arm64); the layout's
  `static_assert` hold, the report block fits, `spike_wait` times out as intended with no partner.
- `spike_cuda.cu`: compiles **and links** with `nvcc 12.8.1 -O3 -std=c++20 -Xcompiler '-Wall,-Wextra'`, no warnings.
  It has never run: the container has no NVIDIA driver, so no CUDA call has been executed. `-arch` was left at the
  default (compute_50) in that check.
- `spike_hip.cpp`: **not compiled with hipcc.** The machine it was written on has no ROCm and there is no arm64 ROCm
  image to borrow, so it was syntax-checked through a throwaway hip-to-cuda shim (`nvcc -x cu` with a local header that
  renames `hip*` to `cuda*`), which builds clean. That catches syntax, formats and types, not HIP semantics: `hipSetDevice`,
  `hipHostRegister` on pages CUDA registered, `hipHostGetDevicePointer` and the kernels have still to be proven on the
  9060 XT.
