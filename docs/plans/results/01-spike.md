# Phase 1 result - shared-memory spike (Gate 1)

Plan: [../OPTION_A_EXCLUSIVE_RAM.md](../OPTION_A_EXCLUSIVE_RAM.md), Phase 1. Tool: `tools/xhelper_spike/run.sh`.

**Not run yet.** Nothing in this file is a measurement: the spike needs the plan's box (Ubuntu, RTX 5060 Ti primary,
RX 9060 XT helper, `nvcc` and `hipcc`), and the machine the spike was written on has neither toolkit. Fill the tables
below from a real run and delete this line.

## Run

```sh
git rev-parse --short HEAD                    # the commit the spike was built from
nvidia-smi --query-gpu=name,pci.bus_id,memory.total,driver_version --format=csv
sudo lspci -vv | grep -E "VGA|LnkSta:"        # both cards' link speed and width
free -g; uname -r; /opt/rocm/bin/hipcc --version; nvcc --version
cd tools/xhelper_spike && ./run.sh 2>&1 | tee ../plans-results-01.txt
```

Box used: ______ (which board: B550 box with the 9060 XT in a chipset x4 Gen3 slot, or the X570 x8/x8 Gen4 box)
Commit: ______ Date: ______

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
| `cudaHostRegister` on the 256 MiB memfd mapping | ______ |
| `hipHostRegister` on the same pages | ______ |
| `cudaHostGetDevicePointer` alias | ______ |
| `hipHostGetDevicePointer` alias | ______ |

If either registration fails: the staged design from the plan's Phase 1 fallback applies - re-run with the child using
`hipHostMalloc` staging buffers and memcpy into the mapping, and measure its round trip too. Record which of
`memfd_create`, `shm_open`, and anonymous `MAP_SHARED` before `fork` was used.

## 2. Integrity

| Direction | GPU checksum | Bad words on the GPU | Bad words on the host |
|---|---|---|---|
| RTX writes region A, RX reads it | ______ | ______ | ______ |
| RX writes region B, RTX reads it | ______ | ______ | ______ |
| zero-copy, no memcpy, both aliases | ______ | ______ | ______ |

**Gate: 0 bad words in all three rows.** ______

## 3. Bandwidth (50 reps, median and p99)

| Size | Card | Through the shared mapping | Private pinned buffer |
|---|---|---|---|
| 160 KiB H2D | RTX 5060 Ti | ______ | ______ |
| 160 KiB D2H | RTX 5060 Ti | ______ | ______ |
| 800 KiB H2D | RTX 5060 Ti | ______ | ______ |
| 800 KiB D2H | RTX 5060 Ti | ______ | ______ |
| 64 MiB H2D | RTX 5060 Ti | ______ | ______ |
| 64 MiB D2H | RTX 5060 Ti | ______ | ______ |
| 160 KiB H2D | RX 9060 XT | ______ | ______ |
| 160 KiB D2H | RX 9060 XT | ______ | ______ |
| 800 KiB H2D | RX 9060 XT | ______ | ______ |
| 800 KiB D2H | RX 9060 XT | ______ | ______ |
| 64 MiB H2D | RX 9060 XT | ______ | ______ |
| 64 MiB D2H | RX 9060 XT | ______ | ______ |

## 4. Round-trip latency (10 000 pings)

| Variant | p50 | p99 | p99.9 | timeouts |
|---|---|---|---|---|
| both spin | ______ | ______ | ______ | ______ |
| child futex, parent spin | ______ | ______ | ______ | ______ |
| both futex | ______ | ______ | ______ | ______ |

## 5. Round trip with work (80 KiB in, kernel, 200 KiB back)

| Metric | Value |
|---|---|
| rounds | ______ |
| p50 | ______ µs |
| p99 | ______ µs |
| wrong results | ______ |
| timeouts | ______ |

## Gate 1

- Integrity both directions: PASS / FAIL ______
- Round trip with work p50 < 150 µs: PASS / FAIL ______
- Round trip with work p99 < 500 µs: PASS / FAIL ______
- Decision: continue to Phase 2 / stop and report ______
