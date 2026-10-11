# Phase 4 result - the engine side, `ExternalExperts` (Gate 4)

Plan: [../OPTION_A_EXCLUSIVE_RAM.md](../OPTION_A_EXCLUSIVE_RAM.md), Phase 4. Branch `xhelper/p4-run`. Box: the B550
box of [00-baseline.md](00-baseline.md): RTX 5060 Ti 16 GB (CUDA engine) and RX 9060 XT 16 GB (helper, chipset x4
Gen3). Date: 2026-10-11.

**Gate 4: PASS.**

## What is in it

| File | What |
|---|---|
| `include/strata/core/external_experts.hpp`, `src/core/external_experts.cpp` | memfd + mapping, header, the offer (ranked pairs the primary does not hold), `posix_spawn` of the helper with `HIP_VISIBLE_DEVICES=N` and an empty `CUDA_VISIBLE_DEVICES`, wait for Ready (a note every 10 s, 120 s limit). `begin` classifies as `RemoteExperts::begin` does, then writes one request; `finish` waits on the response (spin, then futex, checking every 100 ms that the helper process still exists) and scatters the rows. A timeout, an error answer, a row nobody asked for or a dead helper fail that layer once, log `the helper stopped (...)`, and from then on `begin` owns nothing. In the HIP build and off Linux, `open` fails with a message |
| `include/strata/core/expert_source.hpp`, `src/core/expert_source.cpp` | `ExpertDispatch::external`, asked after the remote tiers in the one-token path (begin, skip owned rows before any blob read, finish) and the window path (kept out of the PCIe share, begin with `kind = 2`, finish). `RouterLookahead::set_skip`: a mask of experts never warmed |
| `src/program/generate.cpp` | `--external-experts PATH`, `--external-experts-device N`, `--external-experts-gib X`, `--external-experts-ram exclusive\|inclusive`; env `STRATA_XH_SPIN_US`, `STRATA_XH_TIMEOUT_MS`, `STRATA_XH_TRACE`. Refused with one line each: with `--expert-cache-device1..3`, `--remote-expert-opt`, `--peer-device`, `--layer-split`, `--batch`, `--pipeline-windows 2`, `--vram-elastic`, `--kv-grow`; without `--expert-profile` or the pool; in a HIP build; off Linux. The helper opens after the remote tiers, before the RAM copy. Exclusive mode adds its pairs to the ones the RAM copy leaves out. The adaptive tier never promotes an expert the helper loaded (unconditionally, also after the helper stops). The look-ahead skips them. A per-request `external:` stats line |
| `tests/xhelper/external_experts_test.cpp` | ctest `xhelper_external_experts_test` (CUDA build, Linux; exit 77 without `STRATA_TEST_PACK`, `STRATA_TEST_NATIVE`, `STRATA_TEST_HELPER`) |

Not done from the plan's list: the engine does not `cudaHostRegister` the mapping. Only the CPU reads and writes it on
the engine side, so registering would not be used.

Both backends build: CUDA (`strata`, the test) and HIP (`strata`, `strata-expert-helper`, the parity test).

## Test

IQ3_XXS pack (as in [03-helper.md](03-helper.md)), helper from the HIP build:

```
external_experts_test: helper pid 137012, 64 of 64 pairs loaded (0.12 GiB) in 0.7 s
external_experts_test: 200 requests, 4439 helper rows checked (worst rel L1 0.01938, bar 3e-02), 4591 rows left to the CPU untouched; begin 1.3 ms, wait 93.0 ms
external_experts_test: after kill -9: finish failed (external experts: the helper process exited (layer 0)), alive 0
PASS
```

The bar is 3e-2, not the plan's 5e-3, for the reason in 03-helper.md.

## Gate 4 run

Setup's Coder config (`strata-coder-iq1_m.json`: IQ1_M, `--resident-experts`, `--spec 4`, `--mtp`, 64K context, int8
K/V) with the p4 engine and `--external-experts build-hip/strata-expert-helper --external-experts-device 0
--external-experts-ram inclusive`, `STRATA_XH_TRACE=1`, through the server. ROCm's library dirs were added to
`lib_dirs` for the helper.

```
[helper pid 139312] loaded 7906 of 7906 offered experts (15.08 GiB of device slots), reading the mapping directly
strata generate: external helper (HIP device 0): 7906 experts, 15.08 GiB, loaded in 35.4 s; with CUDA0's 4382 that is 12288 of 12288 on the GPUs; RAM copy inclusive (keeps them too)
```

The Coder has 256 experts per layer, so the two cards hold every expert: the CPU computes no expert rows.

The prompt set four times, 20 requests, one after another: all answered, a helper `external:` line on every request,
0 timeouts, 0 file reads. Decode tok/s (prompt tok/s on p1/p2), against 0a (same config without the helper, Phase
0, same day):

| Prompt | 0a | pass 1 | pass 2 | pass 3 | pass 4 |
|---|---|---|---|---|---|
| p1-code-4k | 52.8 (1547.9) | 74.1 (1495.1) | 75.0 (1762.3) | 74.9 (1768.0) | 74.9 (1767.5) |
| p2-code-32k | 53.0 (1797.6) | 71.9 (1799.5) | 81.5 (1851.7) | 73.4 (1850.7) | 74.1 (1848.6) |
| p3-chat-short | 49.9 | 74.7 | 73.5 | 75.6 | 73.9 |
| p4-chat-short | 55.5 | 85.4 | 87.9 | 85.5 | 85.8 |
| p5-tool-use | 44.1 | 91.2 | 94.4 | 93.8 | 87.4 |

Decode is 1.4-2.1x 0a's. Prompt speed is unchanged: inclusive mode keeps the RAM copy, and the prompt path does not
use the helper. These are one Gate 4 run, not Phase 6's measurement. The replies are not token-identical to 0a's
(the helper's rows are GPU rows, rounded differently from the CPU's), so decoded lengths differ.

Per-request helper lines (pass 1; passes 2-4 within a few percent):

| Prompt | rows | layers | wait ms | round trip p50 / p99 us |
|---|---|---|---|---|
| p1 | 126,515 | 10,603 | 2245.5 | 198 / 393 |
| p2 | 101,654 | 11,030 | 2082.7 | 177 / 402 |
| p3 | 54,224 | 5,230 | 1015.0 | 186 / 376 |
| p4 | 131,386 | 6,222 | 1672.4 | 284 / 464 |
| p5 | 17,959 | 1,055 | 266.5 | 254 / 410 |

Almost every layer waited on the helper: with all experts on a GPU, the CPU pool has nothing to compute, so the
helper's round trip is the layer's whole expert time beyond the RTX's. `0 exchanged with the VRAM tier`: the
adaptive tier has no candidates, because every expert the RTX lacks is on the helper, and those are never promoted.
