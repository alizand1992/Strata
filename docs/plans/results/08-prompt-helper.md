# Phase 7.1 - the helper in prompt processing

Plan: [../OPTION_A_EXCLUSIVE_RAM.md](../OPTION_A_EXCLUSIVE_RAM.md), Phase 7.1 (merged into `main` at `9dbaa13a`). Box: the B550
box of [00-baseline.md](00-baseline.md) (Ryzen 9 3900X, 32 GB DDR4; RTX 5060 Ti 16 GB for the engine, RX 9060 XT 16 GB
on the chipset's x4 Gen3 link for the helper). Date: 2026-10-20.

Config: the live server's (`strata-iq3_s-2gpu.json`): IQ3_S, `--resident-experts --spec 4 --mtp --kv int8`,
`--max-context 262144`, helper exclusive. Both arms use the same build of the branch (`build-p7-cuda`, `build-p7-hip`);
"on" adds `--external-experts-prompt 2048` (and `STRATA_XH_TRACE=1`). One fresh server and one pass of the fixed prompt
set each, same day, one after the other. Greedy.

## Results

| Prompt | off: prompt tok/s | on: prompt tok/s | off: decode | on: decode | off: OS read / request | on: OS read / request |
|---|---|---|---|---|---|---|
| p3 (70 tokens) | 3.9 | 16.3 | 41.2 | 34.5 | 64.4 GB | 52.5 GB |
| p4 (75) | 4.5 | 39.3 | 37.5 | 33.8 | 13.3 GB | 3.1 GB |
| p5 (407, tool use) | 14.7 | 35.3 | 30.1 | 36.1 | 20.8 GB | 3.7 GB |
| p1 (5,182) | 130.4 | 321.7 | 34.5 | 42.1 | 31.2 GB | 11.0 GB |
| p2 (33,537) | 247.1 | 370.1 | 44.2 | 45.7 | 137.8 GB | 45.9 GB |

"OS read" is the engine's `file tier I/O this request` line (what the OS read from the SSD). p3 is the first request
of each fresh server, so its read includes warming up. Decode moves both ways between single passes; this change does
not touch the decode path, and the helper holds 97 fewer experts with it on (8,024 against 8,121: its 187 MiB of prompt
buffers come off the expert budget).

The helper answered every prompt request: 0 timeouts, 0 computed on the CPU instead (p2: 816 requests, 5.96 M rows, 41.6
s waited on the helper of the 90.6 s prompt). The replies read the same and are on task (p1's reasoning explains the
file it was given, p5 makes the same `grep` tool call); they are not bit-identical to the arm without it, as expected
(the helper's MMQ rounding, the sums' order).

## What it says

- Short prompts, which is what an agent's turn reads, are 4-9x faster here (p3-p5), and the SSD reads per request drop
  by 3.6-5.6x on p4/p5. The 4K prompt is 2.5x faster, the 32K one 1.5x.
- What is still read from the SSD is the part of the RAM copy that does not fit at 256K on 32 GB (26.26 GiB wanted,
  22.27 GiB kept) - not the helper's experts.
- At 32K the RTX waits on the helper for about 46% of the prompt time: the helper's MMQ and the x4 Gen3 link set the
  pace there. FP16 transfers or a wider link are what is left to try.

Not tested: the CPU fallback when the helper stalls or exits mid-prompt (no stall happened in these runs), a real
agent session (only the fixed prompt set), and other context sizes.

Since 2026-10-20 the box's own server (`strata-iq3_s-2gpu.json`) runs with `--external-experts-prompt 2048`. Its first
request after the restart (p4) read at 12.3 tok/s with 48 prompt requests and 0 not answered; this first request includes
warming up and is not part of the table above.
