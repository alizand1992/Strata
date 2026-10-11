# Phase 6 result - measurement and acceptance

Plan: [../OPTION_A_EXCLUSIVE_RAM.md](../OPTION_A_EXCLUSIVE_RAM.md), Phase 6. Branch `xhelper/p5-p6-run`. Box: the B550
box of [00-baseline.md](00-baseline.md) (Ryzen 9 3900X, 32 GB DDR4; RTX 5060 Ti 16 GB on CPU lanes for the engine,
RX 9060 XT 16 GB on the chipset's x4 Gen3 link for the helper). Date: 2026-10-11. All arms use the same build of this
branch (CUDA engine `build-cuda`, helper `build-hip`), the server on 127.0.0.1, and the fixed prompt set, greedy.
One fresh server and one pass per arm.

**Acceptance: PASS.** Arm E starts without the "does not fit" warning, and its CPU pool took **0** expert blobs from
outside the RAM copy on every request. It answered 100 requests in a row. After a `kill -9` of the helper mid-request,
that request and the ones after it completed and the engine stayed up, once a fix described under Robustness was in.

## Arms

The configs are setup's: the Coder is `strata-coder-iq1_m.json`; IQ3_XXS is 0b's (the in-place pack, `--resident-experts`).
Both use `--spec 4 --mtp`, 64K context and int8 K/V. The helper arms add `--external-experts build-hip/strata-expert-helper
--external-experts-device 0 --external-experts-ram <mode>` and `STRATA_XH_TRACE=1`.

| Arm | Model | RAM mode | Decode tok/s (p1..p5) | Prompt 4K tok/s | Prompt 32K tok/s | Decode-pool blobs outside the RAM copy, per request | RAM copy | `free` used at start |
|---|---|---|---|---|---|---|---|---|
| A | Coder | resident, RTX only | 52.2 / 49.1 / 50.2 / 53.5 / 47.2 | 1545.1 | 1799.2 | 0 | 18.48 GiB | 24.0 GB |
| B | Coder | resident + helper, inclusive | 74.0 / 70.8 / 73.0 / 85.4 / 91.6 | 1481.6 | 1796.3 | 0 | 18.48 GiB | 23.5 GB |
| C | Coder | resident + helper, exclusive | 74.2 / 70.7 / 75.0 / 84.4 / 97.9 | 854.5 | 1575.1 | 0 | 0.00 GiB | 4.9 GB |
| D | IQ3_XXS | resident (soft fallback), RTX only | 25.4 / 27.6 / 30.8 / 20.0 / 14.1 | 183.8 | 311.0 | 8247 / 7746 / 3556 / 12346 / 3924 | 23.26 GiB | 28.7 GB |
| E | IQ3_XXS | resident + helper, exclusive | 68.4 / 63.3 / 74.4 / 67.7 / 64.1 | 226.9 | 332.2 | 0 / 0 / 0 / 0 / 0 | 15.96 GiB | 21.1 GB |

Short prompts, prompt tok/s (p3 70 tokens, p4 75, p5 407): A 111.7 / 131.8 / 404.8, B 129.2 / 140.2 / 410.7,
C 94.9 / 101.5 / 243.3, D 14.7 / 24.3 / 38.6, E 12.4 / 7.5 / 40.8.

"Decode-pool blobs" is new in this branch (`strata serve: decode pool: N expert blobs from outside the RAM copy this
request`). It counts the blobs the CPU pool took, in decode and verify windows, for experts outside the page-locked
RAM copy. The existing "blob reads from the file" counter also counts the prompt path's reads. Cumulative file reads
after the five prompts were: A 0, B 0, C 74,609, D 42,520, E 4,161. In C and E, all of them were prompt-path reads.

Startup lines:

```
A: resident RAM mode: 18.48 GiB of experts in RAM (page-locked), 4379 in the GPU cache
B: external helper (HIP device 0): 7909 experts, 15.08 GiB, loaded in 31.4 s; with CUDA0's 4379 that is 12288 of 12288 on the GPUs; RAM copy inclusive (keeps them too)
C: external helper (HIP device 0): 7909 experts, 15.08 GiB, loaded in 27.8 s; ... RAM copy exclusive (leaves them out)
   resident RAM mode: 0.00 GiB of experts in RAM (pageable), 4379 in the GPU cache
D: WARNING: the whole resident RAM mode does not fit (FileExpertSource: resident complement 31.32 GiB exceeds available RAM (27.51 GiB) minus the 4 GiB safety headroom); 23.26 GiB of the experts the GPU does not hold, the hottest by the expert profile, are kept in RAM and the rest are read from the model folder through the OS file cache
E: external helper (HIP device 0): 9438 experts, 15.36 GiB, loaded in 33.3 s; with CUDA0's 5341 that is 14779 of 24576 on the GPUs; RAM copy exclusive (leaves them out)
   resident RAM mode: 15.96 GiB of experts in RAM (page-locked), 5341 in the GPU cache
```

During arm D, 79 GB of unrelated model files were deleted from the same SSD (the owner's unused UD-Q2_K_XL). D's
numbers are within 0b/0c's ranges from Phase 0 (decode 14.1-30.8 here, 10.8-34.0 there).

## The helper, per request (arm E)

| Prompt | helper rows | CPU pool entries | helper's share | layers | layers that waited on the helper | round trip p50 / p99 us | timeouts |
|---|---|---|---|---|---|---|---|
| p1 | 110,328 | 14,014 | 88.7% | 9,900 | 8,769 (88.6%) | 208 / 486 | 0 |
| p2 | 97,993 | 10,267 | 90.5% | 11,943 | 11,141 (93.3%) | 173 / 419 | 0 |
| p3 | 46,982 | 4,400 | 91.4% | 4,452 | 4,197 (94.3%) | 196 / 401 | 0 |
| p4 | 115,266 | 19,663 | 85.4% | 9,850 | 7,963 (80.8%) | 214 / 606 | 0 |
| p5 | 30,109 | 7,768 | 79.5% | 2,331 | 1,500 (64.4%) | 244 / 742 | 0 |

"Share" is the helper's rows against the helper's plus the CPU's, of the entries the RTX did not compute. "Waited"
counts the layers where the CPU finished its rows before the helper's answer was there, so the helper set the pace. In
B and C (the Coder, every expert on a GPU), the CPU pool had 0 entries and nearly every layer waited on the helper.

## What the arms say

- With the helper, IQ3_XXS on 32 GB decodes at **63-74 tok/s, 2.2-4.5x arm D** (14-31). Every expert is on a GPU or in
  RAM, so decode reads nothing from the SSD.
- The Coder decodes **1.4-2.1x faster** with the helper (71-98 against 47-54 tok/s), inclusive or exclusive.
- Inclusive mode leaves prompts as they were (B vs A). Exclusive mode reads the helper's experts from the files in the
  prompt path: C's 4K prompt ran at 855 against 1545 tok/s. In E it costs less than it gains: the 4K and 32K prompts
  ran faster than D (227 / 332 against 184 / 311), but the 70-75-token prompts slower (7.5-12.4 against 14.7-24.3).
- The helper usually sets the pace (64-94% of layers wait on it). Its round trip is about 200 us at the median and up
  to 740 us at p99. The x4 Gen3 link and the 9060 XT's memory bandwidth are what is left to win (see 03-helper.md).

## Byte identity of the default path

Two builds: the commit before any helper code (`369ad8b`, code as `fb58e0d`) and this branch. Both ran the Coder with
setup's config plus `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`, each prompt on a fresh server, each build twice:

| Prompt | base 1 | base 2 | branch 1 | branch 2 | Result |
|---|---|---|---|---|---|
| p1-code-4k | 512 tok | 512 | 512 | 512 | **all four identical** (reasoning and answer) |
| p3-chat-short | 232 | 233 | 260 | 241 | the base build differs from itself |
| p4-chat-short | 391 | 380 | 512 | 361 | the base build differs from itself |
| p5-tool-use | 96 | 108 | 96 | 107 | the base build differs from itself |

p1 shows the default path unchanged: the same tokens in all four runs. The short prompts cannot show it either way.
The parent build gives different replies on two runs of the same config with these switches, so something else in the
engine is not repeatable on them. That is outside this plan, and was not looked into. The only code the branch runs
without `--external-experts` is the decode-pool counter (one `pinned()` lookup per CPU blob) and branches on a null
`external`.

## Robustness

**100 requests in a row on arm E** (the five prompts in turn, one fresh server): **100 of 100 answered**, decode 56-75
tok/s throughout, 0 helper timeouts.

**`kill -9` of the helper during a request.** A p4 request was sent, and the helper was killed 9 s later, in its decode.

- First attempt: this was not the test asked for. The test script found the helper with `pgrep -f strata-expert-helper`,
  which also matches the engine (its command line names the helper), so it killed both. The server restarted the
  engine and the next requests succeeded.
- Second attempt: the helper alone (by process name). The request failed with `external experts: the helper process
  exited (layer 0)`, and **the engine exited with code 1** right after it. A pool failure in a verify window ends the
  engine in serve mode (`generate.cpp`, the window loop's `ERR ...; return 1`, which predates this plan). The server
  restarted it (~90 s), and the request after that succeeded. That is not what the plan asks for: the engine has to
  stay up.
- Fix (this branch): when the helper fails in a layer, the CPU pool computes the rows it owned in that layer from the
  model files, in both the window and the one-token path. The window, the request and the engine go on, and from the
  next layer the helper owns nothing. This goes beyond the plan, whose request was allowed to error.
- Third attempt, with the fix: the request during the kill **completed** (512 tokens, 49.2 tok/s over the whole
  request). The two after it completed too (p3 12.1 tok/s, p4 11.9 tok/s: the helper's 15.4 GiB of experts now come
  from the SSD). The engine started once, with no restart:

```
strata: external experts: the helper stopped (the helper process exited); its experts are read from the model files from now on
strata serve: external: 110081 rows, 8993 layers, ..., timeouts 1; CPU pool 25942 entries (the helper has stopped)
strata serve: decode pool: 32406 expert blobs from outside the RAM copy this request
strata serve: external: 0 rows, 0 layers, ...; CPU pool 51343 entries (the helper has stopped)
```

`xhelper_external_experts_test` still passes: `finish()` reports the failure; the pool's fallback is above it.

The engine does not restart the helper. After the helper stops, it runs at file-tier speed until the server restarts
it.
