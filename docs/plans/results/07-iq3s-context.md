# IQ3_S with the helper at 64K, 128K and 256K context

Box: the B550 box of [00-baseline.md](00-baseline.md) (32 GB RAM, RTX 5060 Ti 16 GB + RX 9060 XT 16 GB on x4 Gen3).
Date: 2026-10-11. Build: `xhelper/p5-p6-run` (09626e1). Model: `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` from ISTA-DASLab's
GGUF repo. Shard 1 was downloaded (sha256 `4c1eb2ce…2aca3`, as published). Shard 2 is byte-identical to IQ3_XXS's
(same published sha256) and is a hard link to it. The pack was made with `tools/iq_pack.py` (experts read from the
GGUF in place): 46.84 GiB of experts.

Config: arm E's of [06-results.md](06-results.md) (`--resident-experts --spec 4 --mtp --kv int8`, helper exclusive,
`STRATA_XH_TRACE=1`), with the IQ3_S pack and `--max-context` 65536 / 131072 / 262144. One fresh server and one pass
of the prompt set each.

The RTX-only arm (S0) was not measured. Its RAM copy took the engine's whole budget (23.08 of 38.56 GiB, "does not
fit"), and Claude Code's low-memory guard stopped the run during the first request. The owner then asked to skip it.

| | 64K | 128K | 256K |
|---|---|---|---|
| RTX expert cache | 4,341 slots, 8.28 GiB | 3,836, 7.33 GiB | 2,825, 5.41 GiB |
| Helper (RX 9060 XT) | 8,112 experts, 15.36 GiB | 8,119 | 8,120 |
| RAM copy | 23.20 GiB, **fits** | 23.66 of 24.15 GiB ("does not fit") | 23.57 of 26.07 GiB ("does not fit") |
| `free` available at start | 3.1 GB | 2.7 GB | 2.8 GB |
| Decode tok/s p1 / p2 / p3 / p4 / p5 | 56.6 / 55.7 / 59.0 / 60.3 / 46.7 | 52.6 / 49.6 / 50.2 / 43.4 / 35.1 | 33.4 / 41.8 / 10.4 / 3.0 / 2.0 |
| Prompt tok/s 4K / 32K | 147.4 / 243.5 | 140.1 / 244.3 | 123.0 / 237.1 |
| Prompt tok/s p3 / p4 / p5 (70 / 75 / 407 tokens) | 4.9 / 4.9 / 12.7 | 5.1 / 4.8 / 14.8 | 4.2 / 3.8 / 12.6 |
| Decode-pool blobs outside the RAM copy, per request | 0 / 0 / 0 / 0 / 0 | 413 / 320 / 137 / 672 / 139 | 1,288 / 1,230 / 5,898 / 78,188 / 16,061 |
| Helper's share (its rows / its + CPU's) | 71-87% | 68-87% | 76-87% (until it stopped) |
| Helper round trip p50 / p99 | 184-315 us / 591-891 us | 205-447 us / 3.5-17.3 ms | 208-243 us / 4.4-14.8 ms |

**256K, p3:** `strata: external experts: the helper stopped (no answer within 2000 ms)`. The helper process was alive
(it stopped normally at the end: "stopping after 27053 requests"). There was no OOM kill, but the box had swapped
(1.1 GB in swap after the runs). One answer took over 2 s. The engine then dropped the helper for the session and
computed its experts on the CPU from the files, which is p3-p5's 2-10 tok/s.

## What it says

- **64K is the size that fits IQ3_S on this box:** decode 47-60 tok/s, with nothing read from the SSD in decode.
- **Each step up in context takes VRAM from the RTX's experts.** At int8 the K/V costs 505 slots (0.95 GiB) more at
  128K and 1,516 (2.87 GiB) more at 256K. On 32 GB those experts have nowhere to go but the SSD. 128K loses 10-25% of
  decode, and the box runs close to swapping. 256K swaps.
- **Short prompts are slow at every size (4-15 tok/s).** In exclusive mode the prompt path reads the helper's
  15.4 GiB of experts from the files, and with RAM full the file cache cannot keep them.
- **One slow answer is enough to lose the helper.** The 2 s timeout treats a swap stall like a dead process.

## A late answer no longer stops the helper

Change (this branch, after the runs above): a timeout with the helper's process still there is now transient.
`finish()` returns false, the CPU pool computes that layer's helper rows from the files (the fallback of
[06-results.md](06-results.md)), and the helper stays in use. `begin()` owns nothing until the late answer arrives,
because it may not write a new request over one the helper could still be reading. After that the helper gets requests
again. The helper only counts as gone if its process exits, it answers with an error, or it has not answered at all
for `STRATA_XH_GIVE_UP_MS` (default 60000). The engine prints one line when a stall starts and one when it ends:

```
strata: external experts: no answer within 300 ms (layer 0); the CPU computes the helper's rows until it answers
strata: external experts: the helper answered again after 0.3 s
```

`xhelper_external_experts_test` has a new step: it sends SIGSTOP to the helper with a request in flight (timeout set
to 300 ms in the test), and checks that `finish()` fails, the helper stays alive, and `begin()` owns nothing. After
SIGCONT it checks that the helper answers again and its rows still match (rel L1 0.0146, bar 3e-2). The kill -9 step
is unchanged. PASS. Both backends build (CUDA `strata` and the test; HIP `strata` and `strata-expert-helper`).

## Reruns at 128K and 256K with the change

Same configs, int8 K/V (the owner does not want K/V below 8 bits), one fresh server and one pass each:

| | 128K before | 128K rerun | 256K before | 256K rerun |
|---|---|---|---|---|
| RAM copy | 23.66 GiB | 23.58 GiB | 23.57 GiB | 22.44 GiB |
| Decode tok/s p1 / p2 / p3 / p4 / p5 | 52.6 / 49.6 / 50.2 / 43.4 / 35.1 | 48.5 / 55.3 / 46.4 / 50.5 / 30.7 | 33.4 / 41.8 / 10.4 / 3.0 / 2.0 | 32.8 / 41.0 / 34.3 / 29.8 / 27.0 |
| Prompt tok/s 4K / 32K | 140.1 / 244.3 | 128.8 / 246.3 | 123.0 / 237.1 | 137.7 / 237.5 |
| Prompt tok/s p3 / p4 / p5 | 5.1 / 4.8 / 14.8 | 4.8 / 4.5 / 14.8 | 4.2 / 3.8 / 12.6 | 5.6 / 4.4 / 15.5 |
| Decode-pool blobs outside the RAM copy | 137-672 | 219-754 | 1,230-78,188 | 1,219-5,551 |
| Helper round trip p99 | 3.5-17.3 ms | 2.2-17.3 ms | 4.4-14.8 ms | 4.9-13.9 ms |
| Timeouts | 0 | 0 | 1 (helper dropped) | 0 |

At 256K the helper stayed in use for all five prompts, and decode held at 27-41 tok/s instead of dropping to 2-10.
But there was **no timeout in this rerun**, so the new code path did not run here. Only the test exercises it. The
rerun had no 2 s stall to recover from. The first 256K attempt of the rerun did not start: the engine was still
page-locking its RAM copy after 15 minutes (2.38 GiB really free, 0.9 GB in swap from the runs before it), and the
script gave up. The second attempt page-locked 22.44 GiB in 61 s.

The prompt numbers come from the prompt set's 4K and 32K prompts on each context size. No prompt longer than 32K
was run, so prefill deep into a 128K or 256K context is not measured.
