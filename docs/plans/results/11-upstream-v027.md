# Results 11: the xhelper work merged with upstream v0.2.7

Branch `merge/v0.2.7`: upstream `5e8a4cca` (v0.2.7, engine 0.1.52) merged into this fork's `main` (`77f11cd8`, on
v0.2.2). A merge this time, not a rebase. Measured on the box on 2026-10-27, one pass each, greedy.

## The merge

Six files had conflicts. In five, both sides had added something and both are kept (the helper's CMake targets beside
upstream's DeepSeek-V4.1 ones, `set_external` beside upstream's layer-major prefill, the helper's stats path beside
upstream's DeepSeek environment, the decode pool line beside upstream's new `STRATA_RAM_ADAPT` line). Upstream
rewrote `DETAILS.md`; this work's `--kv-grow` note moved into its settings table.

One change was added: layer-major prefill (new upstream, on by default only for a CUDA layer split) now declines when
the helper takes prompt experts. Only the chunk-major path computes them, so without this a prompt read layer-major
would leave the helper's experts out. The box does not split layers, so its default path does not change.

Checked: `serve/test_server.py` and `serve/test_monitor.py` (369 tests) pass. The CUDA engine and HIP helper build
(`-DSTRATA_PREFILL_MMQ=ON`). Not run: `tests/terminal` (no pytest in the box's venv).

## How it was measured

The box's config (`strata-iq3_s-2gpu.json`: IQ3_S, `--max-context 262144 --kv int8 --resident-experts --spec 4 --mtp
--kv-grow`, helper exclusive with `--external-experts-prompt 2048`). "old" is the server the box ran (`77f11cd8`,
`build-v022-cuda`, `build-v022-hip`); "merged" is `merge/v0.2.7` with only the two binary paths changed. The fixed
prompt set (`prompts/run_set.py`), then agent replays (`prompts/agent_replay.py`) of ~21K and ~85K tokens on the same
server and ~192K on a fresh one. Both scripts are new: the earlier runner and replay were never committed, so the
replay numbers here compare only with each other, not with results 09 and 10. A replay turn adds ~500 tokens (a
question, ~1,300 characters of source as a tool result, the last reply) and has a 128-token thinking budget.

The merged arm's ~21K first prompt reused part of an earlier run's cache, so it is left out.

## Results

Fixed prompt set, prompt tok/s / decode tok/s:

| Prompt | old | merged | merged, `--conversation-cache-mib 0` |
|---|---|---|---|
| p1 (5,182 tokens) | 654.5 / 58.9 | 659.4 / 60.2 | 464.0 / 59.3 |
| p2 (33,537) | 785.4 / 60.1 | 790.0 / 59.5 | 768.5 / 60.9 |
| p3 (70) | 79.8 / 63.4 | 50.7 / 64.5 | 84.6 / 59.9 |
| p4 (75) | 109.4 / 59.3 | 95.3 / 59.6 | 116.7 / 59.2 |
| p5 (407, tool call) | 270.8 / 53.7 | 229.0 / 46.9 | 298.2 / 50.5 |

p1 is the first request after the server starts in all three arms; one pass does not say whether the 464 is noise.

Agent replays:

| Context | | old | merged | merged, cache 0 |
|---|---|---|---|---|
| ~21K (21,382) | first prompt tok/s / decode | 782.5 / 50.8 | (cache reuse) | - |
| | 6 turns: prompt tok/s | 212-357 | 196-302 | - |
| | 6 turns: wall, total | 33.2 s | 35.1 s | - |
| ~85K (84,614) | first prompt tok/s / decode | 793.5 / 57.4 | 804.4 / 56.9 | 807.5 / 50.3 |
| | 4 turns: prompt tok/s | 233-330 | 200-277 | 302-330 |
| | 4 turns: wall, total | 24.8 s | 27.0 s | 23.1 s |
| ~192K (192,085) | first prompt tok/s / decode, wall | 766.0 / 48.9, 256.3 s | 753.9 / 48.0, 261.2 s | - |
| | 4 turns: prompt tok/s | 255-342 | 58-257 | - |
| | 4 turns: wall, total | 27.0 s | 37.5 s | - |

Helper stall: the helper stopped with SIGSTOP for 3 s during a 1,156-token reply on the merged build. The engine
logged "no answer within 2000 ms (layer 46)", the CPU computed the helper's rows, the helper answered again after
3.0 s, the reply finished (`stop`) and the engine stayed up.

## What it says

- Long first prompts and decode are even between old and merged (within 2%).
- Short prompts and follow-up turns are slower merged, and the cause is upstream #3493: `--conversation-cache-mib`
  went from off to on by default (sized from the free RAM, up to 8 GiB). Before each turn the engine now copies the
  current conversation's K/V to RAM: 0.2-0.3 s at ~21K, 0.25-1.1 s at ~85K, 0.5-4.2 s at ~192K (the engine log's
  "conversation cache: parked" lines). With `--conversation-cache-mib 0` the merged build's follow-ups at ~85K are
  302-330 tok/s, the old build's 233-330.
- The cache pays when a client switches between conversations (it restores one instead of reading it again). With
  one conversation at a time it only costs time and RAM on this box. Not measured: conversations switching back and
  forth, more than one pass per arm.
