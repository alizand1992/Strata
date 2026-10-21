# The xhelper work rebased onto upstream

Box: the B550 box of [00-baseline.md](00-baseline.md) (Ryzen 9 3900X, 32 GB DDR4; RTX 5060 Ti 16 GB for the engine,
RX 9060 XT 16 GB on the chipset's x4 Gen3 link for the helper). Date: 2026-10-20.

## The change

The 22 commits of this work (Option A Phases 1-7.1, `--kv-grow` beside the helper, the Monitor's helper card) were
rebased from upstream `477ac13c` onto upstream `main` at `9de04cd4` (Niko1221/Strata, 943 commits later; engine 0.1.45
-> 0.1.46.3). `main` is now `54df2d15`; the old `main` is `25824736`. Every conflict kept both sides: `.gitignore` (twice), `CMakeLists.txt` (the protocol test beside upstream's memory guard test), `DETAILS.md` (the `--kv-grow`
entry: this work's resident-mode text, upstream's HIP paragraph), `server.py` (the helper's stats path is set before
upstream's offline policy, which stays last), `telemetry.py` (upstream names each card itself; the helper fields are
added to it), and in the engine: the router lookahead skips the helper's experts inside upstream's new
confidence loop, `set_skip` comes before upstream's `lookahead.start`, the exclusive RAM copy leaves the helper's
experts out beside upstream's split lend regions, and the decode pool's file-tier count sits beside upstream's
resident RAM lines.

Checked: `serve/test_server.py` (346 tests) and `serve/test_monitor.py` (10) pass; the CUDA and HIP builds compile with
no new warnings (the 23 CUDA warnings are all in upstream lines).

## How it was measured

The box's own config (`strata-iq3_s-2gpu.json`): IQ3_S, `--max-context 262144 --kv int8 --resident-experts --spec 4
--mtp --kv-grow`, helper exclusive with `--external-experts-prompt 2048`. "main" is the old `main` (`build/`,
`build-hip/`); "rebased" is `54df2d15` (`build-rb-cuda`, `build-rb-hip`). The same prompts and replays as
[09-kv-grow.md](09-kv-grow.md): the fixed prompt set, then agent replays (`agent_replay.py`: repo source as the first
message, then short follow-up turns that each reuse the context, 256 tokens a reply) of ~20K and ~87K tokens on the
same server, and of ~195K tokens on a fresh server per arm. One pass each, same evening, greedy.

The HIP build needs `-DSTRATA_PREFILL_MMQ=ON` (the default is off). A first rebased run without it had a helper that
declined prompt work ("MMQ covers no layer's formats on this card"): its prompts are left out here and the arm was run
again. The ~195K servers need `STRATA_ENGINE_READY_S=2400` right after another arm: started back to back, both arms'
page-locking of the 22 GiB RAM copy outran the 900 s default.

## Start

| | main | rebased |
|---|---|---|
| RTX expert cache | 4,455 slots, 8.50 GiB | 4,451 slots, 8.49 GiB |
| Helper | 8,019 experts, 15.17 GiB | 8,018 experts, 15.17 GiB |
| RAM copy kept | 22.24 GiB | 22.15 GiB |

## Results

Fixed prompt set, prompt tok/s / decode tok/s:

| Prompt | main | rebased |
|---|---|---|
| p1 (5,182 tokens) | 271.5 / 51.6 | 340.2 / 57.7 |
| p2 (33,537) | 455.8 / 53.1 | 639.0 / 55.8 |
| p3 (70) | 37.2 / 55.5 | 37.5 / 60.4 |
| p4 (75) | 85.5 / 41.8 | 83.9 / 56.3 |
| p5 (407, tool call) | 144.8 / 49.6 | 137.0 / 48.3 |

Agent replays:

| Context | | main | rebased |
|---|---|---|---|
| ~20K (21,089) | first prompt tok/s / decode, wall | 382.5 / 45.8, 61.3 s | 640.6 / 50.6, 38.1 s |
| | 6 turns: prompt tok/s | 143-277 | 70-267 |
| | 6 turns: decode tok/s | 58-69 | 58-66 |
| | 6 turns: wall, total | 38.3 s | 43.8 s |
| ~87K (86,668) | first prompt tok/s / decode, wall | 487.0 / 36.4, 185.8 s | 724.0 / 53.0, 125.2 s |
| | 4 turns: prompt tok/s | 120-237 | 174-206 |
| | 4 turns: decode tok/s | 55-57 | 57-60 |
| | 4 turns: wall, total | 29.2 s | 27.3 s |
| ~195K (194,906) | first prompt tok/s / decode, wall | 484.5 / 38.2, 410.2 s | 702.1 / 45.8, 284.5 s |
| | 4 turns: prompt tok/s | 99-217 | 153-189 |
| | 4 turns: decode tok/s | 48-51 | 48-52 |
| | 4 turns: wall, total | 36.4 s | 32.5 s |

The ~195K first prompt: the RTX waited 226.6 s on the helper in main and 139.3 s rebased; the OS read 230.7 and
239.0 GB from the SSD. main had one helper stall there (no answer within 2000 ms at layer 0; the CPU computed those
rows until it answered); rebased had none. `--kv-grow` took the same steps in both (to 40,960 cells and back, to
106,496, to 196,608 and 204,800; slot counts within 5 of each other). The replies read the same and are on task.

## What it says

- Long prompts are 1.25-1.7x faster rebased: the first reply of an agent session takes 0.6-0.7x the time at 20K, 87K
  and 195K, and the 32K prompt of the fixed set runs at 639 against 456 tok/s.
- Short prompts are even (p3-p5 within 6%). Decode is even to faster (p4 +35%, the first reply at 87K +46%); p5 is
  2.6% slower.
- Follow-up turns are even at 87K and 195K. At 20K the rebased arm was 5.5 s slower over six turns, almost all of it
  the first turn (11.8 s against 7.9 s); one pass does not say whether that is noise.
- Which of upstream's commits the gains come from is not measured. Not tested: more than one pass per arm, a real agent
  session, conversations switching back and forth, the CPU fallback after a helper stall on the rebased build.

The box's server runs `54df2d15` since 21:02 (`build-main-cuda`, `build-main-hip`, the same options; config backup
`strata-iq3_s-2gpu.json.bak-rebase`).
