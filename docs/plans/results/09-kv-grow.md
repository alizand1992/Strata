# --kv-grow beside --resident-experts and --external-experts

Branch `xhelper/kv-grow`. Box: the B550 box of [00-baseline.md](00-baseline.md) (Ryzen 9 3900X, 32 GB DDR4; RTX 5060 Ti
16 GB for the engine, RX 9060 XT 16 GB on the chipset's x4 Gen3 link for the helper). Date: 2026-10-20.

## The change

`--kv-grow` (the elastic K/V: VRAM for the cells the requests reach, the rest lent to the expert cache) was refused in
the resident low-RAM mode and beside `--external-experts`. In that mode the RAM copy holds only what the GPUs do not,
so an expert the cache gave up to the K/V had no copy but the files. Now:

- an expert the cache gives up, or moves into the prompt path's loan region, takes the RAM place of the coldest expert
  of its layer that the copy holds and no GPU does, if that one is colder (`FileExpertSource::replace_resident`, one
  file read each; the RAM copy stays the same size);
- the refill after the K/V shrinks skips the helper's experts, and an expert refilled from the RAM copy gives its RAM
  place to the hottest expert of its layer that no GPU and not the copy holds;
- the two refusals are lifted. Without `--kv-grow` nothing changes.

## How it was measured

The box's own config (`strata-iq3_s-2gpu.json`): IQ3_S, `--max-context 262144 --kv int8 --resident-experts --spec 4
--mtp`, helper exclusive with `--external-experts-prompt 2048`. Both arms use the same build of this branch
(`build-p7-cuda`, `build-p7-hip`); "on" adds `--kv-grow`. Fresh server per arm, one pass each, same day, one after the
other, greedy. The fixed prompt set, then two agent replays (`agent_replay.py`: repo source as the first message, then
short follow-up turns that each reuse the context, 256 tokens a reply) of ~20K and ~87K tokens; the ~195K replay ran
on another fresh server per arm.

## Start

| | off | on |
|---|---|---|
| RTX expert cache | 2,819 slots, 5.40 GiB | 4,455 slots, 8.50 GiB (K/V: 16,384 of 262,144 cells mapped) |
| RAM copy wanted / kept | 26.26 / 22.45 GiB | 23.16 / 22.24 GiB |

## Results

Fixed prompt set, prompt tok/s / decode tok/s:

| Prompt | off | on |
|---|---|---|
| p1 (5,182 tokens) | 204.5 / 41.6 | 312.3 / 50.5 |
| p2 (33,537) | 379.2 / 45.8 | 466.2 / 53.0 |
| p3 (70) | 15.1 / 42.5 | 8.8 / 42.6 |
| p4 (75) | 44.3 / 40.4 | 63.1 / 52.7 |
| p5 (407, tool call) | 43.9 / 35.3 | 146.9 / 47.1 |

Agent replays, the follow-up turns (each ~440-550 new tokens and 256 generated):

| Context | | off | on |
|---|---|---|---|
| ~20K (21,089) | first prompt tok/s / decode | 355.3 / 38.3 | 432.7 / 47.6 |
| | 6 turns: prompt tok/s | 38-154 | 164-261 |
| | 6 turns: decode tok/s | 46-58 | 47-69 |
| | 6 turns: wall, total | 77.4 s | 38.6 s |
| ~87K (86,668) | first prompt tok/s / decode | 450.0 / 40.9 | 453.4 / 48.3 |
| | 4 turns: prompt tok/s | 43-168 | 148-226 |
| | 4 turns: decode tok/s | 43-56 | 37-60 |
| | 4 turns: wall, total | 45.4 s | 31.5 s |
| ~195K (194,906) | first prompt tok/s / decode | 477.5 / 34.3 | 485.2 / 41.7 |
| | 4 turns: prompt tok/s | 54-135 | 57-200 |
| | 4 turns: decode tok/s | 50-55 | 33-48 |
| | 4 turns: wall, total | 44.5 s | 45.5 s |
| | 4 turns: OS read from the SSD | 1.9-3.1 GB a turn | 1.2-2.3 GB a turn |

At ~195K the K/V had taken 1,253 slots (3,202 of 4,455 still hold experts, against the off arm's fixed 2,819), and
1,687 experts it gave up or moved took colder ones' RAM places. The second growth step (to 204,800 cells, 53 slots)
fell on turn 2, the slowest turn of that arm (17.4 s, decode 33.2).

The replies read the same in both arms and are on task (the reasoning names the last file and what it does); with 256
tokens most of them end inside the reasoning, in both arms.

## What it says

- Up to ~90K, the context this box's sessions spend most of their time in, agent turns take 0.5-0.7x the time, and
  decode is ~15-25% faster at the first reply.
- At ~195K the two are even on turn time (44.5 against 45.5 s for four turns): the cache still holds 383 more experts
  than the fixed one, but each growth step costs a pause, and the experts it gave up are computed by the CPU instead
  of the helper.
- Shrinking costs a short request: p3 came right after the K/V shrank from the 32K prompt (161 slots refilled, 156 RAM
  places read from the files) and was the one prompt slower with it on (8.0 s against 4.6 s; that request read
  1.56 GB from the SSD). A continued conversation does not shrink; switching to an unrelated short one does.
  `STRATA_KV_GROW_HOLD=1` keeps the K/V as long as the longest parked conversation (not measured here).

The off arm had two helper stalls (2.0 s, then answered again after 2.1 s; the CPU computed those rows), one before
p3 and one at the start of its 195K run. The on arm had none. Not tested: more than one pass per arm, a real agent
session, `STRATA_KV_GROW_HOLD=1`, conversations switching back and forth.
