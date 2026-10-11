# Phase 0 baselines (plan section "Phase 0")

Plan: [../OPTION_A_EXCLUSIVE_RAM.md](../OPTION_A_EXCLUSIVE_RAM.md). Pre-flight: [00-preflight.md](00-preflight.md).
Box: the B550 Tomahawk MAX box (Ryzen 9 3900X, AVX-2 only, 32 GB DDR4, Ubuntu, kernel 7.0.0-38). RTX 5060 Ti 16 GB
(driver 580.178.04, CUDA 13.0) on CPU lanes; RX 9060 XT 16 GB on the chipset's x4 Gen3 link. Date: 2026-10-11.
Repository at `e0b011b` (code as `fb58e0d`; only docs changed since), engine 0.1.41.

Every run uses the server (`serve/server.py --engine strata`, 127.0.0.1) and the fixed prompt set in
[prompts/](prompts/), sent one after another to a fresh server, greedy (`temperature 0`). Speeds are the server's
`timings` (`prompt_per_second`, `predicted_per_second`). One pass per run: these are single measurements, and the
short prompts' numbers move with what the file cache holds. Nothing else ran on the GPUs; the owner's llama-server was
stopped first.

Engine args common to all runs (setup's config for this box): `--expert-profile <profile> --expert-cache auto
--prefill auto --spec 4 --spec-min-p 0.5 --mtp ~/Strata-data/mtp/rt --max-context 65536 --kv int8`.

## The IQ3_XXS pack

Setup's pack for IQ3_XXS only writes `experts.bin` (~43 GB) for the low-RAM mode, and that does not fit on this disk
(32 GB free). 0b and 0c use a pack made the way setup makes it otherwise, `tools/iq_pack.py --gguf <shard 1> --out
~/Strata-data/packs/iq3_xxs` (3 s, 1.5 GB: `dense.bin`, index, tokenizer), which reads the experts from the GGUF
shards in `~/models/qwen-3.8-next-gsq/` in place. Expert profile: `data/expert-profile.bin` (0a: `-coder.bin`).

## Results

Prompt tokens: p1 5,182, p2 33,537, p3 70, p4 75, p5 407. Every reply ends on `length` at 512 tokens (p5: 256) or
earlier on `stop` / `tool_calls`; the decoded count is in brackets where it is not the limit.

| Prompt | 0a prompt / decode tok/s | 0b prompt / decode | 0c prompt / decode | 0d prompt / decode |
|---|---|---|---|---|
| p1-code-4k | 1547.9 / 52.8 | 185.4 / 24.2 | 210.4 / 27.3 | 414.5 / 27.4 |
| p2-code-32k | 1797.6 / 53.0 | 318.7 / 30.2 | 375.7 / 27.8 | 597.1 / 27.5 |
| p3-chat-short | 121.8 / 49.9 (272) | 16.8 / 30.8 (228) | 24.4 / 34.0 (228) | 18.9 / 26.0 (256) |
| p4-chat-short | 135.0 / 55.5 (342) | 36.9 / 19.4 (512) | 43.2 / 17.4 (452) | 19.8 / 30.6 (362) |
| p5-tool-use | 404.1 / 44.1 (107, tool call) | 25.2 / 20.7 (118, tool call) | 59.4 / 10.8 (117, tool call) | 58.0 / 29.7 (68, tool call) |

### 0a - Coder (IQ1_M), `--resident-experts`, RTX only

Setup's own config (`strata-coder-iq1_m.json`). Startup (engine log):

```
strata generate: expert cache auto: 9.24 GiB free, 700 MiB reserved (+218 MiB for the draft head) -> 3364 slots
strata generate: expert cache 4382 slots, 8.34 GiB of VRAM; policy is
FileExpertSource: allocating 18.47 GiB page-locked cache complement
FileExpertSource: 9.18 GiB really free for a 18.47 GiB page-locked complement; the model files' cached pages were released: 17.56 GiB free
FileExpertSource: copied cache complement through layer 48/48 (18.47 GiB, 13 s)
FileExpertSource: mapped pinned cache complement ready: resident 18.47 GiB, pinned 18.47 GiB in 25 s; 18.47 GiB page-locked after taking the pages in steps;
strata generate: resident RAM mode: 18.47 GiB of experts in RAM (page-locked), 4382 in the GPU cache; adaptive swaps exchange them with the GPU cache (no file reads)
strata serve: the prompt path borrows 1803 CUDA0 cache slots (3.40 GiB)
strata serve: 354 MiB of VRAM free with everything loaded
```

Per request (cumulative since start):

```
resident RAM: 18.47 GiB of experts in RAM, 2726 exchanged with the VRAM tier, 0 blob reads from the file
resident RAM: 18.47 GiB of experts in RAM, 5441 exchanged with the VRAM tier, 0 blob reads from the file
resident RAM: 18.47 GiB of experts in RAM, 7107 exchanged with the VRAM tier, 0 blob reads from the file
resident RAM: 18.47 GiB of experts in RAM, 8597 exchanged with the VRAM tier, 0 blob reads from the file
resident RAM: 18.47 GiB of experts in RAM, 9116 exchanged with the VRAM tier, 0 blob reads from the file
```

### 0b - IQ3_XXS, `--resident-experts`

The warning and the soft fallback, as the plan expected:

```
strata generate: expert cache 5341 slots, 8.65 GiB of VRAM; policy is
FileExpertSource: RAM budget 23.02 GiB: 14102 of the 31.31 GiB of experts the GPU cache does not hold, by profile rank; the rest are read from the files
FileExpertSource: 6.03 GiB really free for a 23.02 GiB page-locked complement; the model files' cached pages were released: 18.66 GiB free
FileExpertSource: mapped pinned cache complement ready: resident 23.02 GiB, pinned 23.02 GiB in 52 s; 23.02 GiB page-locked after taking the pages in steps;
strata generate: WARNING: the whole resident RAM mode does not fit (FileExpertSource: resident complement 31.31 GiB exceeds available RAM (27.27 GiB) minus the 4 GiB safety headroom); 23.02 GiB of the experts the GPU does not hold, the hottest by the expert profile, are kept in RAM and the rest are read from the model folder through the OS file cache
strata serve: the prompt path borrows 2103 CUDA0 cache slots (3.37 GiB)
```

Per request: 10298, 20370, 25366, 42917, 47170 blob reads from the file (cumulative; 4382 -> 16052 exchanged with
the VRAM tier). `free -g` after the five prompts: 27 GB used, 3 GB available.

### 0c - IQ3_XXS, `--mmap-experts --resident-budget-gib 20`

```
strata generate: experts via mmap (--mmap-experts; the GGUF shards in place, no experts.bin)
strata generate: expert cache 5341 slots, 8.65 GiB of VRAM; policy is
FileExpertSource: RAM budget 20.00 GiB: 12254 of the 31.31 GiB of experts the GPU cache does not hold, by profile rank; the rest are read from the files
FileExpertSource: mapped pinned cache complement ready: resident 20.00 GiB, pinned 20.00 GiB in 41 s; page-locked and mapped
strata generate: the file tier reads through the file cache (re-checked with the RAM copy built (20.00 GiB): 8.1 GiB available, 0.0 GiB of it still to be taken by the RAM copy, 20.0 GiB of experts read from the files: the file cache can keep the ones that come back)
strata serve: 356 MiB of VRAM free with everything loaded
```

Per request: 13529, 27573, 33237, 55603, 62151 blob reads from the file (cumulative; 4101 -> 14445 exchanged).

### 0d - Coder (IQ1_M), HIP engine alone on the RX 9060 XT

The plan says `./setup.sh --backend hip`; that rewrites setup's config and `engine/`, which 0a uses, so the HIP engine
was built the way `build_engine_hip()` builds it, into `build-hip/` only (gfx1200, setup's ROCm 7.10 in `.venv`), and
run with 0a's args plus what setup adds for HIP: `"backend": "hip"`, the ROCm library dirs, `STRATA_RESIDENT_PIN=0`.
Two things on this box that setup's own HIP build would also hit:

- `ninja` is only in `.venv/bin` (not on PATH); CMake needs `-DCMAKE_MAKE_PROGRAM` pointed at it.
- ROCm's clang picks GCC 14 (`/usr/lib/gcc/x86_64-linux-gnu/14` exists) but only GCC 13's C++ headers are installed:
  `'cstdlib' file not found` while CMake tests the HIP compiler. Built with `--gcc-install-dir=.../13` in
  `CMAKE_HIP_FLAGS`; `sudo apt install libstdc++-14-dev` is the fix that needs no flag.

No hipBLASLt tuning table: the installed hipBLASLt is 100200 and the shipped table is `gfx1200-hipblaslt-100202.txt`,
so setup would also run without one (plain hipBLAS for the prompt's dense products).

```
strata generate: PCIe probe: 3.1 GB/s host->device (best of 3.1 3.1 3.1 3.1) -> pcie_frac 0.09 (default 0.55)
strata generate: GPU 0: AMD Radeon Graphics (gfx1200)
strata generate: expert cache auto: 9.80 GiB free, 700 MiB reserved (+218 MiB for the draft head) -> 3589 slots
strata generate: expert cache 4674 slots, 8.90 GiB of VRAM; policy is
FileExpertSource: allocating 17.92 GiB pageable resident cache complement
FileExpertSource: pageable resident cache complement ready: resident 17.92 GiB, pinned 0.00 GiB in 19 s
strata generate: resident RAM mode: 17.92 GiB of experts in RAM (pageable), 4674 in the GPU cache; adaptive swaps exchange them with the GPU cache (no file reads)
strata serve: 592 MiB of VRAM free with everything loaded
```

Per request: 0 blob reads from the file (2809 -> 9385 exchanged with the VRAM tier). The RTX stayed idle (129 MiB).

### What the four say

- With every expert outside VRAM in RAM (0a), decode is 44-56 tok/s and no expert is read from the file.
- The 9060 XT alone (0d) decodes the Coder at 26-31 tok/s, about half of the RTX, and reads long prompts at about a
  third of its speed (415-597 vs 1548-1798 tok/s). Its host link measured 3.1 GB/s, the chipset's x4 Gen3.
- IQ3_XXS on 32 GB has 31.3 GiB of experts outside VRAM and room for 20-23 GiB of them. The other ~8-11 GiB are read
  through the file cache (47-62 thousand blob reads over five prompts), and decode drops to 11-34 tok/s, prompts to
  17-376 tok/s. This is the gap the exclusive RAM tier is for.
- 0b's bigger RAM copy (23 GiB) did not beat 0c's 20 GiB with more file cache left: 0c read prompts 13-136% faster;
  decode was mixed (0c faster on p1/p3, slower on p2/p4/p5). One pass each, so this is not a ranking of the two modes.
