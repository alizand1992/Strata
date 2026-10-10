# Plan B: hot experts in RAM, cold experts on the SSD ("disk pin"), with batched promotion

Goal: run a model whose experts do not fit RAM (IQ3_XXS on 32 GB; IQ3_S / UD-IQ4_XS on 64 GB) by keeping only the
hottest experts in RAM and reading the rest from an NVMe SSD, and make the RAM tier **learn** what is hot during a
conversation instead of fixing it at start.

Target box: Ryzen 9 3900X, 32 GB DDR4, RTX 5060 Ti 16 GB, NVMe on CPU lanes, Linux.

## What already exists (read before writing code)

A three-level layout is already in the engine as the **RAM budget** (`--resident-budget-gib N`, engine 0.1.31,
docs/DETAILS.md "A RAM budget"):

- VRAM: the adaptive GPU expert cache.
- RAM: the N GiB of experts the GPU does not hold that the **expert profile ranks hottest**, copied in at start
  (`FileExpertSource::pin_cache_complement`, the `budget_bytes > 0` branch, `src/core/expert_source.cpp:2240-2293`).
- SSD: everything else, read from the mapped files (`--mmap-experts`), with a **lookahead**: while the CPU works on a
  layer, a thread applies the next layer's router to this layer's input and asks the OS for the predicted experts'
  pages (`STRATA_LOOKAHEAD=0` turns it off).
- Batched reads exist: `FileExpertSource::copy_blobs` (`:2086`) merges neighbouring blobs into one request, and the
  prompt path stages whole chunks of experts.
- Measured: Unsloth UD-Q4_K_XL (72 GiB of experts) on a 64 GB PC with an RTX 5070 at N = 40: 7-8.5 tok/s, vs ~3 tok/s
  before the budget and lookahead.

**The gap: the RAM tier is static during a run.** The GPU tier adapts, but a swap only updates RAM when the incoming
expert was already in the RAM copy. `generate.cpp:7666-7669`:

> an exchange: `in` from the RAM copy, `out` held by its slot alone takes its place there; otherwise ... `in` is not
> (a RAM budget)

and `stage_exchange` refuses unless `has_resident(layer, in)` (`expert_source.cpp:2812`). So under a budget:

1. An expert promoted to the GPU **from the SSD** evicts a GPU expert that is then **dropped to SSD-only**, even if it
   was hot a moment ago.
2. An SSD-only expert the CPU keeps reading is never promoted into RAM; only a restart with a saved profile
   (`--expert-profile-save`, #477) re-ranks the RAM tier.

## Step 0 - measure before building (no changes, ~1-2 hours)

1. IQ3_XXS with `--mmap-experts --resident-budget-gib N` on the RTX alone. Start with N = 20 on 32 GB (the engine
   clamps to available RAM minus 4 GiB headroom and prints what it kept).
2. Replay one real agent session (a fixed transcript from your notifications repo) with `--serve` and record per
   request: decode and prompt tok/s, `resident RAM: ... blob reads from the file`, and disk read MB/s (`vmstat 1`).
3. Repeat after `--expert-profile-save` + restart with that profile, to see how much a learned-across-restarts RAM
   tier already gives.
4. Count, from the engine's stats, how many SSD reads hit experts that had been on the GPU earlier in the session.
   That share is the most that change 1 below can save.

If file reads per token are already low after step 3, stop: the static tier plus a saved profile is good enough.

## Design: an adaptive RAM tier, changed in batches

Invariant (unchanged): every expert is in **exactly one** of {a GPU slot, the RAM copy, file-only}; the file always
has the bytes. RAM and GPU never both hold an expert.

### 1. Demote into RAM instead of dropping (fixes gap 1)

When the GPU tier evicts `out` and `in` comes from the SSD, keep `out` by giving it the RAM slot of the **coldest RAM
expert of the same layer** (blob sizes are per layer, so a same-layer slot always fits; the same rule the GPU tier's
exchange already follows). That victim falls to file-only.

- Bytes: `out` is already copied device-to-host into an exchange buffer for ordinary exchanges (`a_back`,
  `generate.cpp` ~7672); reuse that copy, then `memcpy` into the victim's offset.
- Bookkeeping: a new `detail::demote_into_complement(offsets, victim, out)`: `out` takes `victim`'s offset, `victim`
  becomes `kNoCacheComplement`. Unit-test it like `exchange_cache_complement` (`expert_source.cpp:213`).
- After the swap, `release()` the file pages of the expert now in RAM (`expert_source.cpp:2130`) so the OS file cache
  does not hold a second copy. Today `release()` only works on Windows (`STRATA_FILE_RELEASE=1`) and returns 0 on
  Linux (`expert_source.cpp:2176-2179`); add the Linux branch with `posix_fadvise(POSIX_FADV_DONTNEED)` on the file
  range (for a mapped view, `madvise(MADV_DONTNEED)` drops this process's mapping but not the page cache).

### 2. Promote hot SSD experts into RAM (fixes gap 2)

- **Counters:** per `(layer, expert)`, count CPU-path reads that went to the **file**, decayed each round by the same
  `--adapt-decay` as the GPU tier. The GPU tier's routing counts (`a_usage`) already exist; this only adds a "where was
  it read from" split.
- **Decision once per round, not per token (your batching idea):** every `--ram-adapt-every` windows (new flag,
  default off), per layer, pair the hottest file-only experts with the coldest RAM experts, and keep a pair only when
  `file_hits(in) >= ram_use(victim) + hysteresis` (start with the GPU tier's 1.5, `generate.cpp:7657`). Cap the round
  at `--ram-adapt-swaps` promotions.
- **One batched read per round:** sort the round's promotions by file offset and read them with one `copy_blobs`
  call, so neighbouring experts merge into large sequential requests. In the pack's `experts.bin` an expert sits at
  `layer_offsets_[layer] + e * blob_bytes` (`expert_source.cpp:1608`), so a layer's experts are contiguous; with a GGUF
  read in place each expert is three slices (gate, up, down), so merging works per role. NVMe does far better on a few
  large reads than on many 1-2 MB random ones.
- **Applied between decode windows on a helper thread**, like `--adapt-async 1`: read into staging buffers, then flip
  ownership in the offset table at a window boundary, never while the CPU pool may be reading a victim blob. Reuse
  the lag rule (`STRATA_ADAPT_LAG`, #764) so a swap takes effect only after the windows that could still use the old
  bytes have finished.

### 3. Batching on the PCIe side too

GPU-tier swaps are already decided per round and capped (`--adapt-swaps`), so they are batched decisions. Two
cheap additions:

- Submit a round's host-to-device copies together. `STRATA_DMA_BATCH=1` (`cudaMemcpyBatchAsync`,
  docs/BATCHED_DMA.md) exists for the verifier's uploads; the measured gain there was in submission time (-83%), not
  in transfer time (-1%), so expect little throughput change. The bytes moved are the same.
- The lever that actually reduces PCIe bytes is **swapping less**: a larger round with stronger hysteresis makes fewer,
  better swaps. Expose the round size and hysteresis for the RAM tier separately from the GPU tier and calibrate both
  (`--calibrate` already does this for `--adapt-every` / `--adapt-swaps`).

### 4. Warm starts

Save the RAM tier's ranking with the existing profile save (`--expert-profile-save`), so a restart begins with the
learned RAM set instead of the shipped profile's.

## Work, in pull-request-sized steps

1. `demote_into_complement` + unit tests (CPU only, no GPU needed).
2. Demotion on eviction under a budget, behind `STRATA_RAM_DEMOTE=1`; default byte-identical.
3. File-hit counters and the stats line (`ram tier: N promoted, M demoted, K file reads this request`).
4. Batched promotion rounds (`--ram-adapt-every`, `--ram-adapt-swaps`), offset-sorted `copy_blobs`, applied between
   windows with the lag rule.
5. Linux page release for promoted experts.
6. Profile save of the RAM tier.

Each step is opt-in. Build CUDA and HIP for every file touched (AGENTS.md).

## Measurements to report

Same transcript, same day, same build, budget fixed:

| arm | decode tok/s | prompt tok/s | file reads / request | SSD MB/s |
|---|---|---|---|---|
| static budget (today) | | | | |
| + demote on eviction | | | | |
| + batched promotion | | | | |

Plus round cost (ms per round on the helper thread) and how often a window waited for a round.

## Risks

- **Locality may be weak.** If routing over a session is spread evenly, an adaptive RAM tier cannot beat a good static
  profile. Step 0 decides this.
- **Pinned memory is fixed at start**, so the tier must reuse offsets (same-layer swaps), never allocate or free.
  That is why promotion pairs within a layer.
- **Races**: a victim blob may be in use by the CPU pool. Only flip ownership at window boundaries with the lag rule;
  add a stress test that runs rounds every window and checks every computed blob against the file's bytes.
- **SSD wear**: reads only, no writes, so no wear concern.
- **Page cache double counting**: without releasing file pages, the OS may hold a promoted expert twice. Measure RSS
  plus file cache before and after.

## Effort and expected result

Medium: one subsystem (`FileExpertSource` + the adaptive loop in `generate.cpp`), no new process. The expected gain
depends entirely on step 0: the fewer file reads per token the static budget leaves, the less there is to win. On the
X570 box (64 GB) IQ3_S fits without a budget at all, so this plan matters most for the 32 GB box, and for UD-IQ4_XS /
UD-Q4_K_XL on 64 GB.
