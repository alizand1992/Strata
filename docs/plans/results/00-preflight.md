# Phase 0 pre-flight (plan section 2)

Plan: [../OPTION_A_EXCLUSIVE_RAM.md](../OPTION_A_EXCLUSIVE_RAM.md). Box: the B550 Tomahawk MAX box (Ryzen 9 3900X,
32 GB). Date: 2026-10-10. Repository at `fb58e0d` (main).

**Result: blocked. Nothing in Phase 0 or later can run on this box yet.** Four things are missing, and three of them
need root or the owner's call (see "Blockers" at the end).

## Output

```
$ nvidia-smi --query-gpu=name,pci.bus_id,memory.total,driver_version --format=csv
name, pci.bus_id, memory.total [MiB], driver_version
NVIDIA GeForce RTX 5060 Ti, 00000000:2B:00.0, 16311 MiB, 580.178.04

$ ls -l /dev/kfd /dev/dri/renderD*
crw-rw----+ 1 root render 226, 128 Oct  7 08:44 /dev/dri/renderD128
crw-rw----+ 1 root render 226, 129 Oct  7 08:44 /dev/dri/renderD129
crw-rw----  1 root render 508,   0 Oct  7 08:44 /dev/kfd

$ getfacl /dev/kfd        # no ACL entry for the user (the render nodes have one: user:ali:rw-)
user::rw-  group::rw-  other::---

$ id
uid=1000(ali) gid=1000(ali) groups=1000(ali),4(adm),24(cdrom),27(sudo),30(dip),46(plugdev),100(users),114(lpadmin)

$ for n in /sys/class/kfd/kfd/topology/nodes/*; do ...name, gfx_target_version...; done
nodes/0: (CPU)          gfx_target_version 0
nodes/1: ip discovery   gfx_target_version 120000      # gfx1200, the RX 9060 XT

$ sudo lspci -vv
sudo: a password is required          # link state read from sysfs instead, below

$ nvcc --version
nvcc: command not found               # also no /usr/local/cuda*; no nvcc anywhere on the disk
$ ls /opt/rocm; /opt/rocm/bin/hipcc --version
No such file or directory             # no hipcc or libamdhip64 anywhere on the disk

$ free -g
               total        used        free      shared  buff/cache   available
Mem:              31          21           1           0           9           9
Swap:              7           3           4
$ uname -r
7.0.0-38-generic
```

### PCIe links (sysfs, `current_link_speed` / `current_link_width`)

| Device | Path | Current | Max |
|---|---|---|---|
| RTX 5060 Ti | `00:03.1 -> 2b:00.0` (CPU lanes) | 2.5 GT/s x8 (idle downclock) | 32 GT/s x16 (card is x8) |
| RX 9060 XT, chipset port | `00:01.2` (CPU -> B550 chipset) | 8.0 GT/s x4 | 16 GT/s x8 |
| | `02:00.2 -> 03:00.0` (chipset bridge, AMD 43e9/43ea) | 8.0 GT/s x4 (Gen3 x4 max) | 8.0 GT/s x4 |
| | `04:00.0 -> 05:00.0 -> 06:00.0` (the card's internal switch) | 2.5 GT/s x4 idle; 32 GT/s x16 inside the card | |

So the 9060 XT's real link is the chipset's **x4 Gen3** (~3.9 GB/s), as the plan's machine table says. The
32 GT/s x16 that `06:00.0` reports is the card's internal switch, not the slot.

### What else is on the GPUs right now

```
$ nvidia-smi --query-compute-apps=pid,used_memory --format=csv
68320, 15418 MiB                       # of 16311 MiB
$ cat /sys/class/drm/card*/device/mem_info_vram_used   # 9060 XT
59891712                               # ~57 MiB, the card is free
```

PID 68320 is a `llama-server` (Vulkan build, `~/llama.cpp/build-release`) serving
`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS` with `--n-cpu-moe 24`, 12.5 GB RSS, ~6 cores busy. It holds the whole RTX and
most of the RAM that Phase 0 measures.

### Model files on disk

`~/models/qwen-3.8-next-gsq/`: GSQ-RCO IQ3_XXS (2 parts, 75.8 GB) and IQ2_XS. **No Coder (IQ1_M)**, which runs 0a,
0d, Gate 4 and arms A-C use (setup lists it at 58.4 GB). No Strata `.venv`, no engine build.

## Blockers

| # | What | Needed for | Who can fix it |
|---|---|---|---|
| 1 | The user is not in `render`/`video`, so `/dev/kfd` cannot be opened: HIP sees no device | 0d, Gate 1, Gate 3 | owner: `sudo usermod -aG render,video ali`, then log out and in |
| 2 | No CUDA toolkit (`nvcc`) | Gate 1 (`spike_cuda.cu`), the CUDA engine build for 0a-0c, the CUDA half of Gate 2 | owner: CUDA 13 from NVIDIA's apt repo (`cuda-toolkit-13-0`); Ubuntu's `nvidia-cuda-toolkit` is 12.0, too old for sm_120 |
| 3 | No ROCm | Gates 1-3, 0d | agent: setup installs TheRock ROCm into `.venv` without sudo (~10 GB) |
| 4 | The RTX and ~12 GB of RAM are held by the running llama-server | every Phase 0 run and Gate 1's RTX half | owner: stop it for the measuring session |
| 5 | Coder (IQ1_M) not downloaded | 0a, 0d, Gate 4, arms A-C | owner: OK a 58.4 GB download (103 GB free on `/`) |

## Changes since (2026-10-10)

- The owner installed `cuda-toolkit-13-0` (nvcc 13.0.88) and added the user to `render`/`video` (applies after a new
  login).
- ROCm 7.10.0a20251120 installed into `.venv` by setup's own `rocm_root("gfx1200")` (TheRock `gfx120X-all` wheels).
- Coder (IQ1_M) downloaded by `./setup.sh --yes --family coder --model IQ1_M --backend cuda --build --no-start`
  into `~/Strata-data`. Setup's disk check passed when it started (need ~91 GB, 93 free), but the ROCm install took
  ~18 GB of that afterwards, so setup was stopped after the download and resumed once the owner freed 39 GB (the
  unused GSQ-RCO IQ2_XS shard 1; its shard 2 was a hard link of IQ3_XXS's shard 2 and is kept under that name).
- Disk left for IQ3_XXS: its own `experts.bin` (~43 GB) does not fit beside the Coder's. The helper needs a pack with
  an `experts.bin` (`ExpertSource::blob()` returns null for a GGUF read in place, `src/core/expert_source.cpp:2115`),
  so Gate 3 uses the Coder's pack.
