# Phase 5 - config (manual)

Plan: [../OPTION_A_EXCLUSIVE_RAM.md](../OPTION_A_EXCLUSIVE_RAM.md), Phase 5. Only the plan's minimum is done here: a
working server config written by hand. Setup does not detect the two-vendor case or build the helper; that part of
Phase 5 is optional and was not done.

## What you need

1. The CUDA engine built from this branch (setup's usual build, or `cmake -S . -B build-cuda -DSTRATA_ENABLE_CUDA=ON`).
2. The helper from a HIP build of the same branch, built into a different directory (`build-hip`) so setup's engine is
   not replaced. On the B550 box, with setup's ROCm in `.venv`:

   ```sh
   R=~/Strata/.venv/lib/python3.12/site-packages/_rocm_sdk_devel
   L=~/Strata/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx120X_all
   PATH=~/Strata/.venv/bin:$PATH ROCM_PATH=$R HIP_PATH=$R cmake -G Ninja -DCMAKE_MAKE_PROGRAM=$HOME/Strata/.venv/bin/ninja \
     -S . -B build-hip -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
     -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DCMAKE_HIP_COMPILER=$R/llvm/bin/clang++ -DCMAKE_HIP_COMPILER_ROCM_ROOT=$R \
     "-DCMAKE_PREFIX_PATH=$R;$L" -DCMAKE_HIP_PLATFORM=amd -DHIP_PLATFORM=amd -DSTRATA_NATIVE_EXPERTS=ON \
     "-DCMAKE_HIP_FLAGS=--rocm-path=$R --rocm-device-lib-path=$R/lib/llvm/amdgcn/bitcode"
   cmake --build build-hip --target strata-expert-helper
   ```

   Without `-DHIP_PLATFORM=amd`, ROCm's `hip-config.cmake` stops with "Unexpected HIP_PLATFORM". Without the
   `--rocm-*` flags, clang cannot find the device libraries.

3. Linux, one NVIDIA card (the engine) and one AMD card (the helper), and an expert profile (`--expert-profile`).

## The config

Setup's config (`strata-<model>.json`) with four more engine args and the ROCm library dirs. The helper inherits the
engine's environment, and it needs `libamdhip64` and friends on `LD_LIBRARY_PATH`. The server builds that from
`lib_dirs`. This is arm E of Phase 6 (IQ3_XXS on 32 GB):

```json
{
 "exe": "/home/ali/Strata-p4/build-cuda/strata",
 "args": [
  "--pack", "/home/ali/Strata-data/packs/iq3_xxs",
  "--native", "/home/ali/models/qwen-3.8-next-gsq/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
  "--ple-gguf", "/home/ali/models/qwen-3.8-next-gsq/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf",
  "--expert-profile", "/home/ali/Strata/data/expert-profile.bin",
  "--expert-cache", "auto", "--prefill", "auto", "--spec", "4", "--spec-min-p", "0.5",
  "--mtp", "/home/ali/Strata-data/mtp/rt", "--max-context", "65536", "--kv", "int8",
  "--resident-experts",
  "--external-experts", "/home/ali/Strata-p4/build-hip/strata-expert-helper",
  "--external-experts-device", "0",
  "--external-experts-ram", "exclusive"
 ],
 "lib_dirs": [
  "/usr/local/cuda-13.0/bin", "/usr/local/cuda-13.0/lib64",
  "/home/ali/Strata/.venv/lib/python3.12/site-packages/_rocm_sdk_devel/lib",
  "/home/ali/Strata/.venv/lib/python3.12/site-packages/_rocm_sdk_libraries_gfx120X_all/lib"
 ],
 "env": { "STRATA_XH_SPIN_US": "200" },
 "...": "the rest as setup wrote it (cwd, tokenizer, model_name, log, port, gpu)"
}
```

- `--external-experts-device` is the HIP device index. With one AMD card it is 0: HIP does not count the NVIDIA card.
- `--external-experts-gib X` caps the helper's VRAM. The default is everything free minus 512 MiB.
- `--external-experts-ram inclusive` keeps the helper's experts in the RAM copy too. Prompts are then as fast as
  without the helper, but the model needs the RAM it needed before.
- `STRATA_XH_TRACE=1` adds the helper's round-trip percentiles to the per-request `external:` line and passes
  `--verbose` to the helper. `STRATA_XH_TIMEOUT_MS` (default 2000) is how long one layer waits for the helper. After
  that the CPU computes the layer, and the helper is used again once it has answered. `STRATA_XH_GIVE_UP_MS` (default
  60000) is how long it may go without answering before it counts as gone.

## What it prints

At start (arm E of Phase 6):

```
strata generate: external helper (HIP device 0): 9438 experts, 15.36 GiB, loaded in 39.8 s; with CUDA0's 5341 that is 14779 of 24576 on the GPUs; RAM copy exclusive (leaves them out)
```

Per request (arm E, p1, with `STRATA_XH_TRACE=1`):

```
strata serve: decode pool: 0 expert blobs from outside the RAM copy this request
strata serve: external: 110328 rows, 9900 layers, begin 20.8 ms, wait 1298.8 ms, 8769 layers waited on the helper, timeouts 0; CPU pool 14014 entries; round trip p50 208 us p99 486 us over 9900 layers
```

If the helper stops, one line says so, and its experts are read from the model files from then on:

```
strata: external experts: the helper stopped (<reason>); its experts are read from the model files from now on
```
