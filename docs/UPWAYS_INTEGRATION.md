# Pathways: Upways 4.0 & 5.0 Neural Reconstructor Integration

This document summarizes the architecture, runtime selection, shader passes, compact ML G-buffer contract, and multi-GPU Band-Parallel execution for **Upways 4.0** ([`UpwaysPipeline`](file:///home/naoki/Development/Pathways/src/rt/UpwaysPipeline.cpp)) and **Upways 5.0 Pure-ML** ([`UpwaysV5Engine`](file:///home/naoki/Development/Pathways/src/rt/UpwaysV5Engine.cpp)), activated via `--denoiser upways` (`--upways`), `--upscaler upways`, or `--scaler upways <preset>`.

---

## 1. Runtime Version & Precision Selection

- **Version selection (`--upways-version auto|4|v4|5|v5`, default `auto`)**:
  - Handled by [`UpwaysPipeline::resolveVersion`](file:///home/naoki/Development/Pathways/src/rt/UpwaysPipeline.cpp#L262-L282).
  - `auto` selects **Upways 4.0** by default unless a custom `--upways-weights` file with exact size `190,208` bytes (`upways_v5::TOTAL_WEIGHT_BYTES`) is supplied.
  - `5` / `v5` explicitly selects **Upways 5.0 Pure-ML** using embedded or disk weights ([`upways_v5_weights.bin`](file:///home/naoki/Development/Pathways/data/models/upways_v5_weights.bin) / [`upways_v5_weights_int8.bin`](file:///home/naoki/Development/Pathways/data/models/upways_v5_weights_int8.bin)).
- **Precision selection (`--upways-precision auto|fp8|int8|fp16`, default `auto`)**:
  - Handled by [`UpwaysPipeline::selectPrecision`](file:///home/naoki/Development/Pathways/src/rt/UpwaysPipeline.cpp#L284-L324).
  - Queries hardware Wave32 16×16×16 `VK_KHR_cooperative_matrix` WMMA support (`FP8` → `INT8` → `FP16`).
  - **Upways 4.0** supports `FP8`, `INT8`, and `FP16` WMMA fallback.
  - **Upways 5.0** supports `FP8` (E4M3fn) and `INT8` (QAT) only; requesting `fp16` on V5 logs a warning and falls back to `FP8` (or `INT8` if FP8 is unsupported).

---

## 2. Upways 4.0 vs Upways 5.0 Pure-ML

| Feature | Upways 4.0 (`UpwaysPipeline`) | Upways 5.0 Pure-ML (`UpwaysV5Engine`) |
|---|---|---|
| **Parameters** | `17,392` (`upways::ACTIVE_PARAMETER_COUNT`) | `158,400` (`upways_v5::ACTIVE_PARAMETER_COUNT`) |
| **Weight Blob** | `24,064` B (`15,616` B FP8/INT8 + `8,448` B FP32) | `190,208` B (`150,016` B FP8/INT8 + `40,192` B FP32) |
| **Weight Headers / Metadata** | [`upways_weights.hpp`](file:///home/naoki/Development/Pathways/src/rt/upways_weights.hpp) | [`upways_v5_weights.hpp`](file:///home/naoki/Development/Pathways/src/rt/upways_v5_weights.hpp) |
| **WMMA Precision Tiers** | FP8 E4M3fn, INT8 QAT, FP16 fallback | FP8 E4M3fn, INT8 QAT |
| **Compute Passes** | **4 passes**: `upways_denoise_prep.comp`, `upways_denoiser_wmma.comp`, `upways_upscaler_wmma.comp`, `upways_upscale_resolve.comp` | **5 passes**: `upways_v5_prep.comp`, `upways_v5_unet_wmma.comp`, `upways_v5_affinity.comp`, `upways_v5_upscaler_wmma.comp`, `upways_v5_resolve.comp` |
| **Push Constants** | `80`-byte [`UpwaysPushConstants`](file:///home/naoki/Development/Pathways/src/rt/UpwaysPipeline.hpp#L37-L60) | `80`-byte [`UpwaysV5PushConstants`](file:///home/naoki/Development/Pathways/src/rt/UpwaysV5Engine.hpp#L26-L49) |
| **Super-Resolution** | Continuous DRS (`--render-scale`) via upscaler + resolve passes | Native 1× denoiser or DRS (`--scaler upways <preset>`) |

---

## 3. Compact Live ML G-Buffer (41 B/px)

When `ML_CAPTURE_COMPACT_LIVE_BIT` is enabled for live Upways reconstruction, primary and secondary wavefront kernels (plus `nrc_encode_infer.comp`) emit a 6-image compact G-buffer (`41 B/px` total at render resolution):
1. `uMlIrradiance` (`RG32UI`, 8 B/px): packed `RGB9E5` indirect diffuse + indirect specular irradiance ([`rgb9e5.glsl`](file:///home/naoki/Development/Pathways/shaders/compute/rgb9e5.glsl))
2. `uMlPrimaryGuide` (`RGBA32UI`, 16 B/px): packed primary albedo/spec-albedo, roughness, normal, depth, reactive mask
3. `uMlSurfaceMv` (`R32UI`, 4 B/px): packed 2D surface motion vectors
4. `uMlDiffHitDist` (`R8_UNORM`, 1 B/px): first-bounce diffuse hit distance
5. `uMlSpecVmv` (`RG32UI`, 8 B/px): first-bounce specular virtual motion vectors & hit distance
6. `uMlSkipCompact` (`R32UI`, 4 B/px): packed direct lighting / miss / emissive skip buffer

---

## 4. Multi-GPU Upways Band-Parallel Execution

When both `--mgpu` and Upways (`4.0` or `5.0`) are active, [`MultiGpuCoordinator`](file:///home/naoki/Development/Pathways/src/mgpu/MultiGpuCoordinator.cpp) and [`MultiGpuManager`](file:///home/naoki/Development/Pathways/src/mgpu/MultiGpuManager.cpp) partition the frame into top (GPU 0) and bottom (GPU 1) horizontal bands with shared-border halo overlap instead of 2D checkerboard tiling, preserving full-neighborhood spatiotemporal filtering without cross-GPU tile seams:

- **`--upways-band-halo <8..64>`** (default `24`, rounded to a multiple of `4`): render-resolution LR halo rows computed past the split boundary on each GPU so convolution, affinity, and temporal gather filters match single-GPU boundaries across the horizontal split.
- **`--upways-band-balance` / `--no-upways-band-balance`** (default enabled): dynamically adjusts `topFraction` via [`computeUpwaysBandSplit`](file:///home/naoki/Development/Pathways/src/rt/UpwaysPipeline.hpp#L87) to balance frame times between GPU 0 and GPU 1.
- **Fused Pack & Stitch**: GPU 1 packs only its kept HR rows into `RGB9E5` via [`upways_band_pack.comp`](file:///home/naoki/Development/Pathways/shaders/compute/upways_band_pack.comp) (or directly from the final Upways pass when `UPWAYS_FLAG_FUSED_BAND_PACK` / `UPWAYS_V5_FLAG_FUSED_BAND_PACK` is set), transfers them over P2P DMA-BUF or zero-copy pinned host memory, and GPU 0 stitches them into the final output via [`upways_band_stitch.comp`](file:///home/naoki/Development/Pathways/shaders/compute/upways_band_stitch.comp).

---

## 5. Verification Suite

```bash
ctest --test-dir build/linux-release -R "UpwaysPipeline|MgpuFramePacing" --output-on-failure
```

Verifies:
- `80`-byte push constant ABI (`UpwaysPushConstants` / `UpwaysV5PushConstants`), pyramid alignment, log1p/expm1 & `RGB9E5` roundtrips, and [`computeUpwaysBandSplit`](file:///home/naoki/Development/Pathways/src/rt/UpwaysPipeline.hpp#L87) invariants.
- Embedded weight header byte-for-byte parity with [`data/models/upways_weights*.bin`](file:///home/naoki/Development/Pathways/data/models) and [`data/models/upways_v5_weights*.bin`](file:///home/naoki/Development/Pathways/data/models).
- Upways 4.0 and 5.0 SPIR-V shader validation, `resolveVersion` / `selectPrecision` behavior, live RDNA 4 GPU execution (`0` Vulkan validation errors), and mGPU frame pacing / Band-Parallel stitching.
