# Pathways: Upways 3.0 FP8 WMMA Neural Reconstructor & Continuous Subpixel Resolve

This document specifies the shader contracts, descriptor layouts, memory footprint, and verification suite for **Upways 3.0** ([`UpwaysPipeline`](file:///home/naoki/Development/Pathways/src/rt/UpwaysPipeline.cpp)), Pathways' in-development real-time RDNA 4 neural denoising and super-resolution pipeline (`--denoiser upways` / `--upscaler upways` / `--scaler upways <preset>`).

---

## 1. Architectural Overview

Upways 3.0 is a **3-pass** neural spatiotemporal reconstructor and continuous subpixel upscaler (`73,686` parameters = `89,536` bytes: `69,632 B` FP8 E4M3 WMMA 16×16 blocks + `19,904 B` FP32 scales, biases, 3×3 depthwise filters, and residual `skip_scale`, [`upways_weights.hpp`](file:///home/naoki/Development/Pathways/src/rt/upways_weights.hpp)):
- **Co-Designed Demodulation**: Separate diffuse albedo demodulation and metallic-Fresnel specular demodulation (`demodulateDiffuse` / `demodulateSpecular` / `remodulateDiffuse` / `remodulateSpecular`) paired with invertible log-irradiance compression (`logIrradianceEncode` / `logIrradianceDecode`) in [`upways_common.glsl`](file:///home/naoki/Development/Pathways/shaders/compute/upways_common.glsl).
- **6-Stage 3-Scale RDNA 4 FP8 E4M3 WMMA Spatiotemporal Backbone**: Executes on `VK_KHR_cooperative_matrix` + `VK_EXT_float_e4m3` Wave32 subgroups (`32x4` workgroup = 4 Wave32 subgroups covering `16x8` pixels) across `L0` (render), `L1` ($1/2$), and `L2` ($1/4$) scales, including a dilated ($d=3$) bottleneck, widened `DEC0` (`32 -> 64 -> 32`) inverted-bottleneck decoder, and a 16-channel GRU-gated recurrent hidden state stored in a 4-layer `RGBA16F` `image2DArray`.
- **Continuous Subpixel Ray-Locus Gather & Catmull-Rom Resolve**: Register-resident (`0 B` LDS) display-rate pass performing a 9-tap continuous subpixel ray-locus gather (`w0..w8`, subpixel phase offset, log residual refinement) fused with a 5-tap Jimenez anti-ringing Catmull-Rom high-resolution temporal resolve and zero-copy ping-pong history swap. Runs either coupled with super-resolution (`--upscaler upways`, Halton base-(2,3) camera jitter enabled) or standalone at native `1.0x` resolution (`--denoiser upways`).
- **Default Neural Execution & Debug Fallback**: By default (`config.upways_heuristic_fallback == false`), the trained FP8+FP32 weights drive reconstruction. Passing `--upways-heuristic` enables the analytical bilateral fallback for A/B debugging.
- **Zero Idle VRAM**: `UpwaysPipeline` is lazily allocated on first activation and released when inactive.

---

## 2. 3-Pass Descriptor & Memory Contract ([`UpwaysPipeline.cpp`](file:///home/naoki/Development/Pathways/src/rt/UpwaysPipeline.cpp))

All three passes share the unified `56`-byte [`UpwaysPushConstants`](file:///home/naoki/Development/Pathways/src/rt/UpwaysPipeline.hpp#L20-L36) block.

### Pass 1: [`upways_temporal.comp`](file:///home/naoki/Development/Pathways/shaders/compute/upways_temporal.comp) (`16x16` Workgroup, `6,624 B` / `6.47 KB` LDS)
- **Bindings `0..6`** (`storageImage`): Current render-rate G-buffer & radiance inputs (`uDemodDiffuse`, `uDemodSpecular`, `uAlbedoRoughness`, `uNormalDepth`, `uSurfaceMotion`, `uSpecularMotion`, `uAuxMetadata`).
- **Bindings `7..11`** (`combinedImageSampler`): Linear-sampled ping-pong history (`uPrevIrrDiffHr`, `uPrevIrrSpecHr`, `uPrevGeomNd`, `uPrevGeomHitT`, `uPrevNeuralHidden` `sampler2DArray`).
- **Bindings `12..19`** (`storageImage`): Current geometry history & render-rate transients (`uCurrGeomNd`, `uCurrGeomHitT`, `uTransientLogDiff`, `uTransientLogSpec`, `uTransientGuideDiff`, `uTransientGuideSpec`, `uTransientDilatedMv`, `uTransientTemporalConf`).
- **Binding `20`** (`storageBuffer`): `PyramidFp8Buf` (`PYR_X0` 32-ch + `PYR_HW0` 16-ch FP8 E4M3 packed feature pyramid).

### Pass 2: [`upways_neural_guide.comp`](file:///home/naoki/Development/Pathways/shaders/compute/upways_neural_guide.comp) (`32x4` Workgroup, 4 Wave32 Subgroups, 6 Specialized Stages `SPEC_STAGE_ID = 0..5`, $\le 32\text{ KB}$ LDS)
- **Binding `0`** (`storageBuffer`): `WeightFp8Buf` (`69,632 B` packed `uvec4` FP8 E4M3 16×16 WMMA weight blocks).
- **Binding `1`** (`storageBuffer`): `WeightFp32Buf` (`19,904 B` FP32 per-channel scales, biases, 3×3 depthwise filters, and `skip_scale`).
- **Binding `2`** (`storageBuffer`): `PyramidFp8Buf` (`L0` / `L1` / `L2` multi-scale FP8 E4M3 intermediate activations).
- **Binding `3`** (`storageBuffer`): `SubpixelParamsBuf` (16-ch FP16 subpixel gather & residual parameters per render pixel = `2 * uvec4`).
- **Binding `4`** (`storageImage`): `uCurrNeuralHidden` (4-layer `RGBA16F` `image2DArray` storing 16-ch GRU-updated recurrent hidden state).

### Pass 3: [`upways_upscale_resolve.comp`](file:///home/naoki/Development/Pathways/shaders/compute/upways_upscale_resolve.comp) (`16x16` Workgroup, `0 B` LDS Register-Resident)
- **Bindings `0..6`** (`storageImage`): Render-rate transients & normals (`uNormalDepth`, `uTransientLogDiff`, `uTransientLogSpec`, `uTransientGuideDiff`, `uTransientGuideSpec`, `uTransientDilatedMv`, `uTransientTemporalConf`).
- **Binding `7`** (`storageBuffer`): `SubpixelParamsBuf` (`2 * uvec4` FP16 neural parameters per render pixel).
- **Binding `8`** (`storageBuffer`): `WeightFp32Buf` (reads `SKIP_SCALE_F32_OFFSET` for diffuse/specular residual skip scaling).
- **Bindings `9..12`** (`combinedImageSampler`): High-resolution ping-pong history (`uPrevIrrDiffHr`, `uPrevIrrSpecHr`, `uPrevGuideDiffHr`, `uPrevGuideSpecHr`).
- **Bindings `13..17`** (`storageImage`): Resolved display output `uOutputColor` + next-frame high-resolution ping-pong history (`uCurrIrrDiffHr`, `uCurrIrrSpecHr`, `uCurrGuideDiffHr`, `uCurrGuideSpecHr`).

---

## 3. Verification Suite

```bash
ctest --preset linux-test -R "UpwaysPipeline|FidelityFxFeatures"
```
Verifies:
- Unified `56`-byte `UpwaysPushConstants` ABI and multi-scale `computeUpwaysPyramidDims` alignment.
- Invertible log-irradiance encode/decode roundtrip across $0$ to $10,000$ nits ($< 1\%$ relative error).
- Co-designed diffuse albedo and metallic-Fresnel specular demodulation/remodulation roundtrip.
- `73,686`-parameter (`89,536`-byte: `69,632 B` FP8 + `19,904 B` FP32) weight buffer layout, embedded header vs `data/models/upways_weights.bin` byte-exact integrity, and finite non-zero parameter checks.
- End-to-end RDNA 4 GPU execution (`0` Vulkan validation errors/warnings), finite output verification, and PSNR / edge-retention regression checks in both `1.0x` native denoiser and `1.5x` super-resolution modes.
