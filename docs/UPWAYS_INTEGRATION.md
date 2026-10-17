# Pathways: Gateway (4-Pass) vs. Upways (3-Pass) Architecture & Integration

This document records the architectural separation, shader contracts, descriptor layouts, memory footprints, and verification protocols for the two distinct real-time denoising and super-resolution pipelines in **Pathways**:

1. **`--denoiser gateway` / `--scaler gateway` ([`GatewayPipeline`](file:///home/naoki/Development/Pathways/src/rt/GatewayPipeline.cpp))**:
   - **4-Pass Parameter-Guided Analytical E-SVGF Pipeline** (`1,072` FP16 weights = `2,144` bytes).
   - Uses diffuse-only demodulation, a single-quad $16 \to 32 \to 16$ WMMA MLP to modulate analytical SVGF filter parameters, a 2-step dilated $3\times 3$ À-Trous wavelet cascade ($s \in \{2, 6\}$), and a single-stage $3\times 3$ subpixel upscale resolve.
2. **`--denoiser upways` / `--scaler upways` ([`UpwaysPipeline`](file:///home/naoki/Development/Pathways/src/rt/UpwaysPipeline.cpp))**:
   - **3-Pass Combined Neural Spatiotemporal Reconstructor & Multi-Scale Kernel-Predicting Upscaler** (`101,696` FP16 weights = `203,392` bytes).
   - Eliminates the À-Trous wavelet cascade (`upways_atrous.comp` does not exist). Uses co-designed diffuse + metallic-Fresnel specular demodulation, dielectric-glass-aware firefly suppression, a Shift-Invariant 3-Scale Spatiotemporal Recurrent WMMA U-Net ($13\times 13$ quad = $26\times 26$ pixel receptive field) with variance/history-gated direct log-radiance refinement, and a fused 2-stage hierarchical multi-scale neural KPN ($s=3$ coarse with 5-tap coplanar cardinal extremum suppression + $s=1$ fine = hole-free $9\times 9$ / `81`-tap receptive field) inside shared LDS.

---

## 1. Side-by-Side Architectural Separation

| Component | `--denoiser gateway` (`GatewayPipeline`) | `--denoiser upways` (`UpwaysPipeline`) |
| :--- | :--- | :--- |
| **Shader Files** | `gateway_common.glsl`, `gateway_temporal.comp`, `gateway_neural_guide.comp`, `gateway_atrous.comp`, `gateway_upscale_resolve.comp` | `upways_common.glsl`, `upways_temporal.comp`, `upways_neural_guide.comp`, `upways_upscale_resolve.comp` (**Zero `upways_atrous.comp`**) |
| **Compute Passes** | **4 Passes** (5 dispatches: Temporal $\to$ Quad MLP $\to$ À-Trous $s=2$ $\to$ À-Trous $s=6$ $\to$ Upscale) | **3 Passes** (3 dispatches: Temporal $\to$ 3-Scale U-Net $\to$ Fused 2-Stage Multi-Scale KPN Upscale) |
| **Specular Demodulation** | None (diffuse-only demodulation; raw specular filtered directly) | **Co-Designed Metallic-Fresnel Specular Demodulation** (`demodulateSpecular` / `remodulateSpecular` using per-pixel `metallic`) |
| **Neural Network** | Single-quad 2-layer MLP ($16 \to 32 \to 16$, zero spatial gather, zero recurrence) | **3-Scale Spatiotemporal Recurrent U-Net** ($16\times 16 \to 8\times 8 \to 4\times 4$ quads, 5-ch hidden state, direct log-radiance refinement) |
| **Weight Footprint** | `1,072` FP16 params = **`2,144` bytes** (`gateway_default_weights.hpp`) | `101,696` FP16 params = **`203,392` bytes** (`upways_weights.hpp` / `upways_default_weights.hpp`) |
| **Spatial Filtering & Upscaling** | 2 global `rgba16f` À-Trous image passes ($s=2, 6$) + 1-stage $3\times 3$ resolve | **Fused 2-Stage Multi-Scale Neural KPN in `24x24` LDS** (Stage 1 coarse $s=3$ + Stage 2 fine $s=1$ = hole-free $9\times 9$ / `81`-tap KPN in one pass) |

---

## 2. Upways 3-Pass Descriptor & Memory Contract ([`UpwaysPipeline.cpp`](file:///home/naoki/Development/Pathways/src/rt/UpwaysPipeline.cpp))

### Pass 1: [`upways_temporal.comp`](file:///home/naoki/Development/Pathways/shaders/compute/upways_temporal.comp) (`16x16` Workgroup, `10,944 B` LDS)
- **Bindings `0..15`**: Storage images (`uCurrentDiffuse`, `uCurrentSpecular` with `.a = metallic`, `uNormalDepth`, `uPrevNormalDepth`, `uMotionVectors`, `uAlbedoRoughness`, `uSpecularMotion`, `uPrevSpecularMotion`, `uHistoryDiffuseIn`, `uHistorySpecularIn`, `uHistoryMomentsIn`, `uTemporalDiffuseOut`, `uTemporalSpecularOut`, `uReprojDiffuseOut`, `uReprojSpecularOut`, `uHistoryMomentsOut`).
- **Binding `16`**: `NeuralFeaturesOutBuffer` (`16` `float16_t`s = `2 * uvec4` per $2\times 2$ quad, tiled in $16\times 16$ quad blocks).

### Pass 2: [`upways_neural_guide.comp`](file:///home/naoki/Development/Pathways/shaders/compute/upways_neural_guide.comp) (`32x8` Workgroup, 8 Wave32 Subgroups, `16,384 B` LDS)
- **Binding `0`**: `NeuralFeaturesInBuffer`
- **Binding `1`**: `WeightsBuffer` (`203,392` bytes: `396` $16\times 16$ FP16 WMMA tiles + 9 bias vectors)
- **Bindings `2..7`**: `uTemporalDiffuseInOut`, `uTemporalSpecularInOut`, `uReprojDiffuseIn`, `uReprojSpecularIn`, `uHistoryDiffuseOut`, `uHistorySpecularOut`
- **Binding `8`**: `UpscaleKernelOutBuffer` (`3 * uvec4` = `48` bytes = `24` `float16_t`s per $2\times 2$ quad: 9 neural kernel weights `w0..w8`, `detailGain`, 4 neural edge-stopping parameters `phiZ, phiN, phiLD, phiLS`, and 5 recurrent hidden state channels `hNew0..hNew4`).

### Pass 3: [`upways_upscale_resolve.comp`](file:///home/naoki/Development/Pathways/shaders/compute/upways_upscale_resolve.comp) (`16x16` Workgroup, `19,884 B` LDS)
- **Bindings `0..1`**: `uDenoisedDiffuse` (`m_tempDiffImage`), `uDenoisedSpecular` (`m_tempSpecImage`)
- **Bindings `2..4`**: `uNormalDepth`, `uAlbedoRoughness`, `uMotionVectors`
- **Binding `5`**: `UpscaleKernelInBuffer` (`3 * uvec4` per $2\times 2$ quad)
- **Bindings `6..8`**: `uDisplayHistoryIn`, `uDisplayOut`, `uDisplayHistoryOut`
- **Binding `9`**: `uSpecularMetallic` (bound to `m_cachedDemodSpecularView` where `.a` carries exact per-render-pixel `metallic` $\in [0, 1]$ for `remodulateSpecular`).

---

## 3. Engine Verification Commands

### Automated Invariant Tests
```bash
/home/naoki/Development/Pathways/build/linux-release/bin/test_upways_pipeline
```
Verifies:
- Real shader push constant struct layouts for both `Upways` (3-pass: `20B`, `36B`, `32B`) and `Gateway` (4-pass: `20B`, `36B`, `12B`, `32B`).
- Invertible log transform mathematical roundtrip across $0$ to $10,000$ nits.
- Co-designed diffuse albedo and metallic-Fresnel specular demodulation/remodulation roundtrip.
- `Upways` (`101,696` FP16 params = `203,392` bytes) and `Gateway` (`1,072` FP16 params = `2,144` bytes) weight buffer offsets and embedded weight arrays.
- Disocclusion confidence gating and 90-degree camera flick rejection.
- 3-Pass `Upways` LDS footprints (`Pass 1 = 10.69 KB`, `Pass 2 = 16.00 KB`, `Pass 3 = 19.42 KB`, all `< 32 KB`).
- CLI flag separation (`--denoiser gateway` / `--scaler gateway` vs. `--denoiser upways` / `--scaler upways`).
