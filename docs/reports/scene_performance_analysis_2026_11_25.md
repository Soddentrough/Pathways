# Pathways GPU Performance Engineering & Microarchitectural Analysis (2026-11-25)

**Target Engine**: Pathways v1.24.0 (Wavefront Vulkan 1.4 Path Tracer, Release Clang 20, OpenUSD + FFX SDK)  
**Hardware Under Test**: Single AMD Radeon RX 7900 XTX (`0x1002:0x744c`, RDNA 3 / `gfx1100`, 24,560 MB VRAM, 96 MB Infinity Cache) + cross-target RDNA 4 (`gfx1201`) offline ISA comparison  
**Driver / Toolchain**: AMD Adrenalin `26.11.1 (LLPC)` (`2.0.416`), Vulkan `1.4.359`, RDTS `amdllpc.exe` (LLPC `78.0`) & `shae.exe`  
**Empirical Corpus**: **254 serialized GPU runs** (20 Sweep 1 @ 4K + 20 Sweep 2 @ 1080p + 214 Sweep 3 @ 4K ablations; **0 failures, 0 Vulkan validation errors**) + **122 offline Wave32 shader compilations** (`61` `.spv` binaries across `gfx1100` & `gfx1201`) + full static C++/GLSL pipeline audit, independently cross-validated in [verification_audit.md](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/verification_audit.md).

---

## 1. Executive Summary & Key Findings

Across all 20 canonical Pathways scenes at native 4K (`3840x2160`, 1 SPP, 4 bounces, default flags), single-GPU frame times span **`1.966 ms` (`508.6 FPS`, `16.874 GRays/s` on `DamagedHelmet`)** to **`23.338 ms` (`42.8 FPS`, `1.422 GRays/s` on `cyber-city`)**, with a median of **`10.273 ms` (`~97.3 FPS`)**. Cross-correlating empirical GPU stage/ablation telemetry with offline RDNA 3/4 ISA disassembly and C++/GLSL code inspection yields one foundational conclusion:

> **The primary bottlenecks in Pathways are Shading ALU/BVH/L0-ICache bloat (`40%–77%` of RT frame time), Ray-Traversal + Scattered Queue Bandwidth (`2,784.75 MiB` working set @ 4K, `1.08–3.54 GB/frame` C++-accounted traffic), and Miscalibrated Wavefront Execution Heuristics — NOT pipeline barriers (`Gap1+Gap2+Gap3` = `0.022–0.189 ms`, `<1.2%` @ 4K) or VRAM capacity/batching.**

### 1.1 Headline System Breakdown Across 20 Scenes (4K Default)

| Stage / Metric | Observed 4K Range | Share of Frame Time | Root Hardware & Code Driver |
|---|---:|---:|---|
| **Shading (`Shade` sum, B0..B3)** | `0.880 – 16.737 ms` | **40.2% – 77.7%** | `53.4–85.8 KB` `.text` (>32 KB L0 I-cache), `3–5` inlined BVH sites (`7` w/ ReSTIR), `62.5%–75%` occupancy (`56.25%` on dielectric), Light Tree traversal, `delta_unroll=6` inline loops |
| **Intersection (`Intersect` sum, B0..B2)** | `0.285 – 6.146 ms` | **14.5% – 40.0%** | Non-opaque BLAS splits (`noOpaque` / `gl_RayFlagsNoOpaqueEXT`), Morton `& 0x0FFFu` LSB aliasing + 30-shuffle scatter in `wavefront_intersect.comp`, `RayHit` `32 B` write amplification |
| **Primary RayGen & Classify** | `0.493 – 2.092 ms` | **4.7% – 25.1%** | Redundant `208 B` `Material` reads (`3×`), `144 B` `InstanceGPU` reads (`2–3×`), `VK_KHR_ray_tracing_position_fetch` + primary texture re-sampling discarded before B0, `40,001`-instance TLAS on `PointInstancedMedCity` |
| **Shadow Trace (`Shadow` sum)** | `0.075 – 2.725 ms` | **0.0% – 17.8%** | Active on non-inline bounces (`B1..B3`), transparent shadow `gl_RayFlagsNoOpaqueEXT` forcing all hits through candidate loop (`wavefront_shadow.comp:88-145`) |
| **Barrier / Timestamp Gaps (`Gap1+2+3`)** | `0.022 – 0.189 ms` | **0.40% – 1.12%** | `14–15` barriers + `25+2` timestamps (`VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT`) — measurable, worth fixing for `Shadow`/`Intersect` overlap, but secondary |
| **Fused Accumulation + Tonemap** | `0.128 – 0.143 ms` | **0.6% – 7.2%** | `accum_tonemap_fused` (`14` VGPRs, `100%` occ, `30` transcendentals, `12.5%` VOPD); constant ~`0.137 ms` fixed 4K scanout cost |

### 1.2 Top 7 Highest-Leverage Fundamental Optimizations (Ranked by Cross-Scene ROI)

1. **P0 — Fix `delta_unroll` Default (`6 -> 2..4`), Reclaim 2 VGPRs in `wavefront_shade_dielectric` (`145 -> <=144`), and Specialize `IS_SECONDARY_BOUNCE=1`**:
   - **Empirical Impact**: Lowering `--delta-unroll` `6 -> 2` speeds up `glass-of-water` by **`-36.2%` (`14.320 -> 9.137 ms`)**, `DragonDispersion` by **`-25.9%` (`11.176 -> 8.281 ms`)**, and `DragonAttenuation` by **`-23.3%` (`10.388 -> 7.970 ms`)**; `--delta-unroll 4` already captures `-14.3%`, `-9.7%`, and `-8.0%`. Disabling unrolling altogether via `--wavefront-sort none` hits **`6.832 ms` (`2.10x` faster)** on `glass-of-water`.
   - **ISA & Shader Mechanism**: [Config.hpp:124](file:///c:/Users/naoki/Development/Pathways/src/core/Config.hpp#L124) defaults `delta_unroll = 6` (contradicting `--help` claiming `4`). `wavefront_shade_dielectric` compiles to **`145` VGPRs on `gfx1100`** (`1` VGPR over the `144` cliff), capping Wave32 occupancy at **`9/16` (`56.25%`)** vs `10/16` (`62.5%`) on `gfx1201` (`143` VGPRs), lacks an `_sec` variant, and triggers a `baseLod = 0.0` bug at [wavefront_shade_dielectric.comp:206-233](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shade_dielectric.comp#L206-L233) sampling coarsest `1x1` mips for PSR guides.
   - **Portability**: High win on both RDNA 3 & RDNA 4 (saves dynamic BVH traversal loops and fixes the `145` VGPR cliff on `gfx1100`).

2. **P0 — Enable `--tail-megakernel` (`cutoffBounce = 2`) by Default (and Clean Up Unmasked Sort Config)**:
   - **Empirical Impact**: Improves **`9 / 10` tested scenes**, cutting total 4K frame time by **`-25.3%` on `cornell-caustic` (`7.132 -> 5.330 ms`)**, **`-17.4%` on `glass-of-water`**, **`-15.2%` on `classroom` & `veach-ajar`**, **`-12.8%` on `cornell-box`**, and **`-7.8%` on `living-room`** (replaces Bounce 1 `Intersect` + Bounces `2..3` of `1.27–3.83 ms` with a `0.45–2.31 ms` megakernel pass; cuts `cornell-caustic` queue traffic `2,279 -> 1,790 MB`). Only `infinity-mirror` (`+1.9%`, `+0.09 ms`, `<1 std_dev`) showed negligible difference because Bounce 2 had only `223K` rays (`2.7%`).
   - **Pipeline Mechanism**: [Config.hpp:122](file:///c:/Users/naoki/Development/Pathways/src/core/Config.hpp#L122) ships `enable_tail_megakernel = false`. `wavefront_tail_megakernel` holds **`62.5%` occupancy (`144` VGPRs), `38.0 KB` `.text`, and `0` global atomics**, eliminating queue round-trips and 7+ barriers in late bounces. Note [WavefrontPipeline.cpp:1286](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.cpp#L1286) passes `secondarySortMode` directly rather than clamping out `directional`.

3. **P0 — Fix `getEffectiveWavefrontSortMode`, Retire/Fix `Dual` Morton Sort (`& 0x0FFFu` LSB Bug), and Switch Default `--sec-sort` to `coherent-k8` (Retiring `directional`)**:
   - **Empirical Impact**: `auto` selected the strictly fastest explicit mode on only **`4 / 20` scenes** (`suboptimal_count = 16`): `archetype` beat `dual` on **`10 / 11` remaining non-dielectric sorted scenes** (`-0.1%` to `-1.2%`, paired $p = 0.0059$). Separately, `--sec-sort coherent-k8` (`K=8`) beat default `direct` (`K=4`) on **`9 / 12` scenes** (up to **`-3.4%` on `breakfast-room`**, **`-2.1%` on `PointInstancedMedCity`**, **`-1.7%` on `veach-ajar`**), whereas `directional` regressed **`12 / 12` (`+2.9%` to `+49.2%`)**.
   - **Code Mechanism**: [RayTracingOrchestrator.cpp:265-302](file:///c:/Users/naoki/Development/Pathways/src/rt/RayTracingOrchestrator.cpp#L265-L302) computes unweighted triangle areas (ignoring instances!) and routes to `Dual` above a tiny `0.1%` (`0.001f`) threshold (`17 / 20` scenes). Worse, [wavefront_intersect.comp:167](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_intersect.comp#L167) masks `morton3D_10bit(qPos) & 0x0FFFu` (12 LSBs aliasing every `15.625 cm` instead of MSBs `>> 18u`) across `30` `subgroupShuffle` instructions (`32` `ds_*` in ISA), scattering `materialIndices` and inducing up to `6x` cache-line amplification.

4. **P0 — Eliminate `208 B` `Material` Loads in `Classify` & `Emissive`, Deduplicate `traceShadowRayInline` + Decouple NRC Training Trace**:
   - **Shader & ISA Impact**: `wavefront_classify.comp` ([lines 292, 356, 382](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_classify.comp#L292-L382)), `wavefront_shade_emissive.comp:244`, and `wavefront_shade_complex.comp:314` (even `_sec`!) read the `208 B` `Material` struct instead of `64 B` `ShadeMaterial` (`3.25x` extra memory + `textureQueryLevels` double-`nonuniformEXT` waterfalls). Meanwhile, inlining `traceShadowRayInline` (`2` BVH blocks in [wavefront_shadow_inline.glsl:24-81](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shadow_inline.glsl#L24-L81)) and `nrcEmitSelfTrainingRecord` (`3` BVH blocks in [nrc_record_helper.glsl:38-194](file:///c:/Users/naoki/Development/Pathways/shaders/compute/nrc_record_helper.glsl#L38-L194)) inflates `wavefront_shade_diffuse*` / `complex*` to **`53.4–105.6 KB` `.text`** (**`1.7x–3.3x` the 32 KB RDNA L0 I-cache**) and **`5` (`7` w/ ReSTIR) BVH intersect instances**, pinning `diffuse` at `118` VGPRs (`75%`) and `complex` at `126–129` VGPRs (`62.5%`).
   - **Expected Gain**: Collapsing to `1–2` BVH sites + moving NRC tail trace out drops `.text` toward **`<32 KB`**, lifts `diffuse` toward **`<=96` VGPRs (`100%` occ)** and `complex_sec` (`129 -> <=120` VGPRs, `75%` occ), and cuts primary material fetch traffic **`3.25x`** on `PointInstancedMedCity`, `cyber-city`, and `mineshaft`.

5. **P1 — Fix VMA `HOST_VISIBLE` Static Scene Buffers (`SceneManager.cpp:448-515`) & Trim `2,784.75 MiB` Wavefront Queue Footprint**:
   - **Memory Mechanism**: [SceneManager.cpp:448-515](file:///c:/Users/naoki/Development/Pathways/src/scene/SceneManager.cpp#L448-L515) (`m_materialBuffer` `208 B`, `m_shadeMaterialBuffer` `64 B`, `m_materialArchetypeBuffer` `4 B`, `m_lightBuffer` `128 B`, `m_lightTreeBuffer` `64 B`) and [GpuTlasUpdatePipeline.cpp:72-80](file:///c:/Users/naoki/Development/Pathways/src/rt/GpuTlasUpdatePipeline.cpp#L72-L80) use `VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE` + `HOST_ACCESS_SEQUENTIAL_WRITE_BIT` without `ALLOW_TRANSFER_INSTEAD_BIT`, which forces VMA ([vk_mem_alloc.h:4295-4330](file:///c:/Users/naoki/Development/Pathways/third_party/vma/vk_mem_alloc.h#L4295-L4330)) into the `256 MB` BAR WC heap or System RAM via PCIe on non-ReBAR Windows hosts. Also missing `VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT` ([VulkanContext.cpp:1354](file:///c:/Users/naoki/Development/Pathways/src/vulkan/VulkanContext.cpp#L1354)).
   - **Queue Footprint & Traffic**: `WavefrontPipeline` allocates **`352 B/px` (`2,784.75 MiB` @ 4K)** double-buffered, including **`506.25 MiB` for `SecondaryIndexQueue`** (never read under default `direct`/`archetype`/`none` sorting) and **`506.25 MiB` of reclaimable fields in `32 B` `RayHit`** (`hitData1.z` barycentrics never read, `hitData1.xy` unused on B0/non-emitters, `hitData1.w[15:1]` unused). Compacting `RayHit` (`32 -> 16 B`) + gating `SecondaryIndexQueue` reclaims **`1,012.50 MiB` (`36.4%`)** of 4K queue VRAM and reduces every bounce's `Intersect` write + `Shade` read bandwidth.

6. **P1 — Fix Core Engine & Heuristic Bugs (`Engine.cpp:2361`, `16 GiB` Batching Off-by-One, `QueueCounters` 120 B Padding, `local_size=32` Atomics)**:
   - [Engine.cpp:2361](file:///c:/Users/naoki/Development/Pathways/src/core/Engine.cpp#L2361) uses `||` instead of `&&` (`enable_light_tree || !lightTreeNodes.empty()`), making `--no-light-tree` a complete no-op (`+0.4%..+1.8%` noise across 8 scenes).
   - [RayTracingOrchestrator.cpp:185](file:///c:/Users/naoki/Development/Pathways/src/rt/RayTracingOrchestrator.cpp#L185) tests `< (16ULL << 30)`, misclassifying exact `16 GiB` RDNA 4 GPUs into `8 GiB`-tier `2,000,000`-pixel multi-batching (`--batches 2/4` regressed `10/10` scenes up to `+28.8%` on RX 7900 XTX).
   - [wavefront_shared.glsl:16-44](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shared.glsl#L16-L44) leaves **`120 B` (`30` uints = `46.9%`)** dead in `256 B` `QueueCountersBuffer` while hot global atomics collide in the first 64 B cache line, compounded by **`259,200` `retiredWorkgroups` global atomics per bounce @ 4K** under `local_size_x = 32` (`64` threads cuts retirement atomics 2x).

7. **P1–P2 — Optimize Async Synchronization, Clean Up Unused DGC Preprocess/Resources, & Unblock `NRC` / `ReSTIR` Occupancy**:
   - Move `writeTimestamp` off `VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT` ([WavefrontPipeline.hpp:344-345](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.hpp#L344-L345)), strip `m_dgcStream` fills (`845`) & `dgcStream` shader writes when `ComputeExecutionSet` is unsupported, remove unconditional `centerDepthBuffer` 1×1 copy (`RayTracingOrchestrator.cpp:662-705`), move `mineshaft` TLAS updates from forced `MODE_BUILD_KHR` (`Engine.cpp:2587`) + full CPU Light Tree rebuilds (`SceneManager.cpp:523-598`) to periodic TLAS refit, and cut `nrc_encode_infer`/`nrc_train` LDS (`26–29 KB -> <=16 KB`) to double occupancy (`25% -> 50%`).

---

## 2. Testbed, Methodology & Measurement Hygiene

### 2.1 System & Compiler Environment

| Component | Exact Measured Specification |
|---|---|
| **GPU Hardware** | AMD Radeon RX 7900 XTX (`0x1002:0x744c`, `gfx1100` / RDNA 3), `96` CUs (`192` SIMD32s), `96 MB` Infinity Cache, `24,560 MB` VRAM, Wave32 native (`subgroupSize = 32`) |
| **Host System** | Windows, AMD Ryzen 7 7800X3D (8-Core), `31.7 GB` System RAM |
| **Vulkan Driver & SDK** | AMD proprietary driver `26.11.1 (LLPC)` (`2.0.416`), Vulkan API `1.4.359` |
| **Engine Build** | `build/windows-clang-release/bin/pathways.exe` (`v1.24.0`, Clang 20 Release, OpenUSD + FidelityFX SDK enabled) |
| **Ray Tracing / DGC Caps** | `VK_KHR_ray_query`, `VK_KHR_acceleration_structure`, `VK_KHR_ray_tracing_position_fetch`; DGC single-dispatch (`explicit_preprocess: true`), `VkIndirectExecutionSetEXT` `ComputeExecutionSet: UNSUPPORTED` -> GPU multi-dispatch indirect queues (`4` specialized material pipelines × `2` bounce variant sets) |
| **Offline ISA Toolchain** | AMD RDTS `utils/amdllpc.exe` (`LLPC 78.0`, `--auto-layout-desc --subgroup-size=32`), `llvm-readobj.exe`, `llvm-objdump.exe`, `shae.exe` (`-w 32 --arch-info`), targeting both `gfx1100` (RDNA 3) and `gfx1201` (RDNA 4) |

### 2.2 Benchmark Execution Protocol (254 GPU Runs, Zero Failures)

All GPU timings were collected strictly serialized (at most one `pathways.exe` process active) using `--headless --no-validation --no-accumulation --spp 1 --max-bounces 4` and built-in scene default cameras via [run_scene_analysis.py](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/run_scene_analysis.py):

- **Sweep 1 (4K Native Baseline, `3840x2160`, 20 runs)**: Heavy scenes (`>15 ms`: `cyber-city`, `mineshaft`, `breakfast-room`, `modern-hall`, `BuickRiviera`) ran `12` warmup + `30` measured frames; the remaining `15` scenes ran `20` warmup + `50` measured frames.
- **Sweep 2 (1080p Scaling Baseline, `1920x1080`, 20 runs)**: All `20` scenes ran `20` warmup + `40` measured frames to perform exact two-point linear decomposition ($N_{4\text{K}} = 4 \cdot N_{1080\text{p}}$):
  $$T_{\text{pixel, 1080p}} = \frac{T_{4\text{K}} - T_{1080\text{p}}}{3}, \qquad T_{\text{pixel, 4K}} = 4 \cdot T_{\text{pixel, 1080p}}, \qquad T_{\text{fixed}} = \frac{4 \cdot T_{1080\text{p}} - T_{4\text{K}}}{3}$$
- **Sweep 3 (4K Architectural Ablations, Groups A–G, 214 runs)**: All ablations ran at `3840x2160`, `1 SPP`, `max-bounces 4`, `15` warmup + `35` measured frames, strictly compared against the matched `Group A auto` (`15/35`) run rather than Sweep 1 (`20/50`) to eliminate warmup/frame-window bias.

### 2.3 Measurement Stability, CV Hygiene & Accounting Discrepancies

- **Zero Errors**: All `254 / 254` runs exited cleanly with `Validation Errors: 0`.
- **Coefficient of Variation (`CV = std_dev / avg_frame_time_ms`)**:
  - **Sweep 1 (4K, 20/20)**: `CV` spans `0.0122` (`glass-of-water`) to `0.0349` (`mineshaft`, animated coaster), well below the `0.05` re-run threshold.
  - **Sweep 2 (1080p, 19/20 <= `0.0426`)**: Only **`DamagedHelmet` @ 1080p** (`0.542 ± 0.028 ms`, **`CV = 0.0517`**) exceeds `0.05`, driven by sub-millisecond (`28 µs`) GPU clock/scheduler quantization at `1,844.7 FPS` rather than workload instability ([verification_audit.md:31-32](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/verification_audit.md#L31-L32)). Next highest 1080p CV is `Kitchen_set` (`0.0426`, `1.081 ± 0.046 ms`).
  - **Sweep 3 (4K Ablations, 214/214)**: `CV` spans `0.0043` (`A_none__BuickRiviera`) to `0.0361` (`A_none__bmw-m6`); `0 / 214` exceed `0.05`.
- **Stage Timing Decomposition Verified**: Across all 40 Sweep 1+2 runs, `primary_gpu_time_ms + tonemap_and_merge_time_ms` matches `avg_frame_time_ms` within `<= 0.005 ms`, and `primary_gpu_time_ms - (classify_ms + sum(total_bounce_ms))` stays tightly bounded at **`+0.064 to +0.077 ms`** on static scenes (Wavefront prologue + `RayTracingOrchestrator` clear/`centerDepthBuffer` copy + `m_indirectReadback` readback), **`+0.199 ms`** on `cyber-city` (`VideoBillboardManager` frame upload), and **`+0.308 ms`** on `mineshaft` (GPU TLAS rebuild + dynamic light upload).
- **C++ `estimated_vram_traffic_mb` Accounting Caveat**: [WavefrontPipeline.cpp:1514-1525](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.cpp#L1514-L1525) estimates queue traffic using slightly miscalibrated per-ray constants (`Classify = 64 B` vs actual `68 B`, `Intersect = 32 B` vs actual `52 B` missing `16 B` `inGeom` + `4 B` `matIdx`, `Shade` read missing `4 B` `matIdx`, `Shade` next-ray write `48 B` vs actual `32 B`), and excludes BVH, triangle, texture, material, and Light Tree fetches. Accordingly, Section 3 reports the exact C++ telemetry figure alongside effective queue rate (`reported_MB / primary_gpu_ms`) with that explicit qualification, while Section 5 quantifies true physical byte layouts.
- **Offline ISA Calibration Note**: `scripts/extract_shader_stats.py:179-181` uses a legacy `512`-VGPR / `8`-granule formula. Direct verification via RDTS `shae.exe --arch-info -w 32` confirms both `gfx1100` and `gfx1201` allocate **`1,536` physical Wave32 VGPRs/SIMD32 in `24`-VGPR increments** (`ceil(vgpr / 24) * 24`), creating Wave32 occupancy cliffs at **`96 (100%), 120 (75%), 144 (62.5%), 168 (56.25%), 192 (50%), 216 (43.75%), and 240 (37.5%)` VGPRs**, plus **`64 KiB` LDS/CU**. Subgroup-less compute shaders (`accum_tonemap_fused`, `tonemap_aces`, `caustic_filter`, etc.) were compiled with `options.waveSize = 32` / `subgroupSize = 32` `.pipe` overrides matching Pathways' `requiredSubgroupSize = 32` Vulkan pipelines.

---

## 3. Master 20-Scene Performance Matrix (4K Native vs. 1080p Scaling)

### 3.1 Geometry, Instancing, Acceleration Structures & Scene Topology (4K Sweep 1)

| # | Scene | Cat | Base / Instanced Tris | Inst (TLAS) | Mats / Lights / Tex | BLAS (ms / KB / Comp%) | TLAS (ms / KB / GPUUpd) | NonOp / αMask | Env `p` |
|---:|---|---|---:|---:|---:|---|---|---|---:|
| 1 | `cornell-box` | Proc | 2,232 | 1 (2) | 7 / 1 / 0 | 1.244 / 129.3 (-63.6%) | 0.112 / 0.9 (0) | YES / NO | 0.00 |
| 2 | `many-lights` | Proc | 2,358 | 1 (2) | 7 / 64 / 0 | 1.493 / 136.0 (-63.8%) | 0.110 / 0.9 (0) | YES / NO | 0.00 |
| 3 | `cyber-city` | Proc | 30,044 / 13,581,578 | 5,596 | 48 / 640,835 / 1 | 1.474 / 1650.6 (-65.5%) | 0.749 / 1271.8 (0) | YES / NO | 0.00 |
| 4 | `infinity-mirror` | Proc | 22,142 | 6 | 18 / 15 / 0 | 1.823 / 1302.5 (-63.0%) | 0.168 / 1.8 (70) | YES / NO | 0.00 |
| 5 | `mineshaft` | Proc | 36,172 / 416,692 | 526 | 16 / 32 / 0 | 1.547 / 2222.7 (-61.4%) | 0.497 / 119.8 (42) | YES / NO | 0.00 |
| 6 | `DamagedHelmet` | Show | 15,452 | 1 | 1 / 1 / 5 | 1.551 / 895.2 (-63.5%) | 0.092 / 0.8 (0) | NO / NO | 0.25 |
| 7 | `DragonAttenuation` | Show | 134,995 | 1 (2) | 3 / 1 / 2 | 3.507 / 8052.0 (-62.5%) | 0.109 / 0.9 (0) | YES / NO | 0.25 |
| 8 | `DragonDispersion` | Show | 134,995 | 1 (2) | 2 / 1 / 2 | 3.411 / 8054.9 (-62.4%) | 0.112 / 0.9 (0) | YES / NO | 0.25 |
| 9 | `bmw-m6` | Res | 792,746 | 1 (2) | 26 / 1 / 0 | 10.445 / 54933.0 (-56.4%) | 0.081 / 0.9 (0) | YES / NO | 0.25 |
| 10 | `breakfast-room` | Res | 269,538 | 1 | 15 / 2 / 3 | 3.446 / 17588.1 (-58.9%) | 0.094 / 0.8 (0) | NO / NO | 0.00 |
| 11 | `classroom` | Res | 103,832 | 1 | 44 / 1 / 3 | 2.476 / 6811.1 (-58.7%) | 0.095 / 0.8 (0) | NO / NO | 0.00 |
| 12 | `cornell-caustic` | Res | 103,487 | 1 (2) | 9 / 1 / 0 | 3.030 / 6529.7 (-60.3%) | 0.156 / 0.9 (0) | YES / NO | 0.00 |
| 13 | `glass-of-water` | Res | 406,387 | 1 (2) | 7 / 1 / 0 | 8.131 / 25123.3 (-61.1%) | 0.095 / 0.9 (0) | YES / NO | 0.00 |
| 14 | `living-room` | Res | 143,163 | 1 (3) | 19 / 3 / 5 | 3.609 / 9684.0 (-57.4%) | 0.132 / 1.1 (0) | **YES / YES** | 0.00 |
| 15 | `modern-hall` | Res | 31,087 | 1 (2) | 21 / 168 / 3 | 1.868 / 2080.1 (-57.9%) | 0.111 / 0.9 (0) | YES / NO | 0.00 |
| 16 | `veach-ajar` | Res | 382,690 | 1 (2) | 13 / 1 / 4 | 5.612 / 28075.6 (-53.8%) | 0.096 / 0.9 (0) | YES / NO | 0.00 |
| 17 | `BuickRiviera` | USD | 861,243 | 1 (2) | 43 / 132 / 20 | 11.659 / 57262.0 (-58.1%) | 0.086 / 0.9 (0) | YES / NO | 0.50 |
| 18 | `coffee-maker` | USD | 235,271 | 1 (2) | 7 / 3 / 0 | 4.988 / 14094.2 (-62.3%) | 0.120 / 0.9 (0) | YES / NO | 0.00 |
| 19 | `Kitchen_set` | USD | 536,949 | 1 | 135 / 1 / 0 | 4.139 / 35994.3 (-57.8%) | 0.088 / 0.8 (0) | NO / NO | 0.25 |
| 20 | `PointInstancedMedCity` | USD | 27,456 / 49,177,624 | 40,001 | 14 / 1 / 0 | 1.615 / 1528.6 (-65.0%) | 0.996 / 9089.1 (0) | YES / NO | 0.00 |

### 3.2 Native 4K vs. 1080p Performance, Scaling, Queue Bandwidth & Bottleneck Classification

*(Queue memory footprint = **`2,784.75 MB` @ 4K** and **`696.47 MB` @ 1080p** for every scene with `batches=1`. `Est. Q Rate` = C++-reported `estimated_vram_traffic_mb / primary_gpu_time_ms`.)*

| Scene | 4K Avg (ms) | CV | 4K FPS / GR/s | 1080p Avg ± SD (ms) | 4K/1080 Ratio | `T_fixed` / `T_pix4K` (ms) | VRAM (MB) | Est Q Traffic / Rate | Dominant Bottleneck Tag |
|---|---:|---:|---|---:|---:|---|---:|---:|---|
| `cornell-box` | **6.028** | 0.023 | 165.9 / 5.50 | 1.545 ± 0.024 | 3.90x | +0.051 / 5.977 | 3726.8 | 2433.7 MB (413.1 GB/s) | Balanced Diffuse Shade + Queue BW |
| `many-lights` | **9.466** | 0.025 | 105.6 / 3.51 | 2.394 ± 0.041 | 3.95x | +0.037 / 9.429 | 3726.8 | 2443.1 MB (261.9 GB/s) | Multi-Light Shade ALU (64 Lights) |
| `cyber-city` | **23.338** | 0.015 | 42.8 / 1.42 | 6.014 ± 0.110 | 3.88x | +0.239 / 23.099 | 3982.8 | 2566.1 MB (110.6 GB/s) | Shade + 640K Light Tree + 5.6K TLAS |
| `infinity-mirror` | **4.695** | 0.022 | 213.0 / 7.07 | 1.267 ± 0.037 | 3.71x | +0.124 / 4.571 | 3726.8 | 1353.6 MB (297.0 GB/s) | Multi-Light Complex/Diffuse B0-B1 Shade |
| `mineshaft` | **17.152** | 0.035 | 58.3 / 1.93 | 4.714 ± 0.131 | 3.64x | +0.568 / 16.584 | 3726.8 | 2683.4 MB (157.6 GB/s) | Shade + Coaster BVH/TLAS + Dynamic Lights |
| `DamagedHelmet` | **1.966** | 0.015 | 508.6 / 16.87 | 0.542 ± 0.028* | 3.63x | +0.067 / 1.899 | 3982.8 | 864.2 MB (473.3 GB/s) | Low-Hit Fixed/Classify/Tonemap + Queue BW |
| `DragonAttenuation` | **10.523** | 0.024 | 95.0 / 3.15 | 2.946 ± 0.049 | 3.57x | +0.420 / 10.103 | 3726.8 | 1676.2 MB (161.4 GB/s) | Dielectric `delta_unroll=6` + 56.25% Occ |
| `DragonDispersion` | **11.245** | 0.023 | 88.9 / 2.95 | 3.077 ± 0.045 | 3.65x | +0.354 / 10.891 | 3726.8 | 1667.4 MB (150.1 GB/s) | Dielectric Dispersion + `delta_unroll=6` |
| `bmw-m6` | **10.022** | 0.022 | 99.8 / 3.31 | 2.920 ± 0.046 | 3.43x | +0.553 / 9.469 | 3982.8 | 2347.8 MB (237.5 GB/s) | Complex Coat Shade (62.5% Occ) + 793K BVH |
| `breakfast-room` | **15.351** | 0.013 | 65.1 / 2.16 | 4.582 ± 0.078 | 3.35x | +0.992 / 14.359 | 3726.8 | 2464.5 MB (162.0 GB/s) | BVH Intersect + Shadow + `T_fixed` Latency |
| `classroom` | **10.708** | 0.022 | 93.4 / 3.10 | 2.705 ± 0.049 | 3.96x | +0.037 / 10.671 | 3726.8 | 3540.3 MB (334.7 GB/s) | High-Survival Queue BW (3.54 GB) + Traversal |
| `cornell-caustic` | **7.240** | 0.027 | 138.1 / 4.58 | 1.891 ± 0.035 | 3.83x | +0.108 / 7.132 | 3726.8 | 2278.8 MB (321.1 GB/s) | Dielectric Unroll + Late-Bounce Queue BW |
| `glass-of-water` | **14.304** | 0.012 | 69.9 / 2.32 | 4.299 ± 0.080 | 3.33x | +0.964 / 13.340 | 3726.8 | 2712.4 MB (191.5 GB/s) | Extreme `delta_unroll=6` Refractive Chain |
| `living-room` | **11.602** | 0.020 | 86.2 / 2.86 | 3.247 ± 0.048 | 3.57x | +0.462 / 11.140 | 3726.8 | 2768.5 MB (241.5 GB/s) | Complex+Diffuse Shade & Alpha-Mask AnyHit |
| `modern-hall` | **15.730** | 0.025 | 63.6 / 2.11 | 3.876 ± 0.069 | **4.06x** | -0.075 / 15.805 | 3726.8 | 3438.3 MB (220.4 GB/s) | 168-Light Shade + 41.6% B3 Survival (4K Spill) |
| `veach-ajar` | **7.875** | 0.024 | 127.0 / 4.21 | 2.398 ± 0.037 | 3.28x | +0.572 / 7.303 | 3982.8 | 3204.1 MB (414.4 GB/s) | High-Survival Traversal & Queue BW (3.20 GB) |
| `BuickRiviera` | **20.686** | 0.015 | 48.3 / 1.60 | 5.439 ± 0.106 | 3.80x | +0.357 / 20.329 | 4921.9 | 2567.2 MB (124.9 GB/s) | 132-Light/Env/20-Tex Shade + 861K BVH |
| `coffee-maker` | **9.167** | 0.023 | 109.1 / 3.62 | 2.580 ± 0.049 | 3.55x | +0.384 / 8.783 | 3726.8 | 2325.6 MB (257.7 GB/s) | B0 Complex Metallic Shade (62.5% Occ) |
| `Kitchen_set` | **3.236** | 0.016 | 309.0 / 10.25 | 1.081 ± 0.046 | **2.99x** | +0.363 / 2.873 | 3982.8 | 1078.0 MB (348.2 GB/s) | 75% Sky Miss + Fixed Overhead (33.5% @1080p) |
| `PointInstancedMedCity` | **9.469** | 0.031 | 105.6 / 3.50 | 2.491 ± 0.048 | 3.80x | +0.165 / 9.304 | 3726.8 | 2071.3 MB (221.9 GB/s) | 40,001-Inst TLAS Intersect & Classify (60%) |

*\*`DamagedHelmet` @ 1080p is the sole run with `CV > 0.05` (`0.0517`, `28 µs` sub-ms jitter at `1,844.7 FPS`).*

---

## 4. Per-Scene Deep-Dive Profiles (All 20 Canonical Scenes)

In every profile below:
- **Stage & Bounce Waterfalls** report the multi-frame averaged stage & `Gap1/Gap2/Gap3` timings (`wavefront_averaged_stages` + `stdout.gaps_per_bounce`, summing to `primary_gpu_time_ms - host_overhead`).
- **Ray Funnels & Material Archetypes** report the last sampled frame's `wavefront_profiler_breakdown` counters (`Active`, `Diff / Diel / Cmplx / Emiss`, `Shadow Rays`, `Next Rays`). (`Conductor` and `Passthrough` counts were `0` across all 20 default runs.)
- **Sweep 3 Ablation Deltas** are measured against each scene's matched `Group A auto` (`3840x2160`, `15` warmup / `35` measured) baseline.

---

### 4.1 `cornell-box` (Procedural | 4K `6.028 ms`, 165.9 FPS | 1080p `1.545 ms`, Ratio `3.90x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.603 ms`, `Tonemap = 0.137 ms`, `Gaps = 0.030 / 0.016 / 0.012 = 0.058 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.012 | 1.450 | 0.000 | 0.000 | 0.012 | 0.808 | **2.283** | 2,433.7 MB total |
| **B1** | 0.007 | 1.213 | 0.010 | 0.207 | 0.000 | 0.396 | **1.833** | `413.1 GB/s` |
| **B2** | 0.008 | 0.631 | 0.005 | 0.147 | 0.000 | 0.184 | **0.975** | |
| **B3** | 0.003 | 0.124 | 0.001 | 0.000 | 0.000 | 0.000 | **0.128** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 6,625,755 | 6,181,892 | 284,733 | 0 | 159,130 | 0 | 6,168,817 (**74.4%**) |
| **B1** | 6,168,740 | 5,810,781 | 244,149 | 0 | 113,810 | 1,718,739 | 2,609,934 (**31.5%**) |
| **B2** | 2,607,728 | 2,413,479 | 151,944 | 0 | 42,305 | 879,871 | 1,159,924 (**14.0%**) |
| **B3** | 1,159,310 | 1,055,149 | 85,533 | 0 | 18,628 | 0 | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 5.945 ms`, picked `dual`)**:
  - **Sort (`A`)**: `archetype = 5.869 ms` (**-1.3%**, fastest), `none = 5.908 ms` (-0.6%), `dual = 5.937 ms` (-0.1%).
  - **Sec-Sort (`B`)**: `coherent-k8 = 5.888 ms` (**-1.0%**), `none = 6.092 ms` (+2.5%), `directional = 8.745 ms` (**+47.1%**).
  - **Batches (`C`)**: `1 = 5.994` (+0.8%), `2 = 6.043` (+1.6%), `4 = 6.090` (+2.4%).
  - **Inline Shadows (`D`)**: `--no-inline-shadows = 6.310 ms` (**+6.1%** slower).
  - **Tail / Cutoff (`E`)**: `--tail-megakernel = 5.181 ms` (**-12.8%**, tail=`0.658 ms`), `--no-early-bounce-cutoff = 6.007 ms` (+1.0%).
  - **Formats / Features (`G`)**: `rg11 = 6.004 ms` (+1.0%, TM `0.122`), `fp32 = 6.543 ms` (**+10.1%**, TM `0.303`, `+128 MB`), `--restir = 8.261 ms` (**+39.0%**).
- **(d) Root-Cause Diagnosis**: With only `2,232` triangles and `1` light, BVH traversal fits comfortably in cache; performance is dictated by **wavefront queue bandwidth** (`413.1 GB/s` C++-reported rate — second highest in the suite) and **Bounce 1–3 queue round-trips** (`1.50 ms` of Shade/Shadow/Intersect on `<=31.5%` active rays). Switching to `--tail-megakernel` immediately cuts `0.764 ms` (`-12.8%`), while removing inline shadows (`+6.1%`) or splitting into `8` directional bins (`+47.1%`) highlights how sensitive simple-geometry scenes are to queue traffic and dispatch overhead.

---

### 4.2 `many-lights` (Procedural | 4K `9.466 ms`, 105.6 FPS | 1080p `2.394 ms`, Ratio `3.95x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.610 ms`, `Tonemap = 0.136 ms`, `Gaps = 0.032 / 0.026 / 0.029 = 0.087 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.015 | 3.759 | 0.000 | 0.000 | 0.029 | 0.838 | **4.640** | 2,443.1 MB total |
| **B1** | 0.008 | 1.934 | 0.017 | 0.207 | 0.000 | 0.410 | **2.577** | `261.9 GB/s` |
| **B2** | 0.005 | 0.847 | 0.007 | 0.140 | 0.000 | 0.190 | **1.190** | |
| **B3** | 0.004 | 0.239 | 0.002 | 0.000 | 0.000 | 0.000 | **0.246** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 6,625,755 | 6,258,355 | 284,733 | 0 | 82,667 | 0 | 6,242,702 (**75.3%**) |
| **B1** | 6,242,717 | 5,866,252 | 246,377 | 0 | 130,088 | 1,671,516 | 2,637,663 (**31.8%**) |
| **B2** | 2,636,550 | 2,429,849 | 154,236 | 0 | 52,465 | 830,402 | 1,168,942 (**14.1%**) |
| **B3** | 1,167,825 | 1,060,231 | 87,108 | 0 | 20,486 | 0 | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 9.376 ms`, picked `dual`)**:
  - **Sort (`A`)**: `archetype = 9.297 ms` (**-0.8%**, fastest), `dual = 9.377 ms` (+0.0%), `none = 13.235 ms` (**+41.2%** slower).
  - **Inline Shadows (`D`)**: `--no-inline-shadows = 9.521 ms` (**+1.6%** slower).
  - **Light Tree (`F`)**: `--no-light-tree = 9.490 ms` (+1.2%, invalid due to `Engine.cpp:2361` `||` bug).
  - **Features (`G`)**: `--restir = 20.013 ms` (**+113.5%**, `+10.637 ms`).
- **(d) Root-Cause Diagnosis**: Identical geometry and nearly identical ray counts to `cornell-box` (`+126` tris, `2,443` vs `2,434 MB` queue traffic), yet frame time jumps by **`+3.438 ms` (`+57%`) exclusively in `Shade` (`3.418 -> 6.779 ms`)** while `Intersect` (`1.438 ms`) and `Shadow` (`0.347 ms`) stay flat. With `64` lights, `wavefront_shade_diffuse` (`81.4 KB` `.text`, `75%` occupancy, `5` BVH sites) executes Light Tree importance traversal (`32`-step loop × `128 B` child reads in `wavefront_common.glsl:406-486`) plus `M=8` RIS candidate loops under `--restir`, doubling Bounce 0+1 shade (`3.72 -> 8.32 ms` and `1.91 -> 5.83 ms`, **`+10.637 ms` total**).

---

### 4.3 `cyber-city` (Procedural | 4K `23.338 ms`, 42.8 FPS | 1080p `6.014 ms`, Ratio `3.88x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 1.798 ms`, `Tonemap = 0.137 ms`, `Gaps = 0.046 / 0.084 / 0.059 = 0.189 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.016 | 9.248 | 0.000 | 0.000 | 0.059 | 2.350 | **11.673** | 2,566.1 MB total |
| **B1** | 0.009 | 5.197 | 0.054 | 0.394 | 0.000 | 1.041 | **6.695** | `110.6 GB/s` |
| **B2** | 0.006 | 1.905 | 0.017 | 0.135 | 0.000 | 0.349 | **2.412** | |
| **B3** | 0.015 | 0.387 | 0.013 | 0.011 | 0.000 | 0.000 | **0.425** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 8,290,297 | 4,930,252 | 1,203,454 | 833,957 | 1,322,634 | 0 | 6,579,711 (**79.3%**) |
| **B1** | 6,579,875 | 4,162,729 | 783,380 | 945,937 | 687,829 | 1,905,238 | 2,756,263 (**33.2%**) |
| **B2** | 2,757,790 | 1,617,444 | 500,235 | 394,110 | 246,001 | 553,841 | 884,360 (**10.7%**) |
| **B3** | 884,259 | 535,701 | 162,520 | 123,098 | 62,940 | 5 (avg 12) | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 23.456 ms`, picked `dual`)**:
  - **Sort (`A`)**: `archetype = 23.309 ms` (**-0.6%**, fastest), `dual = 23.465 ms` (+0.0%), `none = 32.021 ms` (**+36.5%** slower).
  - **Sec-Sort (`B`)**: `direct = 23.456 ms` (fastest), `coherent-k8 = 23.493 ms` (+0.2%), `none = 23.703 ms` (+1.1%), `directional = 24.130 ms` (+2.9%).
  - **Batches (`C`)**: `2 = 23.461` (+0.0%), `1 = 23.595` (+0.6%), `4 = 23.678` (+0.9%).
  - **Inline Shadows (`D`)**: `--no-inline-shadows = 23.341 ms` (**-0.5%**, Bounce-0 `shade+shadow` `-0.245 ms`).
  - **AS / Light Tree (`F`)**: `--macro-blas = 25.458 ms` (**+8.5%**, TLAS inst `5,596 -> 4,917`, BLAS `1,650.6 -> 18,214.9 KB`), `--no-light-tree = 23.718 ms` (+1.1%, null due to `Engine.cpp:2361`).
- **(d) Root-Cause Diagnosis**: Slowest scene in the benchmark (`23.338 ms`), overwhelmingly **Shade ALU + Light Tree bound** (`16.737 ms` = `71.7%` of frame time) across **`640,835` lights (`640,386` instanced emissive quads)** and all four active material archetypes (`4.93M` Diffuse + `1.20M` Dielectric + `0.83M` Complex + `1.32M` Emissive at B0). `1.32M` B0 + `0.69M` B1 emissive hits load full `208 B` `Material` in `wavefront_shade_emissive` (`line 244`) and replay `lightTreeSelectPdf` (`mis_common.glsl:228-248`) across the deep `640k`-light tree. Notice two critical insights: (1) detached Bounce-0 shadows beat inline by `0.245 ms` here because `wavefront_shadow` (`54` VGPRs, `100%` occ) avoids spilling already-saturated shade pressure, and (2) `--macro-blas` collapsed only `-12.1%` of instances (`5,596 -> 4,917`) while inflating BLAS `11.0x` and regressing Classify (`+0.36 ms`), Shade (`+0.63 ms`), and Intersect (`+0.33 ms`) due to coarse 2D `(X,Z)` grid vertical overlap on skyscrapers.

---

### 4.4 `infinity-mirror` (Procedural | 4K `4.695 ms`, 213.0 FPS | 1080p `1.267 ms`, Ratio `3.71x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.670 ms`, `Tonemap = 0.138 ms`, `Gaps = 0.025 / 0.008 / 0.013 = 0.046 ms`)**:
  *(Early bounce cutoff terminated Bounce 3 dynamically on `<=0.2%` survival.)*

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.013 | 2.033 | 0.000 | 0.000 | 0.013 | 0.528 | **2.587** | 1,353.6 MB total |
| **B1** | 0.008 | 0.986 | 0.006 | 0.071 | 0.000 | 0.031 | **1.102** | `297.0 GB/s` |
| **B2** | 0.004 | 0.094 | 0.002 | 0.020 | 0.000 | 0.000 | **0.121** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, animated `gpu_tlas_updates = 70`)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 4,663,643 | 3,867,247 | 5,239 | 164,131 | 627,026 | 0 | 3,707,120 (**44.7%**) |
| **B1** | 3,706,672 | 3,431,208 | 1,809 | 67,172 | 206,483 | 323,467 | 224,229 (**2.7%**) |
| **B2** | 223,130 | 201,971 | 198 | 4,870 | 16,091 | 30,247 | 19,162 (**0.2%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 4.733 ms`, picked `dual`)**:
  - **Sort (`A`)**: `archetype = 4.718 ms` (**-0.3%**, fastest), `dual = 4.742 ms` (+0.2%), `none = 5.643 ms` (**+19.2%** slower).
  - **Sec-Sort (`B`)**: `direct = 4.733 ms` (fastest), `coherent-k8 = 4.830 ms` (+2.0%), `none = 4.878 ms` (+3.1%), `directional = 5.420 ms` (+14.5%).
  - **Tail / Cutoff / Unroll (`E`)**: `--tail-megakernel = 4.823 ms` (+1.9%, tail=`0.114 ms`), `--no-early-bounce-cutoff = 4.905 ms` (**+3.6%** — largest cutoff effect in Group E), `--delta-unroll 4 = 4.868 ms` (+2.9%), `2 = 4.830 ms` (+2.0%).
- **(d) Root-Cause Diagnosis**: Only `56.2%` of primary rays hit geometry (`4.66M` active) and specular/emissive chains quickly exit or hit `15` area lights (`627K` B0 emissive), leaving only `2.7%` rays by B2 — making this the textbook showcase where **`--early-bounce-cutoff` saves `+3.6%` (`0.172 ms`)** by skipping Bounce 3 entirely and where `--tail-megakernel` provides no further gain (`0.121 -> 0.114 ms`). The `15` lights + `627K` emissive hits make sorting mandatory (`none` `+19.2%`), while `archetype` edges `dual`.

---

### 4.5 `mineshaft` (Procedural | 4K `17.152 ms`, 58.3 FPS | 1080p `4.714 ms`, Ratio `3.64x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 1.632 ms`, `Tonemap = 0.128 ms`, `Gaps = 0.057 / 0.026 / 0.051 = 0.134 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.015 | 6.202 | 0.000 | 0.000 | 0.051 | 3.015 | **9.284** | 2,683.4 MB total |
| **B1** | 0.021 | 2.700 | 0.024 | 1.116 | 0.000 | 0.827 | **4.689** | `157.6 GB/s` |
| **B2** | 0.019 | 0.579 | 0.001 | 0.309 | 0.000 | 0.097 | **1.006** | |
| **B3** | 0.002 | 0.049 | 0.001 | 0.053 | 0.000 | 0.000 | **0.105** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `gpu_tlas_updates = 42` @ 4K)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 8,294,400 | 8,205,136 | 89,264 | 0 | 0 | 0 | 7,980,707 (**96.2%**) |
| **B1** | 7,974,319 | 7,817,735 | 132,945 | 0 | 23,639 | 3,253,135 | 2,098,949 (**25.3%**) |
| **B2** | 2,108,271 | 2,030,217 | 44,940 | 0 | 33,114 | 779,873 (avg 743,873) | 246,761 (**3.0%**) |
| **B3** | 250,555 | 238,012 | 9,001 | 0 | 3,542 | 92,859 (avg 88,701) | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 17.411 ms`, picked `archetype`)**:
  - **Sort (`A`)**: `archetype = 17.408 ms` (**fastest**, tied `auto`), `dual = 17.443 ms` (+0.2%), `none = 19.991 ms` (**+14.8%** slower).
  - **AS / Light Tree (`F`)**: `--macro-blas = 17.689 ms` (+1.6%, skipped at `!animatedInstances.empty()` / `isMineshaftCoaster`), `--no-light-tree = 17.730 ms` (+1.8%, null due to `Engine.cpp:2361`).
- **(d) Root-Cause Diagnosis**: Animated coaster scene with highest 4K Sweep 1 jitter (`CV = 0.0349`) and **`+0.308 ms` non-wavefront GPU overhead** (`primary_gpu_time_ms - sum(stages)`) caused by forced **`MODE_BUILD_KHR` TLAS rebuilds every frame** (`Engine.cpp:2587`, `0.497 ms` initial) plus **per-frame CPU alias table + Light Tree rebuilds and `TRANSFER -> COMPUTE` pipeline barrier** (`ProceduralScene.cpp:4429-4445`, `SceneManager.cpp:523-598`). `99%+` of rays are `Diffuse` with `32` lights (`Shade = 9.530 ms`, `55.6%`) and complex multi-instanced tunnel traversal (`526` instances, `32` BLASes, `Intersect = 3.939 ms`, `Classify = 1.632 ms`), while `Shadow` hits `1.478 ms` (`3.25M` B1 shadow rays). `auto` correctly selected `archetype`, making this one of only `4/20` optimal `auto` picks.

---

### 4.6 `DamagedHelmet` (Showcase | 4K `1.966 ms`, 508.6 FPS | 1080p `0.542 ms`, Ratio `3.63x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.493 ms`, `Tonemap = 0.141 ms`, `Gaps = 0.012 / 0.004 / 0.006 = 0.022 ms`)**:
  *(Early bounce cutoff terminated Bounce 3 after `0.0%` survival.)*

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.007 | 0.631 | 0.000 | 0.000 | 0.006 | 0.274 | **0.918** | 864.2 MB total |
| **B1** | 0.002 | 0.233 | 0.003 | 0.053 | 0.000 | 0.011 | **0.303** | `473.3 GB/s` |
| **B2** | 0.003 | 0.016 | 0.001 | 0.022 | 0.000 | 0.000 | **0.043** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `1` material, `5` textures, `envSelectProb = 0.25`)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 1,746,205 | 1,746,205 | 0 | 0 | 0 | 0 | 1,701,712 (**20.5%**) |
| **B1** | 1,701,910 | 1,701,910 | 0 | 0 | 0 | 159,373 (avg 159,155) | 52,431 (**0.6%**) |
| **B2** | 52,257 | 52,257 | 0 | 0 | 0 | 9,792 (avg 9,960) | 2,398 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 1.968 ms`, picked `archetype`)**:
  - **Sort (`A`)**: `auto (archetype) = 1.968 ms` (**tied fastest** within 7 µs), `archetype = 1.975 ms` (+0.4%), `dual = 1.978 ms` (+0.5%), `none = 2.103 ms` (+6.9%).
- **(d) Root-Cause Diagnosis**: Fastest scene (`1.966 ms`, **`16.874 GRays/s`**) because **`79.5%` of primary rays miss the single `15.5K`-tri mesh** into the HDRI environment and `99.4%` exit by B1 (`52K` rays at B2). Consequently, **`Classify` (`0.493 ms` = `25.1%`) + `Tonemap` (`0.141 ms` = `7.2%`) + fixed driver/dispatch overheads consume a third of 4K frame time**, and `Est. Q Rate` peaks at **`473.3 GB/s`**. At 1080p (`0.542 ms`), `28 µs` sub-ms jitter yields `CV = 0.0517`. Even on `1` material, `archetype` beats `none` (`1.975` vs `2.103 ms`) because `wavefront_shade_diffuse` + compact queue packing keeps waves dense, whereas `wavefront_shade` runs `128` VGPRs (`62.5%` vs `75%` occupancy).

---

### 4.7 `DragonAttenuation` (Showcase | 4K `10.523 ms`, 95.0 FPS | 1080p `2.946 ms`, Ratio `3.57x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.884 ms`, `Tonemap = 0.135 ms`, `Gaps = 0.026 / 0.026 / 0.032 = 0.084 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.007 | 3.711 | 0.000 | 0.000 | 0.032 | 1.357 | **5.108** | 1,676.2 MB total |
| **B1** | 0.012 | 2.666 | 0.018 | 0.346 | 0.000 | 0.398 | **3.440** | `161.4 GB/s` |
| **B2** | 0.006 | 0.667 | 0.007 | 0.119 | 0.000 | 0.024 | **0.824** | |
| **B3** | 0.001 | 0.022 | 0.001 | 0.039 | 0.000 | 0.000 | **0.064** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `134,995` tris, `91,216` non-opaque)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 4,891,481 | 3,692,369 | 1,199,112 | 0 | 0 | 0 | 4,695,607 (**56.6%**) |
| **B1** | 4,694,883 | 4,043,852 | 651,031 | 0 | 0 | 1,280,402 | 853,771 (**10.3%**) |
| **B2** | 854,228 | 612,679 | 241,549 | 0 | 0 | 280,854 | 91,654 (**1.1%**) |
| **B3** | 91,340 | 65,120 | 26,220 | 0 | 0 | 24,950 (avg 24,975) | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 10.388 ms`, picked `dual`)**:
  - **Sort (`A`)**: **`none = 6.945 ms` (-33.1%, fastest; `auto` is `+49.6%` slower!)**, `archetype = 10.449 ms` (+0.6%), `dual = 10.459 ms` (+0.7%).
  - **Sec-Sort (`B`)**: `coherent-k8 = 10.332 ms` (-0.5%), `none = 10.575 ms` (+1.8%), `directional = 11.606 ms` (+11.7%).
  - **Batches (`C`)**: `1 = 10.477` (+0.9%), `2 = 10.779` (+3.8%), `4 = 11.364` (+9.4%).
  - **Tail / Cutoff / Unroll (`E`)**: `--tail-megakernel = 9.941 ms` (**-4.3%**, tail=`0.736`), `--no-early-bounce-cutoff = 10.482 ms` (+0.9%), **`--delta-unroll 4 = 9.555 ms` (-8.0%)**, **`--delta-unroll 2 = 7.970 ms` (-23.3%)**.
- **(d) Root-Cause Diagnosis**: Classic **inline `delta_unroll=6` + 1-VGPR dielectric occupancy bottleneck**. Despite `1.20M` B0 dielectric hits (`24.5%` of active hits), sorted `wavefront_shade_dielectric` spends `7.066 ms` (`67.1%` of total frame time!) tracing up to 6 inline non-opaque intersections per wave at `56.25%` occupancy (`145` VGPRs on `gfx1100`). Lowering `--delta-unroll` `6 -> 4 -> 2` progressively slashes runtime (`-8.0% -> -23.3%`), while `none` (`wavefront_shade`, 1 step/bounce, `62.5%` occ) drops to **`6.945 ms`** (`T_fixed = 0.420 ms` in 1080p scaling also reflects dielectric B1/B2 tail latency).

---

### 4.8 `DragonDispersion` (Showcase | 4K `11.245 ms`, 88.9 FPS | 1080p `3.077 ms`, Ratio `3.65x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.884 ms`, `Tonemap = 0.135 ms`, `Gaps = 0.036 / 0.026 / 0.029 = 0.091 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.013 | 4.021 | 0.000 | 0.000 | 0.029 | 1.376 | **5.440** | 1,667.4 MB total |
| **B1** | 0.015 | 2.934 | 0.021 | 0.266 | 0.000 | 0.490 | **3.726** | `150.1 GB/s` |
| **B2** | 0.005 | 0.765 | 0.004 | 0.114 | 0.000 | 0.033 | **0.922** | |
| **B3** | 0.003 | 0.029 | 0.001 | 0.038 | 0.000 | 0.000 | **0.072** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `134,995` tris, `91,216` non-opaque)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 4,898,495 | 3,699,382 | 1,199,113 | 0 | 0 | 0 | 4,592,633 (**55.4%**) |
| **B1** | 4,592,039 | 3,859,373 | 732,666 | 0 | 0 | 910,059 | 1,040,478 (**12.5%**) |
| **B2** | 1,039,961 | 698,655 | 341,306 | 0 | 0 | 238,904 | 125,350 (**1.5%**) |
| **B3** | 125,489 | 90,864 | 34,625 | 0 | 0 | 22,129 | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 11.176 ms`, picked `dual`)**:
  - **Sort (`A`)**: **`none = 7.388 ms` (-33.9%, fastest; `auto` is `+51.3%` slower!)**, `archetype = 11.099 ms` (-0.7%), `dual = 11.189 ms` (+0.1%).
  - **Sec-Sort (`B`)**: `coherent-k8 = 11.140 ms` (-0.3%), `none = 11.327 ms` (+1.4%), `directional = 12.251 ms` (+9.6%).
  - **Tail / Cutoff / Unroll (`E`)**: `--tail-megakernel = 10.645 ms` (**-4.8%**, tail=`0.872`), `--no-early-bounce-cutoff = 11.275 ms` (+0.9%), **`--delta-unroll 4 = 10.094 ms` (-9.7%)**, **`--delta-unroll 2 = 8.281 ms` (-25.9%)**.
- **(d) Root-Cause Diagnosis**: Identical mesh to `DragonAttenuation` (`134,995` tris) with dispersive RGB wavelength splitting adding `+0.683 ms` (`+9.7%`) to `Shade` (`7.066 -> 7.749 ms`) and slightly higher dielectric survival in B1/B2 (`733K / 341K` vs `651K / 242K`). Same exact bottleneck: `wavefront_shade_dielectric` (`145` VGPRs = `56.25%` occ) + `delta_unroll=6` causes `+51.3%` overhead vs `none` (`7.388 ms`) and `-25.9%` recovery at `--delta-unroll 2`.

---

### 4.9 `bmw-m6` (Research | 4K `10.022 ms`, 99.8 FPS | 1080p `2.920 ms`, Ratio `3.43x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.943 ms`, `Tonemap = 0.137 ms`, `Gaps = 0.035 / 0.023 / 0.030 = 0.088 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.012 | 3.738 | 0.000 | 0.000 | 0.030 | 1.762 | **5.543** | 2,347.8 MB total |
| **B1** | 0.017 | 1.950 | 0.016 | 0.322 | 0.000 | 0.295 | **2.600** | `237.5 GB/s` |
| **B2** | 0.002 | 0.453 | 0.006 | 0.136 | 0.000 | 0.046 | **0.642** | |
| **B3** | 0.004 | 0.041 | 0.001 | 0.043 | 0.000 | 0.000 | **0.089** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `792,746` tris, `17,008` non-opaque)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 8,294,400 | 234,537 | 343,404 | **7,716,459** | 0 | 0 | 7,725,284 (**93.1%**) |
| **B1** | 7,725,371 | 4,765,915 | 39,241 | 2,920,215 | 0 | 1,276,197 | 1,107,685 (**13.4%**) |
| **B2** | 1,107,474 | 650,713 | 31,326 | 425,435 | 0 | 292,890 | 172,225 (**2.1%**) |
| **B3** | 172,425 | 96,387 | 7,756 | 68,282 | 0 | 46,528 (avg 46,246) | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 9.967 ms`, picked `dual`)**:
  - **Sort (`A`)**: **`none = 9.389 ms` (-5.8%, fastest; `auto` is `+6.2%` slower)**, `archetype = 9.875 ms` (-0.9%), `dual = 9.970 ms` (+0.0%).
  - **Sec-Sort (`B`)**: `coherent-k8 = 9.923 ms` (-0.4%), `none = 10.123 ms` (+1.6%), `directional = 11.762 ms` (+18.0%).
  - **Batches (`C`)**: `1 = 9.990` (+0.2%), `2 = 10.343` (+3.8%), `4 = 11.295` (**+13.3%** slower).
- **(d) Root-Cause Diagnosis**: Dominated at Bounce 0 by **`7.72M` `Complex` car-paint/clearcoat rays (`93.0%` of primary!)** + `343K` headlamp/window `Dielectric` rays on a `792.7K`-triangle (`54.9 MB` BLAS) mesh. Under `auto (dual)`, `wavefront_shade_complex` (`126` VGPRs, `62.5%` occ, `75.5 KB` `.text`) + `wavefront_shade_dielectric` (`145` VGPRs, `56.25%` occ, `delta_unroll=6` on only `4.1%` glass rays!) lag monolithic `wavefront_shade` (`9.389 ms`) by **`0.578 ms` (`+6.2%`)** — and when `batches=4` fragments waves on that small dielectric/complex mix, runtime balloons `+13.3%` (`11.295 ms`). Even pure `archetype` beats `dual` by `0.092 ms`.

---

### 4.10 `breakfast-room` (Research | 4K `15.351 ms`, 65.1 FPS | 1080p `4.582 ms`, Ratio `3.35x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 1.237 ms`, `Tonemap = 0.138 ms`, `Gaps = 0.071 / 0.024 / 0.022 = 0.117 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.015 | 2.984 | 0.000 | 0.000 | 0.022 | **4.158** | **7.180** | 2,464.5 MB total |
| **B1** | 0.032 | 1.477 | 0.017 | **1.713** | 0.000 | 1.590 | **4.829** | `162.0 GB/s` |
| **B2** | 0.012 | 0.401 | 0.004 | **0.738** | 0.000 | 0.398 | **1.553** | |
| **B3** | 0.012 | 0.060 | 0.003 | **0.274** | 0.000 | 0.000 | **0.348** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `269,538` all-opaque tris)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 7,752,298 | 6,729,649 | 0 | 1,022,649 | 0 | 0 | 7,418,821 (**89.4%**) |
| **B1** | 7,417,799 | 6,560,675 | 0 | 857,124 | 0 | 1,745,855 | 1,956,377 (**23.6%**) |
| **B2** | 1,954,015 | 1,715,091 | 0 | 238,924 | 0 | 555,531 | 357,288 (**4.3%**) |
| **B3** | 356,188 | 293,047 | 0 | 63,141 | 0 | 142,148 | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 15.482 ms`, picked `dual`)**:
  - **Sort (`A`)**: `archetype = 15.404 ms` (**-0.5%**, fastest), `dual = 15.469 ms` (-0.1%), `none = 18.356 ms` (**+18.6%** slower).
  - **Sec-Sort (`B`)**: **`coherent-k8 = 14.958 ms` (-3.4% — largest Group B gain!)**, `none = 16.184 ms` (**+4.5%** — worst no-sec-sort loss), `directional = 17.554 ms` (+13.4%).
  - **Batches (`C`)**: `1 = 15.582` (+0.7%), `2 = 16.189` (+4.6%), `4 = 17.248` (**+11.4%** slower; `shadow` balloons `+1.296 ms`).
  - **Inline Shadows (`D`)**: `--no-inline-shadows = 15.843 ms` (**+2.3%** slower).
  - **Light Tree (`F`)**: `--no-light-tree = 15.705 ms` (+1.4%, null due to `Engine.cpp:2361`).
- **(d) Root-Cause Diagnosis**: **Most BVH-traversal- and shadow-bound scene in the entire 20-scene suite**: `Intersect` (`6.146 ms` = `40.0%` of frame!) + `Shadow` (`2.725 ms` = `17.8%`, **highest shadow time across all 20 scenes**) total `8.871 ms` (`57.8%` of frame), outstripping `Shade` (`4.922 ms`). Complex interior occlusion (`17.6 MB` BLAS) makes secondary-ray BVH coherence critical — explaining why `--sec-sort coherent-k8` wins **`-0.524 ms` (`-3.4%`)** while turning off secondary sort (`none`) loses **`+4.5%`**, and why fixing the `wavefront_intersect.comp:167` Morton `& 0x0FFFu` LSB bug (`dual` currently loses `-0.078 ms` to `archetype` despite zero glass) is especially impactful here. Note also the highest linear `T_fixed = 0.992 ms` (`3.35x` scaling), reflecting heavy late-bounce/shadow low-wave occupancy tail latency (`Shadow B3` takes `0.274 ms` for just `142K` rays!).

---

### 4.11 `classroom` (Research | 4K `10.708 ms`, 93.4 FPS | 1080p `2.705 ms`, Ratio `3.96x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 1.226 ms`, `Tonemap = 0.130 ms`, `Gaps = 0.052 / 0.028 / 0.016 = 0.096 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.014 | 1.508 | 0.000 | 0.000 | 0.016 | 1.718 | **3.256** | **3,540.3 MB** total |
| **B1** | 0.013 | 1.449 | 0.012 | 0.643 | 0.000 | 1.132 | **3.249** | `334.7 GB/s` |
| **B2** | 0.014 | 0.942 | 0.012 | 0.374 | 0.000 | 0.703 | **2.045** | |
| **B3** | 0.011 | 0.471 | 0.004 | 0.247 | 0.000 | 0.000 | **0.732** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `103,832` all-opaque tris)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 8,061,183 | 6,642,384 | 0 | 1,418,799 | 0 | 0 | 7,769,361 (**93.7%**) |
| **B1** | 7,769,237 | 6,955,418 | 0 | 813,819 | 0 | 3,878,463 | 4,696,975 (**56.6%**) |
| **B2** | 4,695,118 | 4,221,818 | 0 | 473,300 | 0 | 2,067,677 | 2,806,229 (**33.8%**) |
| **B3** | 2,804,112 | 2,504,546 | 0 | 299,566 | 0 | 1,348,892 (avg 1,350,353) | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 10.570 ms`, picked `dual`)**:
  - **Sort (`A`)**: `archetype = 10.560 ms` (-0.1%, fastest), `dual = 10.666 ms` (+0.9%), `none = 10.677 ms` (+1.0%).
  - **Sec-Sort (`B`)**: `direct = 10.570 ms` (fastest), `coherent-k8 = 10.590 ms` (+0.2%), `none = 10.801 ms` (+2.2%), `directional = 14.184 ms` (**+34.2%**).
  - **Batches (`C`)**: `1 = 10.588` (+0.2%), `2 = 10.763` (+1.8%), `4 = 10.715` (+1.4%).
  - **Inline Shadows (`D`)**: `--no-inline-shadows = 10.860 ms` (**+2.7%** slower).
  - **Tail / Cutoff (`E`)**: **`--tail-megakernel = 8.966 ms` (-15.2%, tail=`2.313 ms`)**, `--no-early-bounce-cutoff = 10.676 ms` (+1.0%).
  - **Light Tree (`F`)**: `--no-light-tree = 10.668 ms` (+0.9%, null due to `Engine.cpp:2361`).
  - **Formats / Features (`G`)**: `rg11 = 10.887 ms` (+3.0%, TM `0.116`), `fp32 = 10.952 ms` (+3.6%, TM `0.285`, `+384 MB`), `--restir = 12.939 ms` (**+22.4%**).
- **(d) Root-Cause Diagnosis**: **Highest estimated queue traffic in the 20-scene suite (`3,540.3 MB/frame`, `334.7 GB/s`)** and **`99.0%` pixel-linear 4K scaling (`3.96x`)**. Diffuse interior bouncing keeps `56.6%` of rays alive at B2 and **`33.8%` (`2.81M` rays)** at B3, making `B1` (`3.249 ms`) equal to `B0` (`3.256 ms`) and leaving **`3.91 ms`** across `B1 Intersect + B2 + B3`! Merging B2–B3 into `--tail-megakernel` immediately saves **`1.604 ms` (`-15.2%`)** (`1.132 + 2.045 + 0.732 = 3.909 ms -> 2.313 ms` tail).

---

### 4.12 `cornell-caustic` (Research | 4K `7.240 ms`, 138.1 FPS | 1080p `1.891 ms`, Ratio `3.83x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.814 ms`, `Tonemap = 0.143 ms`, `Gaps = 0.034 / 0.022 / 0.010 = 0.066 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.009 | 1.163 | 0.000 | 0.000 | 0.010 | 0.585 | **1.766** | 2,278.8 MB total |
| **B1** | 0.005 | 0.997 | 0.008 | 0.426 | 0.000 | 0.766 | **2.202** | `321.1 GB/s` |
| **B2** | 0.012 | **1.372** | 0.013 | 0.279 | 0.000 | 0.399 | **2.075** | |
| **B3** | 0.008 | 0.162 | 0.001 | 0.000 | 0.000 | 0.000 | **0.171** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `103,487` tris, `103,451` non-opaque)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 4,621,693 | 1,107,565 | **3,514,126** | 0 | 2 | 0 | 4,274,710 (**51.5%**) |
| **B1** | 4,273,941 | **3,830,327** | 443,610 | 0 | 4 | 3,168,938 | 2,905,714 (**35.0%**) |
| **B2** | 2,904,160 | 1,888,447 | **1,015,713** | 0 | 0 | 1,587,503 | 1,561,776 (**18.8%**) |
| **B3** | 1,561,007 | 1,204,521 | 356,484 | 0 | 2 | 0 | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 7.132 ms`, picked `dual`)**:
  - **Sort (`A`)**: **`none = 6.044 ms` (-15.3%, fastest; `auto` is `+18.0%` slower)**, `archetype = 7.084 ms` (-0.7%), `dual = 7.144 ms` (+0.2%).
  - **Tail / Cutoff / Unroll (`E`)**: **`--tail-megakernel = 5.330 ms` (-25.3% — largest Group E speedup!, tail=`1.114 ms`, traffic `2,279 -> 1,790 MB`)**, `--no-early-bounce-cutoff = 7.165 ms` (+0.5%), `--delta-unroll 4 = 7.186 ms` (+0.8%), `--delta-unroll 2 = 6.754 ms` (**-5.3%**).
  - **Caustics (`G`)**: `--caustics = 9.664 ms` (**+35.5%**, `+2.532 ms`; `total_gaps_ms` jumps `0.065 -> 1.057 ms`).
- **(d) Root-Cause Diagnosis**: Refracting water/glass slab hits **`3.51M` Dielectric rays at B0** and unrolls into diffuse floor (`3.83M` B1) before rebounding into `1.02M` B2 dielectric rays (`Shade B2` spikes to `1.372 ms` — higher than B0 `1.163 ms` and B1 `0.997 ms`!). Enabling `--tail-megakernel` sweeps `B1 Intersect` (`0.766 ms`) + `B2` (`2.075 ms`) + `B3` (`0.171 ms`) (`= 3.012 ms`) into a **`1.114 ms`** zero-atomic pass, saving **`-25.3%` (`1.802 ms`)** and cutting queue traffic by `~489 MB`. Monolithic `none` (`6.044 ms`) also beats `auto` by `15.3%` by skipping `delta_unroll=6` + `145`-VGPR dielectric pressure. Under `--caustics`, photon tracing/splatting/filtering adds `+2.532 ms` including `~1.0 ms` of inter-pass gaps.

---

### 4.13 `glass-of-water` (Research | 4K `14.304 ms`, 69.9 FPS | 1080p `4.299 ms`, Ratio `3.33x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.807 ms`, `Tonemap = 0.140 ms`, `Gaps = 0.036 / 0.054 / 0.028 = 0.118 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.014 | **4.269** | 0.000 | 0.000 | 0.028 | 1.060 | **5.370** | 2,712.4 MB total |
| **B1** | 0.014 | **4.207** | 0.031 | 0.301 | 0.000 | 0.575 | **5.128** | `191.5 GB/s` |
| **B2** | 0.005 | **2.281** | 0.022 | 0.145 | 0.000 | 0.256 | **2.709** | |
| **B3** | 0.003 | 0.081 | 0.001 | 0.000 | 0.000 | 0.000 | **0.085** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `406,387` tris, `406,381` non-opaque)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 8,294,400 | 6,737,812 | **1,556,588** | 0 | 0 | 0 | 8,140,478 (**98.1%**) |
| **B1** | 8,140,048 | 6,941,909 | **1,121,953** | 0 | 76,186 | 2,057,427 | 2,280,354 (**27.5%**) |
| **B2** | 2,281,237 | 1,597,726 | **621,697** | 0 | 61,814 | 637,483 | 757,751 (**9.1%**) |
| **B3** | 758,958 | 498,999 | 248,429 | 0 | 11,530 | 0 | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 14.320 ms`, picked `dual`)**:
  - **Sort (`A`)**: **`none = 6.832 ms` (-52.3%, fastest; `auto` is `+109.6%` / `2.10x` slower!)**, `archetype = 14.291 ms` (-0.2%), `dual = 14.326 ms` (+0.0%).
  - **Inline Shadows (`D`)**: `--no-inline-shadows = 15.391 ms` (**+7.5%** — worst Group D regression!).
  - **Tail / Cutoff / Unroll (`E`)**: `--tail-megakernel = 11.828 ms` (**-17.4%**, tail=`0.850`), `--no-early-bounce-cutoff = 14.353 ms` (+0.2%), **`--delta-unroll 4 = 12.268 ms` (-14.3%)**, **`--delta-unroll 2 = 9.137 ms` (-36.2%)**.
  - **Caustics (`G`)**: `--caustics = 18.252 ms` (**+27.5%**, `+3.932 ms`; `total_gaps_ms` `0.116 -> 1.178 ms`).
- **(d) Root-Cause Diagnosis**: **Headline pathological case in the entire benchmark (`Shade = 10.838 ms` = `75.8%` of frame)**. `99.998%` of the `406K`-tri mesh (`406,381 / 406,387` tris) is flagged `non_opaque`. Under `auto (dual)`, `wavefront_shade_dielectric` (`145` VGPRs = `56.25%` occ on `gfx1100`, `4` BVH blocks) forces `1.56M -> 1.12M -> 622K` glass rays through **`delta_unroll = 6` inline `rayQueryEXT` loops** over that dense non-opaque glass BLAS (`25.1 MB`), making B0+B1+B2 `Shade` alone cost `10.757 ms` (`T_fixed = 0.964 ms` from late-wave dielectric stragglers!). Lowering `delta_unroll` (`6 -> 4 -> 2`) cuts runtime `-14.3% -> -36.2%`, `--tail-megakernel` saves `-17.4%` (`3.37 -> 0.85 ms` on tail bounces), and monolithic `--wavefront-sort none` cuts frame time **in half (`6.832 ms`)**.

---

### 4.14 `living-room` (Research | 4K `11.602 ms`, 86.2 FPS | 1080p `3.247 ms`, Ratio `3.57x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 1.075 ms`, `Tonemap = 0.137 ms`, `Gaps = 0.046 / 0.028 / 0.021 = 0.095 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.011 | 2.658 | 0.000 | 0.000 | 0.021 | 2.057 | **4.747** | 2,768.5 MB total |
| **B1** | 0.018 | 2.196 | 0.016 | 0.450 | 0.000 | 0.938 | **3.619** | `241.5 GB/s` |
| **B2** | 0.012 | 1.045 | 0.010 | 0.235 | 0.000 | 0.371 | **1.675** | |
| **B3** | 0.005 | 0.174 | 0.002 | 0.095 | 0.000 | 0.000 | **0.276** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, **ONLY scene with `alpha_mask_active = true`**: `3,264` alphaNO, `11,166` solidNO, `128,733` op)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 7,804,855 | 3,956,274 | 18,255 | **3,830,326** | 0 | 0 | 7,429,756 (**89.6%**) |
| **B1** | 7,429,987 | 4,407,352 | 9,238 | **3,013,397** | 0 | 2,431,381 | 2,811,734 (**33.9%**) |
| **B2** | 2,811,217 | 1,549,652 | 4,069 | **1,257,496** | 0 | 1,123,506 (avg 1,123,389) | 994,868 (**12.0%**) |
| **B3** | 994,140 | 548,103 | 1,336 | **444,701** | 0 | 426,111 (avg 426,172) | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 11.540 ms`, picked `dual`)**:
  - **Sort (`A`)**: `archetype = 11.421 ms` (**-1.0%**, fastest), `dual = 11.531 ms` (-0.1%), `none = 11.695 ms` (+1.3%).
  - **Sec-Sort (`B`)**: `coherent-k8 = 11.419 ms` (**-1.1%**), `none = 11.803 ms` (+2.3%), `directional = 13.482 ms` (+16.8%).
  - **Batches (`C`)**: `1 = 11.611` (+0.6%), `2 = 11.777` (+2.0%), `4 = 12.380` (+7.3%).
  - **Tail / Cutoff (`E`)**: `--tail-megakernel = 10.638 ms` (**-7.8%**, tail=`2.007 ms`), `--no-early-bounce-cutoff = 11.584 ms` (+0.4%).
  - **Light Tree (`F`)**: `--no-light-tree = 11.645 ms` (+0.9%, null due to `Engine.cpp:2361`).
- **(d) Root-Cause Diagnosis**: Unique in two respects: (1) **50/50 Diffuse (`3.96M`) + Complex (`3.83M`) split across all bounces** (`wavefront_shade_complex_sec` spends `129` VGPRs / `70.8 KB` `.text` on `3.01M / 1.26M / 445K` B1..B3 rays), and (2) **the sole benchmark scene activating `wavefront_intersect_alpha`** (`60` VGPRs, `2` BVH sites, `12.2 KB`, forcing mip-0 candidate alpha texture evaluations in `shading_core.glsl:68` & `wavefront_shadow_inline.glsl:69`). Both `--tail-megakernel` (`-7.8%`), `coherent-k8` (`-1.1%`), and switching `auto` from `dual` to `archetype` (`-1.0%`) deliver clean gains.

---

### 4.15 `modern-hall` (Research | 4K `15.730 ms`, 63.6 FPS | 1080p `3.876 ms`, Ratio `4.06x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.743 ms`, `Tonemap = 0.131 ms`, `Gaps = 0.035 / 0.044 / 0.057 = 0.136 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.007 | **5.660** | 0.000 | 0.000 | 0.057 | 1.167 | **6.891** | **3,438.3 MB** total |
| **B1** | 0.006 | **3.167** | 0.011 | 0.209 | 0.000 | 0.872 | **4.264** | `220.4 GB/s` |
| **B2** | 0.016 | **1.896** | 0.029 | 0.152 | 0.000 | 0.573 | **2.666** | |
| **B3** | 0.006 | **0.955** | 0.004 | 0.000 | 0.000 | 0.000 | **0.965** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `31,087` tris, `168` MIS lights)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 8,294,344 | 5,755,315 | 139,707 | 2,301,971 | 97,351 | 0 | 7,968,598 (**96.1%**) |
| **B1** | 7,968,454 | 5,285,844 | 324,363 | 1,941,141 | 417,106 | 1,400,998 | 5,226,334 (**63.0%**) |
| **B2** | 5,227,389 | 3,546,840 | 136,831 | 1,302,719 | 240,999 | 888,651 (avg 886,998) | 3,450,184 (**41.6%**) |
| **B3** | 3,450,494 | 2,378,964 | 105,346 | 806,831 | 159,353 | 0 | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 15.880 ms`, picked `dual`)**:
  - **Sort (`A`)**: `archetype = 15.692 ms` (**-1.2%**, fastest; largest `archetype vs dual` absolute delta **`-0.188 ms`**), `dual = 15.910 ms` (+0.2%), `none = 23.017 ms` (**+44.9%** slower).
  - **Light Tree (`F`)**: `--no-light-tree = 15.997 ms` (+0.7%, null due to `Engine.cpp:2361`).
- **(d) Root-Cause Diagnosis**: **Highest late-bounce path survival (`41.6%` / `3.45M` active rays at Bounce 3!) and the ONLY scene exhibiting superlinear 4K scaling (`4.06x`, `T_fixed = -0.075 ms`)**. At `1080p`, `~860 MB` of queues + `2.1 MB` BLAS + Light Tree stay much warmer in the `96 MB` Infinity Cache than at `4K` (`3.44 GB` queue traffic across 4 dense bounces × `168` MIS lights × all 4 material archetypes!). Every bounce runs `diffuse` (`81.4/53.4 KB` `.text`) + `complex` (`75.5/70.8 KB` `.text`) + `dielectric` (`58.8 KB`) + `emissive` (`208 B` `Material` reads on `914K` total emitter hits!), explaining why `Shade` eats **`11.678 ms` (`74.2%`)**, why `archetype` beats `dual` (`-0.188 ms`), and why `modern-hall` is a prime candidate for `--tail-megakernel`, Light Tree compaction, and `RayHit` `32 -> 16 B` reduction.

---

### 4.16 `veach-ajar` (Research | 4K `7.875 ms`, 127.0 FPS | 1080p `2.398 ms`, Ratio `3.28x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.871 ms`, `Tonemap = 0.143 ms`, `Gaps = 0.037 / 0.016 / 0.010 = 0.063 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.021 | 1.429 | 0.000 | 0.000 | 0.010 | 1.313 | **2.773** | **3,204.1 MB** total |
| **B1** | 0.011 | 1.416 | 0.001 | 0.264 | 0.000 | 0.728 | **2.420** | `414.4 GB/s` |
| **B2** | 0.001 | 0.794 | 0.001 | 0.124 | 0.000 | 0.441 | **1.362** | |
| **B3** | 0.004 | 0.222 | 0.014 | 0.000 | 0.000 | 0.000 | **0.241** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `382,690` tris, `1` light)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 8,294,400 | 8,218,545 | 75,855 | 0 | 0 | 0 | 8,002,585 (**96.5%**) |
| **B1** | 8,002,684 | 7,932,894 | 69,167 | 0 | 623 | 2,311,792 (avg 2,312,724) | 4,116,410 (**49.6%**) |
| **B2** | 4,116,014 | 4,074,825 | 37,243 | 0 | 3,946 | 1,001,203 (avg 1,000,640) | 2,368,696 (**28.6%**) |
| **B3** | 2,367,511 | 2,344,777 | 20,437 | 0 | 2,297 | 0 | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 7.818 ms`, picked `dual`)**:
  - **Sort (`A`)**: `dual = 7.799 ms` (-0.2%), `archetype = 7.800 ms` (-0.2%), `none = 8.052 ms` (+3.0%).
  - **Sec-Sort (`B`)**: **`coherent-k8 = 7.683 ms` (-1.7%)**, `none = 7.936 ms` (+1.5%), **`directional = 11.662 ms` (+49.2% — worst Group B regression!)**.
  - **Batches (`C`)**: `1 = 7.817` (-0.0%), `2 = 8.206` (+5.0%), `4 = 8.892` (+13.7%).
  - **Inline Shadows (`D`)**: `--no-inline-shadows = 7.836 ms` (+0.2%).
  - **Tail / Cutoff (`E`)**: **`--tail-megakernel = 6.626 ms` (-15.2%, tail=`1.153 ms`)**, `--no-early-bounce-cutoff = 7.818 ms` (+0.0%).
  - **Light Tree (`F`)**: `--no-light-tree = 7.847 ms` (+0.4%).
  - **Formats / Features (`G`)**: `rg11 = 7.791 ms` (-0.3%, TM `0.132`), `fp32 = 8.069 ms` (+3.2%, TM `0.287`, `+128 MB`), `--restir = 9.921 ms` (**+26.9%**).
- **(d) Root-Cause Diagnosis**: Classic high-survival (`96.5 -> 49.6 -> 28.6%`) indirect-lighting interior (`3.20 GB/frame` queue traffic, **`414.4 GB/s` — highest sustained multi-bounce bandwidth rate in suite**). `99.1%` of rays are `Diffuse` shading a single area light behind the ajar door (`126K` non-opaque tris in `28.1 MB` BLAS). Merging B1 Intersect (`0.728 ms`) + B2 (`1.362 ms`) + B3 (`0.241 ms`) (`2.331 ms`) into `--tail-megakernel` (`1.153 ms`) slashes frame time by **`-15.2%`**, and `coherent-k8` cuts another `-1.7%`; conversely, `8`-bin `directional` sort explodes by `+49.2%` due to scatter/dispatch overhead across `4.1M`/`2.4M` secondary rays.

---

### 4.17 `BuickRiviera` (USD | 4K `20.686 ms`, 48.3 FPS | 1080p `5.439 ms`, Ratio `3.80x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 1.270 ms`, `Tonemap = 0.137 ms`, `Gaps = 0.054 / 0.058 / 0.059 = 0.171 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.011 | **8.651** | 0.000 | 0.000 | 0.059 | 2.317 | **11.038** | 2,567.2 MB total |
| **B1** | 0.033 | **5.957** | 0.049 | 0.814 | 0.000 | 0.427 | **7.280** | `124.9 GB/s` |
| **B2** | 0.007 | 0.544 | 0.008 | 0.179 | 0.000 | 0.043 | **0.781** | |
| **B3** | 0.003 | 0.050 | 0.001 | 0.061 | 0.000 | 0.000 | **0.115** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `861,243` tris, `43` mats, `132` lights, `20` tex, `envSelectProb = 0.50`, `4.92 GB` VRAM)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 8,294,400 | 6,968,400 | 253,516 | 1,072,484 | 0 | 0 | 8,150,973 (**98.3%**) |
| **B1** | 8,151,268 | 7,405,462 | 38,266 | 408,395 | 299,145 | 3,595,451 (avg 3,595,664) | 1,329,953 (**16.0%**) |
| **B2** | 1,329,072 | 1,218,745 | 27,808 | 43,702 | 38,817 | 326,629 | 92,517 (**1.1%**) |
| **B3** | 92,260 | 74,143 | 4,581 | 11,255 | 2,281 | 12,058 (avg 11,972) | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 20.818 ms`, picked `dual`)**:
  - **Sort (`A`)**: `archetype = 20.795 ms` (-0.1%), `dual = 20.820 ms` (+0.0%), `none = 23.877 ms` (**+14.7%** slower).
  - **Inline Shadows (`D`)**: `--no-inline-shadows = 20.685 ms` (**-0.6%**, Bounce-0 `shade+shadow` **`-0.188 ms`** [`8.711 -> 6.991 + 1.532 ms`] and `intersect` `-0.042 ms`, partially offset by `2,567 -> 3,048 MB` queue traffic).
- **(d) Root-Cause Diagnosis**: Second-slowest scene (`20.686 ms`) and largest VRAM consumer (`4,921.9 MB`, `861.2K` tris / `57.3 MB` BLAS / `20` textures / `132` lights / `envSelectProb = 0.50`). **`B0 + B1` account for `88.6%` of frame time (`18.318 ms`)**, driven by `15.202 ms` of `Shade` (`73.5%` of total frame time) evaluating `20` textures (`textureQueryLevels` + `textureLod` double-`nonuniformEXT` waterfalls), `132`-light Light Tree traversal, and `50%` HDRI MIS sampling on `7.0M–7.4M` Diffuse + `1.07M` Complex rays, before terminating sharply at B2 (`16.0 -> 1.1%`). Alongside `cyber-city`, this is one of the two scenes where Bounce-0 inline shadow register/I-cache pressure slightly exceeds detached `wavefront_shadow` (`54` VGPRs, `100%` occ).

---

### 4.18 `coffee-maker` (USD | 4K `9.167 ms`, 109.1 FPS | 1080p `2.580 ms`, Ratio `3.55x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.806 ms`, `Tonemap = 0.141 ms`, `Gaps = 0.029 / 0.019 / 0.032 = 0.080 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.011 | **4.000** | 0.000 | 0.000 | 0.032 | 1.249 | **5.292** | 2,325.6 MB total |
| **B1** | 0.008 | 1.822 | 0.014 | 0.446 | 0.000 | 0.160 | **2.450** | `257.7 GB/s` |
| **B2** | 0.007 | 0.212 | 0.004 | 0.120 | 0.000 | 0.038 | **0.380** | |
| **B3** | 0.003 | 0.029 | 0.001 | 0.000 | 0.000 | 0.000 | **0.033** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `235,271` tris, `67,072` non-opaque)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 8,294,400 | 82,881 | 296,682 | **6,937,280** | 977,557 | 0 | 7,306,066 (**88.1%**) |
| **B1** | 7,306,014 | 3,913,371 | 88,047 | **2,729,776** | 574,820 | 2,299,954 | 987,924 (**11.9%**) |
| **B2** | 987,199 | 581,055 | 96,521 | 242,239 | 67,384 | 203,296 (avg 203,937) | 157,023 (**1.9%**) |
| **B3** | 155,993 | 65,818 | 42,315 | 41,291 | 6,569 | 0 | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 9.119 ms`, picked `dual`)**:
  - **Sort (`A`)**: `archetype = 9.055 ms` (**-0.7%**, fastest), `dual = 9.127 ms` (+0.1%), `none = 9.236 ms` (+1.3%).
  - **Tail / Cutoff / Unroll (`E`)**: `--tail-megakernel = 8.963 ms` (**-1.7%**, tail=`0.450`), `--no-early-bounce-cutoff = 9.160 ms` (+0.5%), `--delta-unroll 4 = 9.012 ms` (-1.2%), **`--delta-unroll 2 = 8.745 ms` (-4.1%)**.
- **(d) Root-Cause Diagnosis**: Metallic/glossy showcase where **`6.94M` (`83.6%`) of Bounce 0 rays hit `Complex`** and `978K` (`11.8%`) hit `Emissive` (`208 B` `Material` reads in `wavefront_shade_emissive` + `complex`), followed by `2.73M` Complex + `3.91M` Diffuse in B1 (`84.4%` of frame finished by B1). Because `wavefront_shade_complex` (`126` VGPRs, `62.5%` occ) & `complex_sec` (`129` VGPRs) dominate, optimizing `Complex` (especially stripping `208 B` `Material` at `line 314` on `_sec`) and lowering `delta_unroll -> 2` (`-4.1%` from `297K` glass rays) yield the biggest immediate return.

---

### 4.19 `Kitchen_set` (USD | 4K `3.236 ms`, 309.0 FPS | 1080p `1.081 ms`, Ratio `2.99x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 0.546 ms`, `Tonemap = 0.140 ms`, `Gaps = 0.018 / 0.009 / 0.007 = 0.034 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.006 | 0.769 | 0.000 | 0.000 | 0.007 | 0.572 | **1.355** | 1,078.0 MB total |
| **B1** | 0.006 | 0.346 | 0.003 | 0.205 | 0.000 | 0.141 | **0.700** | `348.2 GB/s` |
| **B2** | 0.004 | 0.099 | 0.004 | 0.162 | 0.000 | 0.063 | **0.331** | |
| **B3** | 0.002 | 0.037 | 0.002 | 0.058 | 0.000 | 0.000 | **0.099** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `536,949` all-opaque tris, `135` mats)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 2,093,810 | 2,093,810 | 0 | 0 | 0 | 0 | 2,066,128 (**24.9%**) |
| **B1** | 2,066,057 | 2,066,057 | 0 | 0 | 0 | 316,670 (avg 317,465) | 606,582 (**7.3%**) |
| **B2** | 607,344 | 607,344 | 0 | 0 | 0 | 360,723 (avg 360,225) | 238,485 (**2.9%**) |
| **B3** | 238,736 | 238,736 | 0 | 0 | 0 | 44,047 (avg 43,809) | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 3.304 ms`, picked `archetype`)**:
  - **Sort (`A`)**: `dual = 3.304 ms` (+0.0%), `auto (archetype) = 3.304 ms` (tied!), `archetype = 3.313 ms` (+0.3%), `none = 3.755 ms` (+13.6%).
  - **Sec-Sort (`B`)**: `coherent-k8 = 3.297 ms` (-0.2%), `none = 3.358 ms` (+1.6%), `directional = 3.805 ms` (+15.2%).
  - **Batches (`C`)**: `1 = 3.328` (+0.7%), **`2 = 3.627` (+9.8%)**, **`4 = 4.257` (+28.8% — worst Group C penalty!)**.
  - **Inline Shadows (`D`)**: `--no-inline-shadows = 3.337 ms` (+1.0%).
  - **AS (`F`)**: `--macro-blas = 3.363 ms` (+1.8%, monolithic single instance skipped).
- **(d) Root-Cause Diagnosis**: Second-fastest scene (`3.236 ms`) and **lowest 4K/1080p ratio (`2.99x`, `T_fixed = 0.363 ms` = `33.5%` @ 1080p!)**. Like `DamagedHelmet`, `74.8%` of primary rays miss into the HDRI sky (`2.09M` hit `100% Diffuse`), leaving B2 (`607K` rays) and B3 (`239K` rays) as small sparse dispatches. When `--batches 4` chops those already-small queues by another `4x` (`~60K` B3 rays/batch, `Classify` 4x, `4x` indirect barriers & small shadow batches), runtime explodes **`+28.8%`**, cleanly demonstrating why sub-saturated wavefronts require single-batch execution (`batches=1`) plus `--tail-megakernel`.

---

### 4.20 `PointInstancedMedCity` (USD | 4K `9.469 ms`, 105.6 FPS | 1080p `2.491 ms`, Ratio `3.80x`)

- **(a) Stage & Bounce Waterfall (4K, `Classify = 2.092 ms`, `Tonemap = 0.136 ms`, `Gaps = 0.058 / 0.015 / 0.015 = 0.088 ms`)**:

| Bounce | Gap1 (ms) | Shade (ms) | Gap2 (ms) | Shadow (ms) | Gap3 (ms) | Intersect (ms) | Bounce Total (ms) | Est. Traffic |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 0.023 | 1.681 | 0.000 | 0.000 | 0.015 | **2.513** | **4.231** | 2,071.3 MB total |
| **B1** | 0.019 | 0.851 | 0.008 | 0.285 | 0.000 | 0.777 | **1.939** | `221.9 GB/s` |
| **B2** | 0.011 | 0.294 | 0.004 | 0.212 | 0.000 | 0.321 | **0.843** | |
| **B3** | 0.005 | 0.081 | 0.003 | 0.073 | 0.000 | 0.000 | **0.162** | |

- **(b) Ray Survival Funnel & Material Archetypes (`Primary = 8,294,400`, `40,001` TLAS instances, `49.18M` instanced tris)**:

| Bounce | Active Rays | Diffuse | Dielectric | Complex | Emissive | Shadow Rays | Next Rays (% of Primary) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **B0** | 5,804,816 | 5,804,816 | 0 | 0 | 0 | 0 | 5,745,212 (**69.3%**) |
| **B1** | 5,745,429 | 5,745,407 | **22** | 0 | 0 | 917,864 (avg 918,374) | 1,632,565 (**19.7%**) |
| **B2** | 1,631,356 | 1,631,277 | **79** | 0 | 0 | 628,267 (avg 628,534) | 662,882 (**8.0%**) |
| **B3** | 663,280 | 663,262 | **18** | 0 | 0 | 201,581 (avg 201,987) | 0 (**0.0%**) |

- **(c) Measured Sweep 3 Ablations (`Group A auto = 9.439 ms`, picked `dual`)**:
  - **Sort (`A`)**: `auto (dual) = 9.439 ms` (tied), `dual = 9.444 ms` (+0.1%), `archetype = 9.446 ms` (+0.1%), `none = 10.430 ms` (**+10.5%** slower).
  - **Sec-Sort (`B`)**: **`coherent-k8 = 9.237 ms` (-2.1%)**, `none = 9.691 ms` (+2.7%), `directional = 11.458 ms` (+21.4%).
  - **Batches (`C`)**: `1 = 9.456` (+0.2%), `2 = 9.497` (+0.6%), `4 = 9.503` (+0.7%).
  - **Inline Shadows (`D`)**: `--no-inline-shadows = 9.650 ms` (+2.2%).
  - **AS (`F`)**: `--macro-blas = 9.534 ms` (+1.0% — skipped due to `250000`-tri budget check!).
  - **Formats (`G`)**: `rg11 = 9.572 ms` (+1.4%, TM `0.122`), `fp32 = 9.910 ms` (+5.0%, TM `0.298`, `+384 MB`).
- **(d) Root-Cause Diagnosis**: **TLAS traversal & Classify benchmark (`Intersect = 3.611 ms` + `Classify = 2.092 ms` = `60.2%` of total frame time, vs `2.907 ms` Shade)**. With **`40,001` TLAS instances (`9.09 MB` TLAS!)** representing `49.18M` city triangles, top-level BVH traversal and `InstanceGPU` (`144 B` × 2–3x loads in `wavefront_classify.comp`) dominate GPU time (`Classify` is `3.4x` slower than `cornell-box`!). Look closely at the material counters: **only `22 + 79 + 18` rays across 8.3M pixels hit Dielectric at B1..B3** (yet `auto` chose `dual`!). Because traversal dominates secondary bounces, **`--sec-sort coherent-k8` scores its second-biggest win (`-2.1%`, `-0.202 ms`)**, while `--macro-blas` was wrongly blocked by the flat `250,000`-triangle budget (`Geometry budget (250000 tris) precluded clustering` at [SceneGeometryPipeline.cpp:292-352](file:///c:/Users/naoki/Development/Pathways/src/scene/SceneGeometryPipeline.cpp#L292-L352)) even though `9` compact proto-BLASes (`1.53 MB`) instanced `40,001` times would benefit from a raised or adaptive instancing threshold if clustered in 3D `(X,Y,Z)`.

---

## 5. Holistic Cross-Scene Synthesis & Taxonomy of Bottlenecks

### 5.1 Five-Cluster Hardware Limiter Taxonomy Across All 20 Scenes

| Cluster | Scenes (4K ms) | Shared Bottleneck Signature | Primary Root Causes | Highest-ROI Interventions |
|---|---|---|---|---|
| **C1. Dielectric / Delta-Unroll Divergent Shading** | `glass-of-water` (`14.30`), `DragonDispersion` (`11.25`), `DragonAttenuation` (`10.52`), `cornell-caustic` (`7.24`) | `Shade` = `51%–76%`; `--wavefront-sort none` **18%–110% faster** than `auto`; `--delta-unroll 2` saves `5.3%–36.2%` | `wavefront_shade_dielectric` (`145` VGPRs = `56.25%` occ on `gfx1100`) runs `delta_unroll = 6` inline `rayQueryEXT` chains on non-opaque meshes + lacks `_sec` variant | `--delta-unroll 2..4` default, reclaim 2 VGPRs (`<=144`), add `_sec`, fix `baseLod=0.0` PSR bug, `--tail-megakernel` |
| **C2. Multi-Light / Complex BSDF Shade-Bound** | `cyber-city` (`23.34`), `BuickRiviera` (`20.69`), `modern-hall` (`15.73`), `many-lights` (`9.47`), `coffee-maker` (`9.17`), `bmw-m6` (`10.02`), `infinity-mirror` (`4.70`) | `Shade` = `61.7%–74.2%`; large `Light` count (`15–640K`) or dominant `Complex` (`1M–7.7M` B0 rays) | `53.4–85.8 KB` `.text` L0 I-cache thrashing (`4–5` BVH blocks), `62.5%` occ (`complex` `126–129` VGPRs), Light Tree `32`-deep traversal, `208 B` `Material` in `emissive` & `complex_sec` | Deduplicate `traceShadowRayInline` + NRC trace (`<32 KB`), switch `complex_sec:314` to `ShadeMaterial`, fix `emissive:244`, `archetype` over `dual` |
| **C3. BVH Traversal & High-Survival Memory-Bound** | `breakfast-room` (`15.35`), `living-room` (`11.60`), `classroom` (`10.71`), `veach-ajar` (`7.88`), `cornell-box` (`6.03`) | `Intersect + Shadow` `29%–58%` (`4.8–8.9 ms`) OR high B2–B3 survival (`12%–34%`) pushing **`2.4–3.54 GB`** queue traffic | Scattered secondary/shadow BVH traversals (`3.5–4.1` GB queue traffic @ 4K), `alphaNO` candidate loop (`living-room`), `& 0x0FFFu` LSB Morton scatter, `RayHit` `32 B` write traffic | `--tail-megakernel` (`-7.8%` to `-15.2%`), `--sec-sort coherent-k8` (`-1.0%` to `-3.4%`), compact `RayHit` (`32 -> 16 B`), fix Morton `& 0x0FFFu` |
| **C4. Multi-Instance TLAS & Classify Overhead** | `PointInstancedMedCity` (`9.47`), `mineshaft` (`17.15`) *(+ `cyber-city` TLAS/Classify)* | `Classify` spikes **`1.63–2.09 ms` (`17%–22%`)** + heavy B0 `Intersect` (`2.51–3.02 ms`) | `526–40,001` TLAS instances (`120 KB–9.1 MB`), `208 B` `Material` (3×) + `144 B` `InstanceGPU` (2–3×) loads in `classify`, Mineshaft forced `MODE_BUILD_KHR` + CPU Light Tree rebuild | Strip `208 B` `Material` from `classify`, `coherent-k8` (`-2.1%`), 3D Macro-BLAS for high-instance-count cities, Mineshaft periodic TLAS refit |
| **C5. Low-Hit / Fixed-Overhead & Micro-Dispatch-Bound** | `DamagedHelmet` (`1.97`), `Kitchen_set` (`3.24`) | `75%–80%` primary misses; `Classify + Tonemap + Fixed` = **`32%–43%`**; lowest `4K/1080` ratio (`2.99x`) | Late bounces drop to `52K–239K` rays (`0.6%–2.9%`), making per-pass dispatch latency, `m_indirectReadback` syncs, and 4x batching (`+28.8%`!) dominant | `--tail-megakernel`, keep `batches=1`, remove `ALL_COMMANDS_BIT` / per-batch readback copies, increase `local_size` `32 -> 64` |

### 5.2 Cross-Suite Architectural Findings

1. **Where Bandwidth Truly Bites — And Why Multi-Batching Fails on 24 GB RDNA 3**:
   Each Wavefront pass writes/reads a **`352 B/px` double-buffered queue footprint (`176 B/px/slot`)** totaling **`2,784.75 MiB` @ 4K** (`1,392.38 MiB/slot`) and **`696.47 MiB` @ 1080p** (`348.23 MiB/slot`) in [WavefrontPipeline.cpp:230-277, 1340-1356](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.cpp#L230-L277):
   - Two components inside that `176 B/px/slot` budget are pure waste or trivially half-sized:
     - **`SecondaryIndexQueue` (`64 B/px` double-buffered = `506.25 MiB` @ 4K)** is allocated unconditionally ([WavefrontPipeline.cpp:243, 255](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.cpp#L243-L255)) even though `direct`, `archetype`, and `none` sorting never read it!
     - **`RayHit` (`32 B/px` per slot = `506.25 MiB` @ 4K)** carries dead `hitData1.z` (`4 B` barycentrics written in `classify`/`intersect`, **never read anywhere**), `hitData1.w[15:1]` unused bits (`only bit 0` ML specular VMV at `wavefront_common.glsl:714` is used), `hitData1.xy` unused on Bounce 0 and non-emitters (`shading_core.glsl:168-174`), plus `packHalf2x16` low-precision wrapped UV packing (`1/1024` vs `packUnorm2x16`). Compacting `RayHit` from `32 B -> 16 B` + gating `SecondaryIndexQueue` immediately saves **`1,012.50 MiB` (`36.4%`)** of 4K queue VRAM and reduces per-ray queue transfer at 100% survival from **`168 B` (B0) / `216 B` (B1..B2) / `184 B` (B3)** ([shader_audit.md](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/shader_audit.md)) down by `32 B/ray` on every intersect+shade hop.
   - Across 1080p vs 4K, **`modern-hall` hits `4.06x` (`15.730` vs `3.876 ms`)** because `3,438.3 MB` of 4K queue traffic + 4 dense bounces over `168` lights overflows the `96 MB` Infinity Cache far worse than `~860 MB` @ 1080p. Yet chopping 4K into `--batches 4` STILL leaves **`696.5 MB` double-buffered (`7.3x` the 96 MB Infinity Cache!)**, so it loses **`10/10` scenes (`+0.7%` to `+28.8%`)** by starving late-bounce CUs and quadrupling `Classify` (`4x`), fills (`17`), and barriers (`96` per frame at 8 batches).

2. **Secondary-Bounce Specialization Inversion (`diffuse_sec` vs. `complex_sec`)**:
   - `wavefront_shade_diffuse_sec` succeeds as designed: stripping primary-only G-buffer/NRC setup reduces ISA instruction count by **`-32.9%` on `gfx1100` (`15,066 -> 10,107`)** (`-42.0%` on ReSTIR variant!) and trims BVH blocks `5 -> 3`.
   - `wavefront_shade_complex_sec`, however, cuts instructions by only **`-6.9%` (`14,234 -> 13,257`)** and **regresses VGPR pressure (`126 -> 129` on `gfx1100`, `124 -> 129` on `gfx1201`)** because line `314` still reads `materials[matId]` (`208 B`), all complex lobes stay compiled, and `IS_SECONDARY_BOUNCE` adds a dedicated transmissive NRC `rayQueryEXT` lookahead ([wavefront_shade_complex.comp:1284-1360](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shade_complex.comp#L1284-L1360)).
   - `wavefront_shade_dielectric` has **no `_sec` variant at all**, forcing secondary glass rays to carry bounce-0 PSR guide extraction (`baseLod = 0.0` sampling `1x1` mips!).

3. **VMA Host-Accessible Static Buffers & `QueueCountersBuffer` Atomic Contention**:
   - Five hot read-only scene tables in [SceneManager.cpp:448-515](file:///c:/Users/naoki/Development/Pathways/src/scene/SceneManager.cpp#L448-L515) (`m_materialBuffer` `208 B/mat`, `m_materialArchetypeBuffer` `4 B`, `m_shadeMaterialBuffer` `64 B`, `m_lightBuffer` `128 B`, `m_lightTreeBuffer` `64 B`) and `m_tlasInputInstancesBuffer` ([GpuTlasUpdatePipeline.cpp:72-80](file:///c:/Users/naoki/Development/Pathways/src/rt/GpuTlasUpdatePipeline.cpp#L72-L80)) combine `VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE` + `HOST_ACCESS_SEQUENTIAL_WRITE_BIT` without `ALLOW_TRANSFER_INSTEAD_BIT`. Under VMA (`vk_mem_alloc.h:4295-4330, 13848-13849`), on any non-ReBAR Windows GPU (`256 MB` BAR) or when BAR pressure exceeds the `32 MB` small-BAR block ceiling, buffers land in uncached WC BAR or **host System RAM over PCIe** — a silent footgun for `cyber-city`'s `640,835`-light (`~82 MB` + `~82 MB` tree) and `Kitchen_set`'s `135`-material tables.
   - Simultaneously, [wavefront_shared.glsl:16-44](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shared.glsl#L16-L44) wastes **`120 B` (`30` uints = `46.9%` of `256 B`)** in dead/reserved fields of `QueueCountersBuffer` while hot workgroup-retirement and counter atomics pack into the first 64 B cache line, multiplied by **`259,200` `retiredWorkgroups` global atomics/bounce @ 4K** caused by `local_size = 32` ([shader_audit.md](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/shader_audit.md)).

---

## 6. Ablation Matrix Evaluation (214 Sweep 3 Runs Across Groups A–G)

All Sweep 3 tests ran at `3840x2160`, `1 SPP`, `max-bounces 4`, `15` warmup + `35` measured frames (`CV = 0.0043–0.0361`, `0` validation errors), compared strictly against the matched `Group A auto` (`15/35`) baseline.

### 6.1 Group A — `--wavefront-sort` (`auto` vs. `none` / `archetype` / `dual`, 20 Scenes)

**Verdict**: `auto` selected the numerically fastest explicit mode on only **`4 / 20` scenes** (`mineshaft`, `DamagedHelmet`, `veach-ajar`, `PointInstancedMedCity`). While `10` of the `16` suboptimal differences fall within `< 1 std_dev` (`Kitchen_set` is a `0.000 ms` tie at `3.304 ms`; `classroom` `+0.010 ms`; `infinity-mirror` `+0.015 ms`; `BuickRiviera` `+0.023 ms`), the aggregate failure splits into two unmistakable hardware defects:

1. **Dielectric / Delta-Unroll Inversion (`none` beats `auto` by `6.2%–109.6%` on 5/5 refractive & car scenes)**:
   - Under `archetype` or `dual`, `wavefront_shade_dielectric` (`145` VGPRs = `56.25%` occ on `gfx1100`) executes `delta_unroll = 6` (`Config.hpp:124`) inline `rayQueryEXT` traversals per bounce, whereas monolithic `wavefront_shade` (`--wavefront-sort none`, `128` VGPRs = `62.5%` occ) steps once per wavefront bounce (`+2.4x–5.7x` faster shade on `glass-of-water`, trading deep internal TIR path completion for speed).
2. **`Dual` Morton Sort Penalty (`archetype` beats `dual` on `10 / 11` remaining sorted scenes, $p = 0.0059$)**:
   - [RayTracingOrchestrator.cpp:265-302](file:///c:/Users/naoki/Development/Pathways/src/rt/RayTracingOrchestrator.cpp#L265-L302) sums unweighted mesh triangle areas (ignoring instances/transforms!) and triggers `Dual` whenever divergent area exceeds **`0.1%` (`0.001f`)** — routing `17 / 20` scenes into `dual` (even `PointInstancedMedCity` with `22 + 79 + 18` dielectric hits across 8.3M rays!).
   - Inside [wavefront_intersect.comp:167](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_intersect.comp#L167), `morton3D_10bit(qPos) & 0x0FFFu` extracts the **12 LSBs** (aliasing every `15.625 cm` instead of MSBs `>> 18u`) across `30` `subgroupShuffle` instructions (`32` `ds_*` in ISA), scattering `materialIndices` writes/reads across cache lines and degrading performance by `0.010–0.188 ms` (`0.1%–1.2%`) relative to pure `archetype`.
   - Conversely, where shading divergence and Light Tree pressure are high (`modern-hall`, `many-lights`, `cyber-city`, `infinity-mirror`, `breakfast-room`, `mineshaft`, `BuickRiviera`, `Kitchen_set`), disabling sorting (`none`) slows frames by **`+10.5%` to `+44.9%`**.

| Scene | `auto` (ms, Picked) | `none` (ms) | `archetype` (ms) | `dual` (ms) | Fastest Explicit | Opt? | `auto` vs Fastest |
|---|---:|---:|---:|---:|---|---|---:|
| `cornell-box` | 5.945 (`dual`) | 5.908 | **5.869** | 5.937 | `archetype` | NO | +0.076 ms (+1.3%) |
| `many-lights` | 9.376 (`dual`) | 13.235 (+41.2%) | **9.297** | 9.377 | `archetype` | NO | +0.079 ms (+0.8%) |
| `cyber-city` | 23.456 (`dual`) | 32.021 (+36.5%) | **23.309** | 23.465 | `archetype` | NO | +0.147 ms (+0.6%) |
| `infinity-mirror` | 4.733 (`dual`) | 5.643 (+19.2%) | **4.718** | 4.742 | `archetype` | NO | +0.015 ms (+0.3%) |
| `mineshaft` | 17.411 (`archetype`) | 19.991 (+14.8%) | **17.408** | 17.443 | `archetype` | YES | +0.003 ms (+0.0%) |
| `DamagedHelmet` | **1.968** (`archetype`) | 2.103 (+6.9%) | **1.975** | 1.978 | `archetype` | YES | -0.007 ms (-0.3%) |
| `DragonAttenuation` | 10.388 (`dual`) | **6.945** (-33.1%) | 10.449 | 10.459 | `none` | NO | **+3.443 ms (+49.6%)** |
| `DragonDispersion` | 11.176 (`dual`) | **7.388** (-33.9%) | 11.099 | 11.189 | `none` | NO | **+3.788 ms (+51.3%)** |
| `bmw-m6` | 9.967 (`dual`) | **9.389** (-5.8%) | 9.875 | 9.970 | `none` | NO | **+0.578 ms (+6.2%)** |
| `breakfast-room` | 15.482 (`dual`) | 18.356 (+18.6%) | **15.404** | 15.469 | `archetype` | NO | +0.078 ms (+0.5%) |
| `classroom` | 10.570 (`dual`) | 10.677 (+1.0%) | **10.560** | 10.666 | `archetype` | NO | +0.010 ms (+0.1%) |
| `cornell-caustic` | 7.132 (`dual`) | **6.044** (-15.3%) | 7.084 | 7.144 | `none` | NO | **+1.088 ms (+18.0%)** |
| `glass-of-water` | 14.320 (`dual`) | **6.832** (-52.3%) | 14.291 | 14.326 | `none` | NO | **+7.488 ms (+109.6%)** |
| `living-room` | 11.540 (`dual`) | 11.695 (+1.3%) | **11.421** | 11.531 | `archetype` | NO | +0.119 ms (+1.0%) |
| `modern-hall` | 15.880 (`dual`) | 23.017 (+44.9%) | **15.692** | 15.910 | `archetype` | NO | +0.188 ms (+1.2%) |
| `veach-ajar` | 7.818 (`dual`) | 8.052 (+3.0%) | 7.800 | **7.799** | `dual` | YES | +0.019 ms (+0.2%) |
| `BuickRiviera` | 20.818 (`dual`) | 23.877 (+14.7%) | **20.795** | 20.820 | `archetype` | NO | +0.023 ms (+0.1%) |
| `coffee-maker` | 9.119 (`dual`) | 9.236 (+1.3%) | **9.055** | 9.127 | `archetype` | NO | +0.064 ms (+0.7%) |
| `Kitchen_set` | 3.304 (`archetype`) | 3.755 (+13.6%) | 3.313 | **3.304** | `dual` | NO | +0.000 ms (+0.0%) |
| `PointInstancedMedCity` | **9.439** (`dual`) | 10.430 (+10.5%) | 9.446 | **9.444** | `dual` | YES | -0.005 ms (-0.1%) |

### 6.2 Group B — `--sec-sort` (`direct` vs. `coherent-k8` / `directional` / `none`, 12 Scenes)

- **`direct` (`K=4`) vs. `none`**: Beats `none` on **`12 / 12` scenes** (`+1.1%` to `+4.5%` loss when disabled).
- **`coherent-k8` (`K=8`) vs. `direct`**: Improves **`9 / 12` scenes** — delivering largest gains on traversal-bound scenes (`breakfast-room -3.4%`, `PointInstancedMedCity -2.1%`, `veach-ajar -1.7%`, `living-room -1.1%`, `cornell-box -1.0%`) with near-zero worst-case drift (`+0.2%` on `cyber-city`/`classroom`, `+2.0%` on `infinity-mirror`).
- **`directional` (`secSort=1`, 8-octant DGC bins)**: **Regresses `12 / 12` scenes (`+2.9%` to `+49.2%`)** (`veach-ajar +49.2%`, `cornell-box +47.1%`, `classroom +34.2%`) because `8` indirect intersect dispatches per bounce ([WavefrontPipeline.cpp:1158-1165](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.cpp#L1158-L1165)), up to `8` global atomics per Wave32, and scattered `inGeom`/`outHits` cache misses (`12.5%–25%` 128 B line efficiency) swamp directional coherence gains.

### 6.3 Group C — `--batches` (`auto=1` vs. `1` / `2` / `4`, 10 Scenes)

- Multi-batching regresses **`10 / 10` scenes at `--batches 2` (`+0.02%` to `+9.8%`) and `--batches 4` (`+0.7%` to `+28.8%`)**.
- Why: Even at `4` batches (`~2.07M` rays/batch), double-buffered queues total **`696.5 MB` (`7.3x` the 96 MB Infinity Cache)**, while inter-batch barriers (`4x` `Classify`, `17` fills, up to `96` barriers at `8` batches, per-batch `4096 B` `m_indirectReadback` copies at [WavefrontPipeline.cpp:1307-1326](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.cpp#L1307-L1326)) add `+0.037–0.055 ms` and `4x` smaller late-bounce waves starve CUs (`breakfast-room shadow +1.296 ms`, `Kitchen_set shadow +0.404 ms`, `bmw-m6 shade +0.906 ms`, `veach-ajar shade +0.834 ms`).
- **Critical `16 GiB` RDNA 4 Heuristic Bug**: [RayTracingOrchestrator.cpp:185](file:///c:/Users/naoki/Development/Pathways/src/rt/RayTracingOrchestrator.cpp#L185) tests `vramBytes < (16ULL << 30)`, which misclassifies exact `16 GiB` cards into the `8 GiB` `2,000,000`-pixel budget (`5` batches @ 4K).

### 6.4 Group D — `--no-inline-shadows` vs. Default Inline Shadows (10 Scenes)

- Default inline shadows (`wavefront_shadow_inline.glsl`) win **`8 / 10` scenes** (`glass-of-water +7.5%`, `cornell-box +6.1%`, `classroom +2.7%`, `breakfast-room +2.3%`, `PointInstancedMedCity +2.2%`, `many-lights +1.6%`, `Kitchen_set +1.0%`, `veach-ajar +0.2%` slower when detached) by avoiding `+64 B/ray` shadow-queue read/write traffic (`PackedShadowRay` `32 B` write + read) and extra dispatch latency.
- **Confirmed Exception on Massive Bounce-0 Workloads**: On `cyber-city` (`-0.5%`, `-0.115 ms`) and `BuickRiviera` (`-0.6%`, `-0.133 ms`), detached Bounce-0 `shade + shadow` drops by **`-0.245 ms` (`cyber-city`)** and **`-0.188 ms` (`BuickRiviera`: `8.711 -> 6.991 + 1.532 ms`, plus `-0.042 ms` intersect)** because `wavefront_shadow` runs at **`54` VGPRs / `100%` occ / `7.55 KB`**, partially offsetting `+~480 MB` shadow-queue traffic (`2,567 -> 3,048 MB` on `BuickRiviera`).

### 6.5 Group E — `--tail-megakernel`, `--no-early-bounce-cutoff`, & `--delta-unroll`

- **`--tail-megakernel` (`cutoffBounce=2`, 10 scenes)**: **Best toggleable win in the entire ablation suite**, accelerating **`9 / 10` scenes by `-1.7%` to `-25.3%`** (`cornell-caustic -25.3%`, `glass-of-water -17.4%`, `classroom -15.2%`, `veach-ajar -15.2%`, `cornell-box -12.8%`, `living-room -7.8%`, `DragonDispersion -4.8%`, `DragonAttenuation -4.3%`, `coffee-maker -1.7%`). Eliminates Bounce-1 `Intersect` + Bounces `2..3` (`1.27–3.83 ms` replaced by `0.45–2.31 ms` in `wavefront_tail_megakernel`, `62.5%` occ, `38.0 KB`, `0` atomics; slashes `cornell-caustic` traffic `2,279 -> 1,790 MB`). Only `infinity-mirror` (`+1.9%`, `0.09 ms`, `tail=0.114 ms`) is noise-neutral.
- **`--no-early-bounce-cutoff` (10 scenes)**: `0.0%` to `+3.6%` slower (`infinity-mirror +3.6%`, `classroom` & `cornell-box +1.0%`), confirming Tier 1 early-bounce cutoff (`0.005f`, [Config.hpp:134-135](file:///c:/Users/naoki/Development/Pathways/src/core/Config.hpp#L134-L135)) is net beneficial.
- **`--delta-unroll` (`6` default vs `4` vs `2`, 6 specular scenes)**: [Config.hpp:124](file:///c:/Users/naoki/Development/Pathways/src/core/Config.hpp#L124) ships `delta_unroll = 6` while `--help` documents `4`. Lowering `6 -> 4 -> 2` yields **`-14.3% -> -36.2%` on `glass-of-water`** (`14.320 -> 12.268 -> 9.137 ms`), **`-9.7% -> -25.9%` on `DragonDispersion`**, **`-8.0% -> -23.3%` on `DragonAttenuation`**, **`+0.8% -> -5.3%` on `cornell-caustic`**, and **`-1.2% -> -4.1%` on `coffee-maker`** (`infinity-mirror +2.9% / +2.0%` within `+0.1 ms` due to low glass share).

### 6.6 Group F — `--macro-blas` & `--no-light-tree`

- **`--macro-blas` (4 instanced scenes)**: Skipped by [SceneGeometryPipeline.cpp:187-188, 292-352](file:///c:/Users/naoki/Development/Pathways/src/scene/SceneGeometryPipeline.cpp#L187-L352)'s `250,000`-triangle cap / dynamic-instance bailout on `PointInstancedMedCity` (`+1.0%`), `mineshaft` (`+1.6%`), and `Kitchen_set` (`+1.8%`). On `cyber-city`, it reduced TLAS instances by only `-12.1%` (`5,596 -> 4,917`) while expanding BLAS footprint **`11.0x` (`1,650.6 -> 18,214.9 KB`)** due to `272 B/tri` staging + coarse 2D `(X,Z)` grid vertical AABB overlap, **regressing runtime by `+8.54%` (`23.456 -> 25.458 ms`)** across Classify (`1.81 -> 2.17 ms`), B0 Shade (`9.25 -> 9.88 ms`), and B0 Intersect (`2.35 -> 2.68 ms`).
- **`--no-light-tree` (8 multi-light scenes)**: Measured `+0.4%` to `+1.8%` across all 8 scenes — confirmed as a **null experiment** because [Engine.cpp:2361](file:///c:/Users/naoki/Development/Pathways/src/core/Engine.cpp#L2361) evaluates `m_config.enable_light_tree || (!m_sceneData.lightTreeNodes.empty())`, and `SceneManager.cpp:498` unconditionally builds `lightTreeNodes`, keeping `UBO_FLAG_LIGHT_TREE` (`1u << 7`) permanently enabled!

### 6.7 Group G — `--accum-format`, `--restir`, & `--caustics`

| Feature / Variant | Scene | Baseline (ms) | Variant (ms) | Delta (ms / %) | Key Measured Side-Effects |
|---|---|---:|---:|---:|---|
| `--accum-format rg11` | `cornell-box`, `classroom`, `veach-ajar`, `PointInstancedMedCity` | `5.945 / 10.570 / 7.818 / 9.439` | `6.004 / 10.887 / 7.791 / 9.572` | `-0.3%` to `+3.0%` | Tonemap `-10%` (`0.130–0.143 -> 0.116–0.132 ms`); `-256 MB` VRAM on `veach-ajar` |
| `--accum-format fp32` | Same 4 scenes | Same | `6.543 / 10.952 / 8.069 / 9.910` | **`+3.2%` to `+10.1%`** | Doubles Tonemap (`0.285–0.303 ms`), **`+128 to +384 MB` VRAM** |
| `--restir` | `cornell-box`, `classroom`, `veach-ajar` | `5.945 / 10.570 / 7.818` | `8.261 / 12.939 / 9.921` | **`+22.4%` to `+39.0%`** (`+2.10..+2.37 ms`) | `+1,012.5 MiB` (`128 B/px`) buffers, `restir_di_reuse` (`117` VGPRs) + `restir_gi_reuse` (`91` VGPRs) |
| `--restir` | `many-lights` (`64` lights) | `9.376` | **`20.013`** | **`+10.637 ms (+113.5%)`** | B0+B1 shade explode (`3.72 -> 8.32 ms`, `1.91 -> 5.83 ms`), B2 shadow rays `0.83M -> 1.83M` from `7`-BVH RIS + Light Tree replay |
| `--caustics` | `cornell-caustic`, `glass-of-water` | `7.132 / 14.320` | `9.664 / 18.252` | **`+2.532 (+35.5%) / +3.932 (+27.5%)`** | Adds photon trace/splat/filter (`10.8 KB` LDS) + **`+1.0 ms` inter-stage gaps** (`0.065 -> 1.057`, `0.116 -> 1.178 ms`) |

---

## 7. Shader Microarchitecture & LLPC Register/Occupancy Audit (`gfx1100` vs. `gfx1201`)

### 7.1 Hardware Occupancy Model & Coverage Summary

- **SPIR-V Coverage (`61` `.spv` files in `build/windows-clang-release/bin/shaders/`)**: `49` active `CMakeLists.txt` targets (`48` compute + `imgui_hdr.frag`) and `12` legacy/orphaned `.spv`s (`8` compute + `4` RTP). Offline `amdllpc.exe` (`LLPC 78.0`) succeeded on **`119 / 122` target compilations**: `61 / 61` on `gfx1201` (RDNA 4) and `58 / 61` on `gfx1100` (RDNA 3), with `3` expected `gfx1100` intrinsic errors on FP8 WMMA (`upways_denoise_prep`, `upways_denoiser_wmma`, `upways_upscaler_wmma` — `.int8` and `.fp16` compile cleanly on both).
- **Verified RDNA 3 / RDNA 4 Wave32 Residency (`shae.exe --arch-info -w 32`)**:
  - **VGPR Bank**: `1,536` physical VGPRs per SIMD32 (`2` SIMDs/CU in `wgp_mode=false`), allocated in **`24`-VGPR granules** (`alloc = min(256, ceil(vgpr / 24) * 24)`). Wave32 occupancy cliffs fall at: **`96` (`16/16` = `100%`), `120` (`12/16` = `75%`), `144` (`10/16` = `62.5%`), `168` (`9/16` = `56.25%`), `192` (`8/16` = `50%`), `216` (`7/16` = `43.75%`), `240` (`6/16` = `37.5%`)**. *(Corrects `scripts/extract_shader_stats.py:179-181`, which assumed 512 VGPRs / 8-VGPR granularity.)*
  - **SGPR & LDS Ceilings**: Hard limit of `106` numbered SGPRs (`.sgpr_count = 107` in PAL metadata includes `2` VCC regs) — **13 active wavefront/ReSTIR/Upways kernels saturate `107` SGPRs**. LDS is **`64 KiB` / CU** (`2` SIMDs share LDS: `<= 2,048 B`/WG at max waves keeps `100%` unless >`2,048 B`, e.g., `16,384 B` limits `8/16 = 50%`, `20,480 B` limits `6/16 = 37.5%`, `26,112–29,184 B` limits `4/16 = 25%`).
  - **Scratch Memory**: **`0 B` scratch and `0` register spills across ALL 56 compute shaders**.

### 7.2 Complete Active Shader Matrix (`48` Compute Variants + `1` Fragment, `gfx1100` / `gfx1201`)

| Shader | Used VGPR (`1100`/`1201`) | Alloc VGPR (`1100`/`1201`) | SGPR (`1100`/`1201`) | LDS (B) | Wave32 Occ (`1100`/`1201`) | Limiter | `.text` Bytes (`1100`/`1201`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `wavefront_classify` | 62 / 62 | 72 / 72 | 107 / 106 | 2,048 | 16 (100%) / 16 (100%) | — | 25,600 / 28,032 |
| `wavefront_intersect` | 56 / 57 | 72 / 72 | 59 / 62 | 2,048 | 16 (100%) / 16 (100%) | — | 9,856 / 10,880 |
| `wavefront_intersect_alpha` | 60 / 59 | 72 / 72 | 80 / 90 | 2,048 | 16 (100%) / 16 (100%) | — | 12,160 / 13,056 |
| `wavefront_shadow` | 54 / 55 | 72 / 72 | 77 / 78 | 2,048 | 16 (100%) / 16 (100%) | — | 7,552 / 7,552 |
| `wavefront_shade` | 128 / 129 | 144 / 144 | 107 / 107 | 2,048 | 10 (62.5%) / 10 (62.5%) | VGPR | **43,264 / 45,952** |
| `wavefront_shade_diffuse` | 118 / 111 | 120 / 120 | 107 / 107 | 2,048 | 12 (75.0%) / 12 (75.0%) | VGPR | **81,408 / 85,760** |
| `wavefront_shade_diffuse_sec` | 112 / 111 | 120 / 120 | 107 / 107 | 2,048 | 12 (75.0%) / 12 (75.0%) | VGPR | **53,376 / 57,728** |
| `wavefront_shade_dielectric` | **145 / 143** | **168 / 144** | 107 / 107 | 2,048 | **9 (56.25%) / 10 (62.5%)** | VGPR | **58,752 / 63,360** |
| `wavefront_shade_complex` | 126 / 124 | 144 / 144 | 107 / 107 | 2,048 | 10 (62.5%) / 10 (62.5%) | VGPR | **75,520 / 80,768** |
| `wavefront_shade_complex_sec` | **129 / 129** | 144 / 144 | 107 / 107 | 2,048 | 10 (62.5%) / 10 (62.5%) | VGPR | **70,784 / 75,264** |
| `wavefront_shade_emissive` | 78 / 77 | 96 / 96 | 107 / 107 | 0 | 16 (100%) / 16 (100%) | — | 17,280 / 18,944 |
| `wavefront_tail_megakernel` | 144 / 141 | 144 / 144 | 107 / 107 | 2,048 | 10 (62.5%) / 10 (62.5%) | VGPR | **38,016 / 40,192** |
| `wavefront_shade_diffuse_restir` | 120 / 120 | 120 / 120 | 107 / 107 | 2,048 | 12 (75.0%) / 12 (75.0%) | VGPR | **99,712 / 105,600** |
| `wavefront_shade_diffuse_sec_restir` | 113 / 112 | 120 / 120 | 107 / 107 | 2,048 | 12 (75.0%) / 12 (75.0%) | VGPR | **56,960 / 61,184** |
| `wavefront_shade_complex_restir` | **157 / 154** | 168 / 168 | 107 / 107 | 2,048 | **9 (56.25%) / 9 (56.25%)** | VGPR | **93,824 / 99,584** |
| `wavefront_shade_complex_sec_restir` | 130 / 130 | 144 / 144 | 107 / 107 | 2,048 | 10 (62.5%) / 10 (62.5%) | VGPR | **74,496 / 79,744** |
| `wavefront_shade_dielectric_restir` | 144 / 141 | 144 / 144 | 107 / 107 | 2,048 | 10 (62.5%) / 10 (62.5%) | VGPR | **60,544 / 65,024** |
| `wavefront_shade_emissive_restir` | 78 / 77 | 96 / 96 | 107 / 107 | 0 | 16 (100%) / 16 (100%) | — | 18,560 / 20,352 |
| `restir_di_reuse` | 117 / 116 | 120 / 120 | 107 / 107 | 0 | 12 (75.0%) / 12 (75.0%) | VGPR | 17,536 / 18,816 |
| `restir_gi_reuse` | 91 / 94 | 96 / 96 | 71 / 74 | 0 | 16 (100%) / 16 (100%) | — | 13,824 / 14,592 |
| `restir_debug_view` | 10 / 10 | 24 / 24 | 24 / 24 | 0 | 16 (100%) / 16 (100%) | — | 3,200 / 3,328 |
| `nrc_encode_infer` | 93 / 80 | 96 / 96 | 50 / 50 | **26,112** | **4 (25.0%) / 4 (25.0%)** | **LDS** | 13,440 / 13,952 |
| `nrc_train` | **240 / 200** | 240 / 216 | 72 / 81 | **29,184** | **4 (25.0%) / 4 (25.0%)** | **LDS** | **39,808 / 47,872** |
| `nrc_resolve` | 31 / 31 | 48 / 48 | 22 / 22 | 0 | 16 (100%) / 16 (100%) | — | 2,432 / 2,560 |
| `caustic_photon_trace` | 75 / 74 | 96 / 96 | 57 / 66 | 2,048 | 16 (100%) / 16 (100%) | — | 7,808 / 8,192 |
| `caustic_splat` | 21 / 21 | 24 / 24 | 42 / 42 | 0 | 16 (100%) / 16 (100%) | — | 2,304 / 2,304 |
| `caustic_filter` | 33 / 32 | 48 / 48 | 28 / 28 | 10,752 | 16 (100%) / 16 (100%) | — | 6,784 / 7,040 |
| `accum_tonemap_fused` | 14 / 14 | 24 / 24 | 24 / 25 | 0 | 16 (100%) / 16 (100%) | — | 4,352 / 4,608 |
| `accum_running_avg` | 12 / 12 | 24 / 24 | 26 / 26 | 0 | 16 (100%) / 16 (100%) | — | 2,560 / 2,816 |
| `accum_merge` | 18 / 18 | 24 / 24 | 58 / 54 | 0 | 16 (100%) / 16 (100%) | — | 3,712 / 4,224 |
| `tonemap_aces` | 14 / 14 | 24 / 24 | 30 / 23 | 0 | 16 (100%) / 16 (100%) | — | 2,816 / 2,944 |
| `update_tlas_instances` | 27 / 27 | 48 / 48 | 20 / 8 | 0 | 16 (100%) / 16 (100%) | — | 768 / 768 |
| `ffx_sdk_prep` | 35 / 33 | 48 / 48 | 46 / 46 | 0 | 16 (100%) / 16 (100%) | — | 10,496 / 11,264 |
| `fsr3_blend` | 9 / 9 | 24 / 24 | 26 / 26 | 0 | 16 (100%) / 16 (100%) | — | 640 / 640 |
| `ffx_rr_prep` | 34 / 36 | 48 / 48 | 86 / 86 | 0 | 16 (100%) / 16 (100%) | — | 2,176 / 2,304 |
| `ffx_rr_compose` | 44 / 44 | 48 / 48 | 70 / 70 | 0 | 16 (100%) / 16 (100%) | — | 1,152 / 1,280 |
| `upways_denoise_prep` (.fp8) | FAIL / 94 | — / 96 | — / 66 | — / 7,680 | — / 16 (100%) | — | — / 15,488 |
| `upways_denoise_prep.int8` | 96 / 91 | 96 / 96 | 62 / 66 | 7,680 | 16 (100%) / 16 (100%) | — | 15,616 / 16,256 |
| `upways_denoise_prep.fp16` | 94 / 92 | 96 / 96 | 62 / 66 | 7,680 | 16 (100%) / 16 (100%) | — | 14,976 / 15,744 |
| `upways_denoiser_wmma` (.fp8) | FAIL / 70 | — / 72 | — / 66 | — / 16,384 | — / 8 (50.0%) | LDS | — / 4,096 |
| `upways_denoiser_wmma.int8` | 96 / 68 | 96 / 72 | 58 / 58 | 16,384 | 8 (50.0%) / 8 (50.0%) | LDS | 4,736 / 4,864 |
| `upways_denoiser_wmma.fp16` | **160 / 97** | 168 / 120 | 58 / 66 | **28,672** | **4 (25.0%) / 4 (25.0%)** | **LDS** | 3,712 / 3,840 |
| `upways_upscaler_wmma` (.fp8) | FAIL / 124 | — / 144 | — / 106 | — / 16,384 | — / 8 (50.0%) | LDS | — / 10,880 |
| `upways_upscaler_wmma.int8` | 133 / 125 | 144 / 144 | 106 / 106 | 16,384 | 8 (50.0%) / 8 (50.0%) | LDS | 13,184 / 13,312 |
| `upways_upscaler_wmma.fp16` | 151 / 132 | 168 / 144 | 106 / 106 | **20,480** | **6 (37.5%) / 6 (37.5%)** | **LDS** | 8,576 / 9,344 |
| `upways_upscale_resolve` (.fp8/.int8/.fp16 — 3 `.spv`s) | 82 / 78 | 96 / 96 | 46 / 50 | 0 | 16 (100%) / 16 (100%) | — | 13,056 / 13,696 |
| `imgui_hdr.frag` *(active UI frag, Wave64)* | 13 / 13 | 24 / 24 | 14 / 14 | 0 | Wave64 | — | 640 / 768 |

**Legacy / Orphaned `.spv` Inventory (`12` binaries in `bin/shaders/`)**:
- **8 Legacy Compute**: `denoise_atrous_*` (4 variants, `46–52` VGPRs, `100%`), `denoise_fused_temporal` (`135`/`121` VGPRs, `20,480 B` LDS, `62.5%`), `wavefront_shade_conductor` (`94` VGPRs, `100%`), `wavefront_shade_passthrough` (`18` VGPRs, `100%`), `wavefront_resolve` (`9` VGPRs, `100%`).
- **4 Legacy RTP**: `raytrace.rgen` (`156`/`155` VGPRs, `56.25%`, `1348`/`1447` insts), `raytrace.rchit` (`212`/`210` VGPRs, `43.75%`, `6112`/`6480` insts, `32.6`/`34.3 KB`), `raytrace.rmiss` (`32` VGPRs, `100%`), `shadow.rmiss` (`1` VGPR, `100%`) (`0 B` metadata scratch; `.rgen/.rchit/.rmiss` emit `8/122/7` `scratch_spills_insts`).

### 7.3 Instruction Mix Breakdown (`gfx1100` / `gfx1201`, Key Compute Shaders)

| Shader | Total Insts | VALU Insts | VOPD Pairs (`% VALU`) | SALU | VMEM (non-BVH) | BVH RT | SMEM | LDS (`ds_*`) | Atomics | Transc. | Wait Mem / ALU | Bar | Cond Br |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `wavefront_classify` | 4813 / 5372 | 2361 / 2172 | 248 (19.0%) / 240 (19.9%) | 933 / 1116 | 149 / 141 | **2** / 2 | 88 / 94 | 2 / 2 | 12 / 12 | 73 / 65 | 157/83 / 150/776 | 0 | 240 / 229 |
| `wavefront_intersect` | 1748 / 1911 | 911 / 884 | 108 (21.2%) / 113 (22.7%) | 293 / 293 | 75 / 73 | 1 / 1 | 10 / 10 | **32 / 32** | 10 / 10 | 21 / 17 | 76/16 / 85/222 | 0 | 70 / 67 |
| `wavefront_intersect_alpha` | 2178 / 2315 | 1084 / 1039 | 157 (25.3%) / 152 (25.5%) | 415 / 405 | 95 / 87 | **2** / 2 | 16 / 16 | **33 / 33** | 10 / 10 | 25 / 17 | 102/18 / 102/266 | 0 | 96 / 88 |
| `wavefront_shadow` | 1338 / 1333 | 556 / 531 | 92 (28.4%) / 76 (25.0%) | 360 / 315 | 45 / 36 | **2** / 2 | 22 / 22 | 3 / 2 | 1 / 1 | 21 / 13 | 81/14 / 65/131 | 0 | 72 / 65 |
| `wavefront_shade` | 8099 / 8649 | 4083 / 3989 | 542 (23.4%) / 536 (23.7%) | 1517 / 1556 | 188 / 178 | 2 / 2 | 132 / 132 | 2 / 2 | 7 / 7 | 273 / 264 | 289/193 / 297/875 | 0 | 364 / 357 |
| `wavefront_shade_diffuse` | 15066 / 15911 | 7928 / 7636 | 860 (19.6%) / 879 (20.7%) | 2719 / 2862 | 366 / 344 | **5 / 5** | 264 / 208 | 7 / 7 | 7 / 7 | 310 / 290 | 549/241 / 500/1594 | 0 | 592 / 576 |
| `wavefront_shade_diffuse_sec` | 10107 / 10902 | 4853 / 4659 | 544 (20.2%) / 575 (22.0%) | 2108 / 2220 | 304 / 291 | **3 / 3** | 166 / 196 | 5 / 5 | 7 / 7 | 248 / 236 | 385/193 / 412/1211 | 0 | 467 / 456 |
| `wavefront_shade_dielectric` | 10750 / 11525 | 5315 / 5235 | 835 (27.2%) / 801 (26.5%) | 2116 / 2143 | 388 / 369 | **4 / 4** | 230 / 228 | 5 / 5 | 5 / 5 | 232 / 216 | 440/150 / 456/1220 | 0 | 457 / 440 |
| `wavefront_shade_complex` | 14234 / 15246 | 6942 / 6782 | 811 (20.9%) / 803 (21.2%) | 2809 / 2894 | 387 / 365 | **5 / 5** | 270 / 258 | 7 / 7 | 7 / 7 | 387 / 367 | 566/298 / 573/1639 | 0 | 623 / 606 |
| `wavefront_shade_complex_sec` | 13257 / 14114 | 6564 / 6306 | 853 (23.0%) / 874 (24.4%) | 2593 / 2660 | 378 / 362 | **4 / 4** | 236 / 253 | 6 / 6 | 9 / 9 | 362 / 346 | 492/278 / 515/1470 | 0 | 573 / 561 |
| `wavefront_shade_emissive` | 3072 / 3349 | 1497 / 1441 | 190 (22.5%) / 194 (23.7%) | 552 / 611 | 153 / 152 | 0 / 0 | 75 / 75 | 0 / 0 | 1 / 1 | 73 / 73 | 140/55 / 145/359 | 0 | 145 / 145 |
| `wavefront_tail_megakernel` | 7242 / 7691 | 3727 / 3575 | 547 (25.6%) / 531 (25.9%) | 1401 / 1422 | 141 / 125 | **4 / 4** | 88 / 120 | 4 / 4 | **0 / 0** | 237 / 221 | 222/173 / 263/763 | 0 | 339 / 324 |
| `wavefront_shade_diffuse_restir` | 18668 / 19874 | 9272 / 8932 | 1029 (20.0%) / 1001 (20.2%) | 3790 / 3957 | 426 / 395 | **7 / 7** | 364 / 359 | 9 / 9 | 7 / 7 | 360 / 332 | 703/272 / 701/2107 | 0 | 778 / 756 |
| `wavefront_shade_diffuse_sec_restir` | 10823 / 11615 | 5111 / 4920 | 532 (18.9%) / 556 (20.3%) | 2282 / 2372 | 312 / 299 | 3 / 3 | 224 / 235 | 5 / 5 | 7 / 7 | 258 / 246 | 441/199 / 444/1309 | 0 | 505 / 496 |
| `wavefront_shade_complex_restir` | 17746 / 18949 | 8353 / 8053 | 973 (20.9%) / 958 (21.3%) | 3932 / 4026 | 449 / 418 | **7 / 7** | 365 / 332 | 9 / 9 | 7 / 7 | 437 / 409 | 719/333 / 671/2171 | 0 | 808 / 787 |
| `wavefront_shade_complex_sec_restir` | 14000 / 14972 | 6852 / 6648 | 866 (22.4%) / 885 (23.5%) | 2778 / 2884 | 387 / 371 | 4 / 4 | 264 / 270 | 6 / 6 | 9 / 9 | 372 / 356 | 531/289 / 538/1574 | 0 | 613 / 602 |
| `wavefront_shade_dielectric_restir` | 11110 / 11897 | 5431 / 5304 | 807 (25.9%) / 782 (25.7%) | 2217 / 2274 | 396 / 377 | 4 / 4 | 254 / 243 | 5 / 5 | 5 / 5 | 232 / 216 | 469/151 / 471/1282 | 0 | 488 / 469 |
| `wavefront_shade_emissive_restir` | 3342 / 3652 | 1554 / 1488 | 188 (21.6%) / 190 (22.7%) | 677 / 748 | 161 / 160 | 0 / 0 | 82 / 82 | 0 / 0 | 1 / 1 | 73 / 73 | 150/54 / 155/405 | 0 | 174 / 174 |
| `restir_di_reuse` | 3310 / 3573 | 1555 / 1497 | 176 (20.3%) / 187 (22.2%) | 784 / 829 | 62 / 61 | 0 / 0 | 52 / 55 | 0 / 0 | 3 / 3 | 93 / 93 | 109/69 / 122/339 | 1 | 168 / 168 |
| `restir_gi_reuse` | 2449 / 2590 | 1398 / 1367 | 171 (21.8%) / 183 (23.6%) | 359 / 383 | 44 / 43 | 0 / 0 | 23 / 25 | 0 / 0 | 3 / 3 | 64 / 63 | 44/54 / 47/206 | 0 | 93 / 93 |
| `nrc_encode_infer` | 2115 / 2004 | 913 / 907 | 57 (11.8%) / 72 (14.7%) | 149 / 165 | 154 / 151 | 0 / 0 | 11 / 14 | **491 / 383** | 3 / 3 | 19 / 19 | **241/14 / 177/86** | 0 | 31 / 31 |
| `nrc_train` | 6327 / 7076 | 2480 / 2668 | 117 (9.0%) / 190 (13.3%) | 478 / 488 | **410 / 510** | 0 / 0 | 6 / 6 | **1587 / 1616** | 10 / 10 | 51 / 57 | **788/34 / 908/293** | **20** | 113 / 116 |
| `accum_tonemap_fused` | 675 / 754 | 359 / 341 | 24 (12.5%) / 21 (11.6%) | 121 / 144 | 5 / 5 | 0 / 0 | 4 / 4 | 0 / 0 | 0 / 0 | **30 / 28** | 6/19 / 6/100 | 0 | 27 / 27 |
| `accum_running_avg` | 372 / 420 | 185 / 184 | 6 (6.3%) / 6 (6.3%) | 87 / 90 | 4 / 4 | 0 / 0 | 3 / 3 | 0 / 0 | 0 / 0 | 1 / 1 | 5/0 / 5/63 | 0 | 9 / 9 |
| `tonemap_aces` | 424 / 469 | 237 / 213 | 23 (17.7%) / 20 (17.2%) | 53 / 86 | 2 / 2 | 0 / 0 | 2 / 2 | 0 / 0 | 0 / 0 | **31 / 29** | 4/22 / 4/57 | 0 | 21 / 21 |

### 7.4 Key ISA & Cross-Generation (`gfx1100` vs. `gfx1201`) Insights

1. **L0 Instruction Cache Thrashing (`53.4–105.6 KB` vs. `32 KB` I-Cache)**: Every primary diffuse/complex shade kernel exceeds RDNA's `32 KB` L0 instruction cache by **`1.7x–3.3x`** because LLPC duplicates the full `image_bvh*_intersect_ray` traversal block **`5` times** in `diffuse`/`complex` and **`7` times** in `*_restir` (`2` from `traceShadowRayInline` in [wavefront_shadow_inline.glsl:24-81](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shadow_inline.glsl#L24-L81), `3` from `nrcEmitSelfTrainingRecord` in [nrc_record_helper.glsl:38-194](file:///c:/Users/naoki/Development/Pathways/shaders/compute/nrc_record_helper.glsl#L38-L194), `+2` from `restirDiRunInlineReuse`).
2. **Wave32 Occupancy Cliffs (`dielectric`, `complex`, `NRC`, `Upways`)**:
   - `wavefront_shade_dielectric` (`145` VGPRs on `gfx1100` -> `168` allocated) misses the `144`-VGPR boundary by **1 VGPR**, losing `10%` relative issue residency (`9/16 = 56.25%` vs `10/16 = 62.5%` on `gfx1201`'s `143` VGPRs).
   - `nrc_encode_infer` uses only `93`/`80` VGPRs yet drops to **`25.0%` occupancy (`4/16` waves)** solely from **`26,112 B` LDS** (`nrc_train` hits **`29,184 B` LDS**, `240` VGPRs, `788/908` mem waits = **`12.5%/12.8%`** stall density, `20` barriers).
   - `upways_denoiser_wmma.fp16` (`28,672 B` LDS -> `25.0%`) and `upways_upscaler_wmma.fp16` (`20,480 B` -> `37.5%`) suffer similar LDS bottlenecks vs `50.0%` in `.int8`/`.fp8`.
3. **RDNA 3 (`gfx1100`) vs. RDNA 4 (`gfx1201`) Architectural Deltas**:
   - `gfx1201` replaces `image_bvh64_intersect_ray` + coarse `s_waitcnt` + `s_barrier` with 8-wide **`image_bvh8_intersect_ray`**, decoupled wait instructions (`s_wait_loadcnt/samplecnt/bvhcnt/kmcnt/dscnt/storecnt`), `s_barrier_signal/wait`, and `4x–8x` more `s_wait_alu` instructions (~`5–12%` larger `.text`), while slashing register pressure on heavy matrix/neural kernels (`upways_denoiser_wmma.fp16`: `160 -> 97`, **`-63` VGPRs**; `nrc_train`: `240 -> 200`, **`-40` VGPRs**; `upways_denoiser_wmma.int8`: `96 -> 68`, **`-28` VGPRs**; `nrc_encode_infer`: `93 -> 80`, `-13`) and exclusively supporting FP8 WMMA.
4. **Low VOPD Dual-Issue Ratio (`6.3%–15.5%` in Neural/Post, `19%–27%` in Shade)**: Short dependent FP16/VALU chains, `v_cmp`/`v_cndmask` branching (`440–808` conditional branches in shade kernels), `216–437` transcendentals, `v_wmma_*` ops, and `ds_*` synchronization limit VOPD pairing (`27.2%` on `dielectric`, vs `55.6%` on `update_tlas_instances` and `39.0%` on `ffx_rr_compose`).

---

## 8. Fundamental Optimization Roadmap (Prioritized Engineering Recommendations)

### Tier 1 — Quick Wins & Heuristic / Bug Fixes (Immediate Deployment)

#### T1.1 — Enable `--tail-megakernel` (`cutoffBounce = 2`) by Default + Clamp `secondarySortMode`
- **Affected Scenes**: All multi-bounce scenes (`9 / 10` empirically improved; up to `-25.3%` on `cornell-caustic`, `-17.4%` on `glass-of-water`, `-15.2%` on `classroom` & `veach-ajar`, `-12.8%` on `cornell-box`).
- **Exact Targets**: [Config.hpp:122-123](file:///c:/Users/naoki/Development/Pathways/src/core/Config.hpp#L122-L123), [WavefrontPipeline.cpp:805-807, 1262-1303](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.cpp#L805-L1303), [wavefront_tail_megakernel.comp:61, 75-100](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_tail_megakernel.comp#L61-L100).
- **Root-Cause Mechanism**: Switching `enable_tail_megakernel` from `false` to `true` (when `max_bounces >= 3`, active bounces >= 2, and `--restir`/`NRC` are compatible) replaces Bounce-1 `Intersect` + Bounces `2..3` queue round-trips (`7+` barriers + `4` indirect pipelines/bounce) with a single `wavefront_tail_megakernel` launch (`62.5%` occ, `38.0 KB` `.text`, **`0` global atomics**). Simultaneously clamp `secondarySortMode` at [WavefrontPipeline.cpp:1286](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.cpp#L1286) instead of forwarding unmasked `directional` mode.
- **Expected Gain (Measured)**: **`-7.8%` to `-25.3%`** on high/mid-survival scenes; **`-1.7%` to `-4.8%`** on low-survival scenes (`infinity-mirror`-class <=3% B2 survival can gate cutoff off when `rays_left` falls below threshold).
- **Visual-Parity & Guardrails**: Verify Monte Carlo convergence (`FLI`/`MSE`/mean luminance delta `<= 1e-4` over 1k SPP vs pure wavefront); gate off automatically on `--caustics` / unsupported `--restir` paths.
- **Verification**: Re-run Group E subset + 1k-SPP convergence parity check on `cornell-caustic`, `glass-of-water`, `classroom`, `veach-ajar`, `infinity-mirror`.

#### T1.2 — Lower `--delta-unroll` Default (`6 -> 2..4`), Reclaim 2 VGPRs in `wavefront_shade_dielectric`, & Fix `baseLod=0.0` PSR Bug
- **Affected Scenes**: All refractive/specular scenes (`glass-of-water`, `DragonAttenuation`, `DragonDispersion`, `cornell-caustic`, `coffee-maker`, `bmw-m6`).
- **Exact Targets**: [Config.hpp:124](file:///c:/Users/naoki/Development/Pathways/src/core/Config.hpp#L124), [wavefront_shade_dielectric.comp:206-233, 440, 458, 463](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shade_dielectric.comp#L206-L233), [wavefront_common.glsl:22-27](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_common.glsl#L22-L27).
- **Root-Cause Mechanism**: Align `delta_unroll = 6` to **`2` (or `4` max)** to avoid 6 sequential inline `rayQueryEXT` loops inside every wavefront bounce. Reclaim **>=2 VGPRs** in `wavefront_shade_dielectric` (`145 -> <=144` on `gfx1100`, matching `gfx1201`'s `143`), specialize `IS_SECONDARY_BOUNCE=1` to strip bounce-0 PSR guide extraction (`<=120` VGPRs), and fix `sampleSceneTextureLod(..., 0.0)` ([wavefront_shade_dielectric.comp:206, 210, 216, 229, 233](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shade_dielectric.comp#L206-L233)) which adds `levels - 1` and currently samples the **1×1 coarsest mip** for PSR guides (similar `baseLod = 4.0` clamp at [nrc_record_helper.glsl:83,88](file:///c:/Users/naoki/Development/Pathways/shaders/compute/nrc_record_helper.glsl#L83-L88) & [wavefront_shade_complex.comp:786,791,1355,1360](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shade_complex.comp#L786-L791)).
- **Expected Gain**: **Measured `-14.3%` to `-36.2%`** (`glass-of-water`), **`-9.7%` to `-25.9%`** (`DragonDispersion`), **`-8.0%` to `-23.3%`** (`DragonAttenuation`) from unroll reduction alone, plus projected **`+11.1%` relative wave residency (`9/16 -> 10/16` = `56.25% -> 62.5%`, then `12/16` = `75%` on `_sec`)** from crossing VGPR cliffs on RDNA 3.
- **Visual-Parity & Guardrails**: Because lower `delta_unroll` terminates deep TIR paths earlier (`max_bounces=4` vs 6-step internal cascade), verify glass brightness & Beer's law attenuation match reference at `max_bounces=4..8` (optionally raise `max_bounces` for dielectrics or keep `delta_unroll = 2..3` when `hasDielectrics`).
- **Verification**: `amdllpc.exe` VGPR check (`<=144` primary, `<=120` `_sec`) + `glass-of-water`/`DragonAttenuation`/`DragonDispersion` frame-time & visual diff.

#### T1.3 — Fix `getEffectiveWavefrontSortMode` (`archetype` Default) & Switch Default `--sec-sort` to `coherent-k8`
- **Affected Scenes**: `16 / 20` scenes where `auto` was suboptimal (especially non-dielectric scenes `cornell-box`, `modern-hall`, `cyber-city`, `many-lights`, `breakfast-room`, `living-room`) + all 12 `--sec-sort` scenes.
- **Exact Targets**: [RayTracingOrchestrator.cpp:231-303](file:///c:/Users/naoki/Development/Pathways/src/rt/RayTracingOrchestrator.cpp#L231-L303), [Config.hpp](file:///c:/Users/naoki/Development/Pathways/src/core/Config.hpp), [wavefront_intersect.comp:158-193](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_intersect.comp#L158-L193).
- **Root-Cause Mechanism**: Replace the broken unweighted `0.1%` triangle-area `Dual` trigger with instance-weighted divergence weighting and default sorted mode to **`Archetype`** (which beat `Dual` on `10 / 11` scenes, $p=0.0059$). Fix `wavefront_intersect.comp:167` (`>> 18u` MSBs instead of `& 0x0FFFu` 12 LSBs aliasing every `15.625 cm` + `30` shuffles) before re-exposing `Dual`. Switch default `--sec-sort` from `direct` (`K=4`) to **`coherent-k8` (`K=8`)** and deprecate/remove `directional` (`+2.9%` to `+49.2%` regression on `12 / 12` scenes) alongside `m_indirectLayout`/`m_preprocessBuffer` dead DGC scaffolding ([DGCManager.cpp:35-58, 142-303](file:///c:/Users/naoki/Development/Pathways/src/rt/DGCManager.cpp#L35-L303)).
- **Expected Gain (Measured)**: **`-0.1%` to `-1.2%`** across sorted scenes from `archetype` over `dual` + **`-0.2%` to `-3.4%`** on `9 / 12` scenes from `coherent-k8` (`-3.4%` `breakfast-room`, `-2.1%` `PointInstancedMedCity`, `-1.7%` `veach-ajar`).
- **Verification**: Sweep 1+3 regression check across all 20 scenes + `headless_tests.ps1`.

#### T1.4 — Fix `Engine.cpp:2361` (`--no-light-tree`), `16 GiB` Batching Off-by-One, & `m_dynamicWavefrontBounces` Reset
- **Affected Scenes**: All multi-light scenes (`--no-light-tree`), future 16 GiB RDNA 4 deployments (`RX 9070 XT` etc.), interactive/progressive accumulation resets.
- **Exact Targets**: [Engine.cpp:2361](file:///c:/Users/naoki/Development/Pathways/src/core/Engine.cpp#L2361), [Engine.cpp:2299-2301](file:///c:/Users/naoki/Development/Pathways/src/core/Engine.cpp#L2299-L2301), [RayTracingOrchestrator.cpp:185](file:///c:/Users/naoki/Development/Pathways/src/rt/RayTracingOrchestrator.cpp#L185), [WavefrontPipeline.cpp:1514-1525](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.cpp#L1514-L1525).
- **Root-Cause Mechanism**: (1) Change `||` to `&&` at `Engine.cpp:2361` (`m_config.enable_light_tree && !m_sceneData.lightTreeNodes.empty()`) so `--no-light-tree` actually toggles `UBO_FLAG_LIGHT_TREE`. (2) Change `vramBytes < (16ULL << 30)` to `<= (16ULL << 30)` (or `< (15ULL << 30)` accounting for driver reserve) at `RayTracingOrchestrator.cpp:185` so 16 GiB RDNA 4 GPUs don't drop into `2,000,000`-px multi-batching (`+0.7%..+28.8%` slowdown). (3) Reset `m_dynamicWavefrontBounces` on `accumReset` (`Engine.cpp:2299-2301`). (4) Fix `estimatedVramTrafficMb` byte constants (`WavefrontPipeline.cpp:1514-1525`). (5) Delete 12 orphaned `.spv` binaries from `bin/shaders/`.
- **Expected Gain**: Enables valid Light Tree ablation (`cornell-box` vs `many-lights` suggests significant savings on `<=64`-light scenes if linear light sampling wins); prevents silent `+10%..+28%` multi-batching regression on 16 GiB RDNA 4 cards.
- **Verification**: Re-run Group F `--no-light-tree` across 8 scenes to obtain true Light Tree vs linear light sampling crossover data.

---

### Tier 2 — Data Layout, Memory/VMA & Kernel Restructuring (Medium Effort, High Leverage)

#### T2.1 — Eliminate `208 B` `Material` Loads in `Classify`, `Emissive`, & `Complex_sec` + Dedup Primary Texture Sampling
- **Affected Scenes**: All scenes, largest absolute gains on `PointInstancedMedCity`, `cyber-city`, `mineshaft`, `coffee-maker`, `modern-hall`, `living-room`, `breakfast-room`.
- **Exact Targets**: [wavefront_classify.comp:292, 356, 382](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_classify.comp#L292-L382), [wavefront_shade_emissive.comp:244, 249, 288](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shade_emissive.comp#L244-L288), [wavefront_shade_complex.comp:314, 317](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shade_complex.comp#L314-L317), [wavefront_shade_diffuse.comp:308](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shade_diffuse.comp#L308).
- **Root-Cause Mechanism**: `wavefront_classify` reads `208 B` `Material` up to `3x` + `144 B` `InstanceGPU` `2–3x` + samples textures then discards values before B0. `wavefront_shade_emissive` reads full `208 B` `Material` just for emission (`matId`). `wavefront_shade_complex` line 314 reads full `208 B` `Material` even under `IS_SECONDARY_BOUNCE=1` (`complex_sec` `129` VGPRs), while `diffuse_sec` & `complex_sec` omit secondary `hasSampledTex` bits forcing mip-0 fetches + double-`nonuniformEXT` `textureQueryLevels` + `textureLod` waterfalls. Switch `classify`/`emissive`/`_sec` to `64 B` `ShadeMaterial` (`3.25x` smaller!) or a `16 B` emissive table, pass precomputed mip counts in `Material`/sampler metadata to remove `textureQueryLevels` waterfalls, and hoist material reads before emission early-out.
- **Expected Gain (Projected)**: `3.25x` reduction in primary/emissive material read traffic; shrinks `complex_sec` below `120` VGPRs (**`62.5% -> 75%` occupancy**, directly benefiting `living-room` `3.0M` B1 complex rays, `coffee-maker`, `bmw-m6`, `modern-hall`).
- **Verification**: `amdllpc.exe` VGPR/SMEM/VMEM diff + Sweep 1 on `PointInstancedMedCity`, `cyber-city`, `living-room`, `coffee-maker`.

#### T2.2 — Compact `RayHit` (`32 -> 16 B`), Gate `SecondaryIndexQueue`, & Pad `QueueCountersBuffer` + `local_size = 64`
- **Affected Scenes**: Bandwidth- and survival-bound scenes (`classroom` `3.54 GB`, `modern-hall` `3.44 GB`, `veach-ajar` `3.20 GB`, `living-room` `2.77 GB`) and fixed-dispatch scenes (`DamagedHelmet`, `Kitchen_set`).
- **Exact Targets**: [WavefrontPipeline.cpp:230-277, 1340-1356](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.cpp#L230-L277), [wavefront_common.glsl:659-694, 714](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_common.glsl#L659-L694), [wavefront_shared.glsl:16-44](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shared.glsl#L16-L44).
- **Root-Cause Mechanism**: (1) `RayHit` (`32 B`) holds dead `hitData1.z` (`4 B` unused barycentrics), `15` unused bits in `hitData1.w`, and per-bounce redundant fields (`hitData1.xy` unused on B0 and non-emitters; `causticApplied` duplicated in `RayState`); move vertex emission area/normal onto `Material` / compact `16 B` `RayHit` (`inst/prim`, `packOct32` normal, `packUnorm2x16` UV, flags). (2) Gate `SecondaryIndexQueue` allocation off unless `Dual` or `Directional` sort is active (`-506.25 MiB` @ 4K). (3) Pack `QueueCountersBuffer` cleanly (removing **`120 B` (`30` uints = `46.9%`)** dead fields) with cache-line (`64 B`) separation between hot counters and retirement atomics, and raise compute `local_size` from `32` to `64` (`-50%` retirement atomics: `259,200 -> 129,600` per bounce @ 4K!).
- **Expected Gain (Projected)**: Saves **`1,012.50 MiB` (`36.4%`)** of 4K queue VRAM (`2,784.75 -> 1,772.25 MiB`), cuts every Intersect+Shade bounce transfer by **`32 B/ray`**, relieves `modern-hall` `4.06x` 4K IC spill, and cuts per-bounce `retiredWorkgroups` global atomics by `50%` (`259,200 -> 129,600` @ 4K).
- **Verification**: Check `queue_memory_footprint_mb` drops `2,784.75 -> ~1,772 MB`, verify zero validation/aliasing bugs across all 20 scenes.

#### T2.3 — Deduplicate `traceShadowRayInline` & Decouple NRC Training Trace to Bring Shade Kernels Under `32 KB` L0 I-Cache
- **Affected Scenes**: Shade-bound multi-light & complex scenes (`cyber-city`, `BuickRiviera`, `modern-hall`, `many-lights`, `breakfast-room`, `living-room`, `bmw-m6`), plus massive **`+113.5%` ReSTIR `many-lights` win**.
- **Exact Targets**: [wavefront_shadow_inline.glsl:21-81](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shadow_inline.glsl#L21-L81), [nrc_record_helper.glsl:38-194](file:///c:/Users/naoki/Development/Pathways/shaders/compute/nrc_record_helper.glsl#L38-L194), [wavefront_shadow.comp:88-145](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shadow.comp#L88-L145), [mis_common.glsl:77-101](file:///c:/Users/naoki/Development/Pathways/shaders/compute/mis_common.glsl#L77-L101), [shading_core.glsl:210-218](file:///c:/Users/naoki/Development/Pathways/shaders/compute/shading_core.glsl#L210-L218).
- **Root-Cause Mechanism**: Inline shadow uses `2` full `rayQueryProceedEXT` blocks (`opaque` vs `non-opaque`) and `nrcEmitSelfTrainingRecord` adds `3` more `rayQueryEXT` sites (even when NRC training writes `<=0.8%` of rays!), ballooning `wavefront_shade_diffuse*` to **`81.4–99.7 KB` (`5–7` BVH sites, `118–120` VGPRs)** and `complex*` to **`75.5–93.8 KB` (`126–157` VGPRs)** — `2.4x–3.3x` the **32 KB RDNA L0 I-Cache**! Collapsing `traceShadowRayInline` into a **single parameterized `rayQueryEXT` loop** (gated by `ENABLE_ALPHA_TEST` spec constant / uniform opaque flag, AND checking `pc.numLights > 0u` at `shadow_inline.glsl:21` and skipping `materialArchetypes` reads when `!transparentShadows`), plus moving rare NRC self-training tail-rays into a dedicated small pass, slashes BVH blocks from `5–7 -> 1–2`. Additionally, hoist `envEncodeBsdfPdf`/BSDF eval out of the `M`-candidate Light Tree loop in `evalStandardDirectNEEScaled` (`shading_core.glsl:210-218`) and reuse `stdBsdfEvalSplit` in `sampleStdBsdfBounce` (`bsdf_standard.glsl:295-356`).
- **Expected Gain (Projected)**: Drops `.text` below **`~32 KB`** L0 I-Cache threshold, lifts `wavefront_shade_diffuse` from `118` (`75%` occ) toward **`<=96` VGPRs (`100%` occ)** and `complex` toward **`<=120` VGPRs (`75%` occ)**, and eliminates redundant Light Tree replays under `--restir`.
- **Verification**: Verify `amdllpc.exe` `.text <= 32,768 B` & `BVH <= 2`, then benchmark `cyber-city`, `BuickRiviera`, `many-lights` (`default` + `--restir`), and `modern-hall`.

#### T2.4 — Fix VMA `HOST_VISIBLE` Static Scene Allocations & Eliminate Pipeline Barrier / Readback / Upload Wastes
- **Affected Scenes**: Non-ReBAR Windows / small-BAR GPUs (bandwidth protection), multi-batch / dynamic scenes (`mineshaft`, `infinity-mirror`, `cyber-city`, `DamagedHelmet`, `Kitchen_set`), `--caustics`.
- **Exact Targets**: [SceneManager.cpp:448-515, 523-598, 615-666](file:///c:/Users/naoki/Development/Pathways/src/scene/SceneManager.cpp#L448-L666), [GpuTlasUpdatePipeline.cpp:72-80, 253-269](file:///c:/Users/naoki/Development/Pathways/src/rt/GpuTlasUpdatePipeline.cpp#L72-L269), [VulkanContext.cpp:1354](file:///c:/Users/naoki/Development/Pathways/src/vulkan/VulkanContext.cpp#L1354), [WavefrontPipeline.hpp:344-345](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.hpp#L344-L345), [WavefrontPipeline.cpp:789-846, 960-964, 1141-1183, 1307-1326](file:///c:/Users/naoki/Development/Pathways/src/rt/WavefrontPipeline.cpp#L789-L1326), [RayTracingOrchestrator.cpp:662-705](file:///c:/Users/naoki/Development/Pathways/src/rt/RayTracingOrchestrator.cpp#L662-L705), [Engine.cpp:2587](file:///c:/Users/naoki/Development/Pathways/src/core/Engine.cpp#L2587), [update_tlas_instances.comp:37-64](file:///c:/Users/naoki/Development/Pathways/shaders/compute/update_tlas_instances.comp#L37-L64).
- **Root-Cause Mechanism**:
  1. Stage `m_materialBuffer`, `m_shadeMaterialBuffer`, `m_materialArchetypeBuffer`, `m_lightBuffer`, `m_lightTreeBuffer`, and `m_tlasInputInstancesBuffer` into **pure `DEVICE_LOCAL`** buffers via transfer staging (adding `VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT` in `VulkanContext.cpp:1354`), preventing WDDM WC-BAR / System-RAM fallback (`vk_mem_alloc.h:4295-4330`). Batch descriptor/upload `vkQueueWaitIdle` cascades (`SceneGeometryPipeline.cpp:495-546`, `SceneManager.cpp:615-666`, `AccelerationStructure.cpp:297, 396`) and dirty-track `EngineDescriptorManager` (`55-68`).
  2. Replace `VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT` in `writeTimestamp` (`WavefrontPipeline.hpp:344`) with explicit compute stages (`COMPUTE_SHADER_BIT`) so `Shadow` and `Intersect` overlap on separate shader/ALU engines; remove `COMMAND_PREPROCESS_BIT_EXT` barriers and `m_dgcStream` fills/shader writes when `ComputeExecutionSet` is unsupported; hoist `m_indirectReadback` copy outside batch loop and gate the 1×1 `centerDepthBuffer` copy (`RayTracingOrchestrator.cpp:662-705`) when DOF/AF is off; drop duplicate ping-pong descriptor bindings (`9/10` & `15/16`).
  3. Switch `mineshaft` from forced every-frame `MODE_BUILD_KHR` (`Engine.cpp:2587`) + full CPU Light Tree rebuilds (`SceneManager.cpp:523-598`) to periodic TLAS refit + partial delta upload, and honor `pc.updateMode` in `update_tlas_instances.comp`.
- **Expected Gain (Projected)**: Cuts `+0.308 ms` mineshaft host/TLAS overhead + frame CV (`0.035`), removes `Gap1/2/3` serialization between `Shadow`/`Intersect` (`0.022–0.189 ms` + async overlap, and **`~1.0 ms` `--caustics` gaps**), and eliminates PCIe bandwidth risk on non-ReBAR GPUs.
- **Verification**: Confirm `0` validation errors, inspect VMA heap placement logs (`DEVICE_LOCAL` only, no `HOST_VISIBLE` on static tables), and re-benchmark `mineshaft` (`4K` & `1080p`) + `--caustics`.

---

### Tier 3 — Architectural & Algorithmic Leapfrogs (Long-Term Engine Evolution)

#### T3.1 — True Wavefront Ray-Data Compaction & Dielectrics-Friendly Execution (SER-less Coherence)
- **Affected Scenes**: `glass-of-water`, `DragonAttenuation`, `DragonDispersion`, `cornell-caustic`, `breakfast-room`, `classroom`, `modern-hall`, `PointInstancedMedCity`.
- **Mechanism & Targets**: Replace `m_rayHit` / `m_rayState` indirect gather-scatter with compacted per-archetype SoA payloads or persistent work-stealing `Wavefront` queues at Bounce 0..1 combined with fixed `wavefront_intersect.comp` coarse 3D spatial+directional Morton hashing (`>> 18u` MSBs + ray octant bits). For `PointInstancedMedCity` (`40,001` instances) and `cyber-city`, replace 2D `(X,Z)` grid `--macro-blas` (`SceneGeometryPipeline.cpp:268-285`) with **3D `(X,Y,Z)` SAH-bounded spatial clustering** with an instanced-geometry cap separate from unique proto-BLAS caps so vertical towers stay tightly bounded while shrinking `40,001 -> ~2,000` TLAS instances.
- **Expected Gain (Projected)**: Major BVH traversal (`Intersect` `3.6–6.1 ms`) & `Classify` (`2.09 ms`) reduction on `PointInstancedMedCity`, `breakfast-room`, and `classroom`.

#### T3.2 — Non-Opaque Geometry Partitioning Optimization & Specular Caustics Optimization
- **Affected Scenes**: `living-room` (sole `alphaNO` scene), `glass-of-water` / `Dragon*` / `cornell-caustic` (`solidNO` meshes), `--caustics` & `--no-inline-shadows` dynamic dispatch.
- **Mechanism & Targets**: Separate pure glass `solidNO` (`406K` tris on `glass-of-water`) from alpha-cutout geometry without enabling `gl_RayFlagsNoOpaqueEXT` globally in `wavefront_shadow` ([wavefront_shadow.comp:88-145](file:///c:/Users/naoki/Development/Pathways/shaders/compute/wavefront_shadow.comp#L88-L145) forces opaque walls through software candidate loops whenever `transparentShadows && enableCaustics`!). For `--caustics`, move photon trace + splat + filter (`caustic_photon_trace/splat/filter`) onto overlapped compute/async work to eliminate the measured `+1.0 ms` barrier gap spike.

#### T3.3 — `NRC` (`25%` LDS Wall) & `Upways` FP16/INT8 Residency Redesign (`<= 16 KiB` LDS)
- **Affected Workloads**: Neural Radiance Cache (`nrc_encode_infer`, `nrc_train`) and Upways (`upways_denoiser_wmma.*`, `upways_upscaler_wmma.*`), relevant to RDNA 3 & RDNA 4 alike.
- **Exact Targets**: `nrc_encode_infer` (`26,112 B` LDS, `93/80` VGPRs -> `25%` occ), `nrc_train` (`29,184 B` LDS, `240/200` VGPRs -> `25%` occ, `788/908` mem waits = `12.5%` stall density, `20` barriers), `upways_denoiser_wmma.fp16` (`28,672 B` LDS -> `25%` occ, `160` VGPRs on `gfx1100`).
- **Root-Cause Mechanism**: Shrink LDS staging footprint per workgroup below `16,384 B` (smaller hidden-layer tile staging / subgroup-level matrix shuffles on Wave32) and pack `NRCQuery` (`80 B -> 64 B`, removing `16 B` pad) + `NRCTrainingRecord` (`128 B`). This immediately **doubles `nrc_encode_infer` occupancy (`25% -> 50%`)** and relieves `nrc_train`'s extreme `1,587/1,616` `ds_*` synchronization stalls. Note `.int8` Upways (`16,384 B` LDS, `50%` occ) already runs cleanly on both `gfx1100` and `gfx1201`, whereas FP8 targets RDNA 4 exclusively.

---

## 9. Appendix: Recorded Artifact Index

| Category | Artifact / Folder | Contents & Scope |
|---|---|---|
| **Holistic Synthesis** | [scene_performance_analysis_2026_11_25.md](file:///c:/Users/naoki/Development/Pathways/docs/reports/scene_performance_analysis_2026_11_25.md) | This complete 9-section performance engineering report |
| **Verification Audit** | [verification_audit.md](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/verification_audit.md) | Authoritative independent cross-check across all 4 workstreams & reconciliations |
| **Benchmark Report** | [benchmark_summary.md](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/benchmark_summary.md) | Sweep 1 (4K), Sweep 2 (1080p + scaling), Sweep 3 (Groups A–G, 214 ablations) tables |
| **Combined Telemetry JSON** | [all_scenes_telemetry.json](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/all_scenes_telemetry.json) | Full machine-readable Sweep 1 + Sweep 2 per-bounce telemetry (40 runs) |
| **Ablation Telemetry JSON** | [ablations_summary.json](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/ablations_summary.json) | Full machine-readable Sweep 3 Groups A–G per-bounce telemetry (214 runs) |
| **Benchmark Harness** | [run_scene_analysis.py](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/run_scene_analysis.py) | Reproducible serialized Sweep 1–3 orchestration & telemetry parser |
| **Raw Run Telemetry** | [raw/](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/raw/) | 40 raw `.json` + `.log` captures (20 scenes × 4K & 1080p) |
| **Ablation Run Telemetry** | [ablations/](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/ablations/) | 214 raw `.json` + `.log` ablation captures (Groups A–G) |
| **Pipeline C++ Audit** | [pipeline_audit.md](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/pipeline_audit.md) | C++/Vulkan barriers, heuristics, DGC, VMA, BLAS/TLAS, tail megakernel audit |
| **Shader GLSL Audit** | [shader_audit.md](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/shader_audit.md) | GLSL struct packing, queue traffic, fetches, texture LOD, BVH/LightTree, atomics |
| **Shader ISA Summary** | [shader_isa_summary.md](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/shader_isa_summary.md) | `gfx1100` vs `gfx1201` Wave32 VGPR/SGPR/LDS/occ/VOPD/BVH/stall tables |
| **Shader ISA Database** | [shader_isa_analysis.json](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/shader_isa_analysis.json) | Complete JSON ISA database across all 61 `.spv` binaries & both targets |
| **Raw ISA & Stats Dumps** | [isa/](file:///c:/Users/naoki/Development/Pathways/output/perf_analysis_2026_11_25/isa/) | 119 disassembled `.isa` + 122 `.stats.txt` outputs (`amdllpc.exe` + `llvm-objdump`) |
