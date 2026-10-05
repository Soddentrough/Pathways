# Pathways — Full Project Review

**Date**: October 5, 2026
**Scope**: Full codebase review (`src/`, `shaders/`, `CMakeLists.txt`, `docs/`, `scripts/`, `tests/`), git history analysis, verification of the September 27 SOTA review findings against current code, architecture/performance/UX/documentation assessment.
**Supersedes**: `docs/reports/codebase_sota_review.md` (2026-09-27) — that report's findings are re-verified in Section 3; this document is the current source of truth.

---

## 1. Executive Summary

**Pathways v1.24.0** is a real-time Vulkan 1.4 path tracer (~43k lines C++23, ~40 GLSL/SPIR-V shaders) whose core differentiator is a **GPU-autonomous wavefront path tracing pipeline** built on `VK_EXT_device_generated_commands` (DGC), with material-archetype microkernel sorting, zero-copy multi-GPU scaling, OpenUSD/glTF ingestion, FSR 3.1, and a custom WMMA neural reconstructor ("Upways").

**Overall assessment: this is a genuinely advanced, well-engineered project that is executing on its stated goals.** The architecture (decoupled wavefront microkernels, SoA ray queues, DGC indirect execution, 64B cache-line-packed geometry) is a legitimate SOTA-aligned design for RDNA3/4, the documentation is unusually thorough, and the benchmark/telemetry infrastructure is excellent. The September 27 SOTA review's critical bugs have been **largely fixed** — DGC stride, compaction race, batched BLAS, Windows multi-GPU, and the Engine monolith (decomposed into 17 subsystems on Oct 4) are all resolved.

However:

- **The documentation has drifted significantly from the code** (README/ARCHITECTURE still describe a 128-byte triangle record that is now 64 bytes, wrong CLI defaults, a removed `staging` transfer mode).
- **One critical multi-GPU architectural risk remains unaddressed**: shader-driven non-posted PCIe reads are now the *default* P2P path (the exact pattern the prior review flagged as a bandwidth collapse).
- **The NRC subsystem still does not do what its name promises** (trains on direct light only, 32 samples/frame).
- There is a meaningful tail of dead code, magic numbers, and test-harness gaps.

---

## 2. Project Goals & Alignment

From `ORIGINAL_REQUEST.md` and the README, the goals are:

| Goal | Status |
|---|---|
| Pure Vulkan 1.4+ baseline, zero backward-compat baggage | **Met** (Vulkan 1.4.341+ required; maintenance4/5/6, sync2, dynamic rendering, BDA, variable descriptor counts enabled; no legacy 1.0–1.3 paradigms in new code) |
| GPU-autonomous DGC wavefront path tracing | **Met — the project's crown jewel** (multi-slice DGC ring buffer, sequence-count addressing, material execution sets, multi-dispatch indirect fallback) |
| Multi-GPU scaling on dual RDNA4 | **Met with a caveat** (see Critical A1) |
| Neural upscaling/denoising (Upways WMMA, FSR 3.1) | **Met** (the two-pass 3D kinematic reconstructor was attempted Oct 3 and reverted — a healthy engineering signal) |
| Real-time interactive quality (dynamic governor, HDR, telemetry) | **Met** |

The one misalignment: the README claims the engine "functions on any compliant Vulkan 1.4 driver," but **DGC is a hard fatal requirement** (`VulkanContext.cpp:511, 587-592`), which per the project's own audit table covers only ~49% of desktop devices. Given the "zero backward compatibility" mandate, keeping DGC mandatory is defensible — but the docs must say so.

---

## 3. Status of the Previous SOTA Review (Sept 27) — Verified Against Current Code

| Old ID | Issue | Status |
|---|---|---|
| CRIT-01 | DGC indirect stride 12B vs 16B | ✅ **Fixed** (`DGCManager.cpp:46,79` — 16B, verified) |
| CRIT-02 | `dgc_compact.comp` workgroup race | ✅ **Fixed** (atomic retirement counter, verified) |
| CRIT-03 | P2P non-posted PCIe read collapse | ❌ **Still open — and now the default path** (see Critical A1) |
| CRIT-04 | Windows multi-GPU 100% failure | ✅ **Fixed** (`VK_KHR_external_semaphore_win32` timeline semaphores, verified) |
| CRIT-05 | Per-BLAS `vkQueueWaitIdle` | ✅ **Mostly fixed** (`buildBLASBatch` used everywhere; one `vkQueueWaitIdle` remains at `AccelerationStructure.cpp:105` but now drains once per batch) |
| CRIT-06 | NRC WG0-only Adam updates | ⚠️ **Mitigated by design** — training now dispatches exactly 1 workgroup, so the gate is harmless, but effective batch is 32 samples/frame |
| CRIT-07 | NRC target = direct light only | ❌ **Still open** (`targetRad = vec3(secDirectL)`, `wavefront_shade_diffuse.comp:872`) |
| CRIT-08 | ReSTIR reservoir leak on deferred shadows | ⚠️ **Latent** — ReSTIR DI is now opt-in (`--restir`, ≥8 lights), but `wavefront_shadow.comp` still does not bind/zero reservoirs |
| CRIT-09 | Upways 2D dispatch vs 1D shader | ❌ **Still broken** but dormant (fallback shader only; see Minor B3) |
| CRIT-10/11 | Headless harness false-failures | ⚠️ **Partially fixed** — binary auto-resolution works, but ~6 of 16 visual-regression configs are never rendered by `run_headless_tests.sh` |
| Minor: 128B→64B geometry compression | — | ✅ **Done** (`TriangleShadeGPU` is 64B with `static_assert`, oct32 normals/tangents, half2 UVs) — **but README/ARCHITECTURE still document the old 128B layout** |
| Minor: `#version 450` shaders | — | ❌ 8 shaders still on 450; `neural_reconstruct.comp` still has invalid `core` qualifier |
| Minor: dead shaders | — | ❌ `wavefront_persistent.comp` (942 ln) & `raytrace_comp.comp` (1,200 ln) still tracked |
| Minor: warning suppressions | — | ⚠️ Dangerous `-Wno-stringop-overflow` removed ✅; blanket `-Wno-unused-*` remain |
| Minor: hardcoded arch in FrameStats | — | ✅ **Fixed** (defaults now "Unknown") |
| Minor: sync PNG encoding | — | ✅ **Fixed** (`savePNGAsync` background worker) |
| Minor: dual VkInstance | — | ✅ **Fixed** (secondary device uses primary's instance) |
| Minor: NRC over-dispatch | — | ✅ **Fixed** (dispatch sized to query count) |
| Minor: push descriptors | — | ⚠️ Feature enabled (`pushDescriptor = VK_TRUE`) but no pass has migrated to it |

---

## 4. Findings

### A. Critical Bugs & Architectural Risks

#### A1. Multi-GPU P2P default is the pattern the last review flagged as a bandwidth collapse (CRIT-03, reopened by design choice)

- `Config.hpp:178`: default transfer mode is now `P2P` (auto-fallback to host zero-copy only on small-BAR ≤256 MB systems).
- Flow: secondary GPU `vkCmdCopyImageToBuffer` → its own device-local P2P BAR buffer (`MultiGpuManager.cpp:2098`) → **primary GPU's `accum_merge.comp` shader reads that remote BAR over PCIe** (non-posted reads).
- The prior review quantified this: RDNA SIMDs cannot hide ~1–2 µs PCIe round-trips; effective throughput collapses from ~28 GB/s to 2–5 GB/s, i.e. a 4K merge (63–165 MB) costs **35–80 ms** in the worst case. MGPU.md acknowledges the problem class and targets ≥1.90× scaling, and benchmarks show 1.72–1.99× — but those numbers are the reason to fix this, not to accept it: the merge pass is on the critical path of every frame in the default configuration on any ReBAR system (which is all modern RDNA4 boards).
- **Fix (recommended, not yet done):** invert to push. Primary allocates the merge-destination buffer in *its own* GDDR6 and exports the DMA-BUF; secondary imports it and issues `vkCmdCopyImageToBuffer`/`vkCmdCopyBuffer` into it (posted PCIe writes at line rate, no completion round-trips); primary merges from local VRAM at 640 GB/s. All DMA-BUF import/export plumbing already exists — this is a buffer-ownership inversion, not a rewrite. Keep P2P-read as a `--mgpu-transfer p2p` diagnostic.
- Secondary benefit: eliminates the `VK_ACCESS_2_HOST_WRITE_BIT`/transfer barrier coupling in `MultiGpuCoordinator::recordMergePass`.
- **Interim mitigation:** re-default to host zero-copy until the inversion lands.

#### A2. NRC (`--nrc`) does not implement Neural Radiance Caching — it is a direct-light cache with a 32-sample optimizer

`docs/nrc_architectural_evaluation.md` and `docs/NRC.md` (expert review) say this plainly, and the code confirms both fatal points remain:

- `wavefront_shade_diffuse.comp:872`: `targetRad = vec3(secDirectL)` — the network is supervised on one bounce of direct lighting only. No path-tail back-propagation (NRC.md Option 2: carry `trainRecordIdx` + throughput in the ray payload, atomic-add emitted radiance into the record at downstream shading events — the existing pass boundaries already provide the sync).
- `NRCManager.cpp:575`: training dispatch is hard-coded to **1 workgroup = 32 samples/frame**. The expert review's B5 hybrid topology (LDS reduction → fp32 global atomic per weight per workgroup → tiny Adam kernel, fp32 master weights) is specified in the docs but unimplemented.
- Also still present: 80-byte uncoalesced query records (B4 recommends 32B), and the multi-SPP queue overflow (documented, unfixed).

**Recommendation:** NRC is correctly disabled by default, but as it stands `--nrc` is a research stub that *regresses* frame time (own eval: +2.22 ms at 4K). Either (a) implement the NRC.md Option-1/Option-2 fix + B5 training topology + 32B records, or (b) relabel it in CLI help as "experimental direct-light cache" so the label matches the function. Budget (a) as a 2–3 week focused effort; it is the highest-value research feature on the roadmap.

#### A3. `upways_reconstruct.comp` is a live landmine: 2D dispatch vs 1D shader indexing (CRIT-09)

`UpwaysPipeline.cpp:594-596` dispatches `(groupsX, groupsY, 1)` but the shader computes `pixelBase = gl_WorkGroupID.x * TILE_M`, ignoring `gl_WorkGroupID.y`. At 1080p only the first 1,920 pixels render; the rest is black. It is dormant today because `SuperResolutionManager.cpp:74-77` prefers `neural_reconstruct.comp.spv` (always built), but any environment where that file is missing silently produces a 99.9%-black image instead of an error. **Fix the indexing or delete the file** (566 lines of superseded code; `neural_reconstruct.comp` replaced it).

#### A4. DGC hard-requirement contradicts documented compatibility

`VulkanContext.cpp:587-592` fatals without DGC; README says "functions on any compliant Vulkan 1.4 driver" and the Requirements section lists "any Vulkan 1.4 compliant GPU supporting ray query and DGC extensions" (the latter is honest, the former is not). Fix the README sentence. (If the RTP-only path should ever run without DGC, gate the fatal on `pipeline_type == Wavefront`.)

---

### B. Minor Bugs, Code Smells & Technical Debt

1. **Stale shader versions:** 8 shaders still `#version 450` (`accum_merge`, `accum_running_avg`, `accum_tonemap_fused`, `dgc_compact`, `fsr3_blend/rcas/upscale`, `tonemap_aces`); `neural_reconstruct.comp:1` is `#version 460 core` — `core` is an OpenGL profile qualifier that is invalid in pure-Vulkan GLSL (glslc tolerates it, but it violates the project's own "pure Vulkan" standard). One-line fixes each.
2. **Orphaned tracked shaders:** `shaders/compute/wavefront_persistent.comp` (942 lines) and `shaders/compute/raytrace_comp.comp` (1,200 lines) are in git, compiled by nothing, referenced by nothing. Notably, `wavefront_persistent.comp` is the skeleton of the "persistent wavefront scheduler" feature (prior review FEAT-05) that was never integrated — decide: integrate or delete.
3. **`upways_reconstruct.comp` 2D/1D mismatch** (see A3).
4. **Dead ReSTIR PT math:** `restir_common.glsl:141,166,181` (`evalUnshadowedTargetGI`, `evalGIJacobian`, `validateReconnectionFootprint`) are unreferenced. Either wire them into a ReSTIR PT pass or delete.
5. **Fragile push-constant bit-packing:** `accum_merge.comp:96,112` overloads `pc.pad` to carry `secPitch` (low 31 bits) and a `mergeGbuffers` flag (bit 31). Declare explicit members — this encoding breaks silently on the next push-constant change.
6. **Magic numbers:** `EPSILON 0.0005` (`wavefront_common.glsl:20`) is a fixed world-space ray offset — light leak on small geometry, acne on >1 km scenes. Use an adaptive normal offset scaled by scene extent or hit distance. Also `1000003u` frame-index decorrelation (`MultiGpuCoordinator.cpp:186`), the `256` counter-clear sizes, and `maxChunkSize = 64 MB` staging are unexplained constants.
7. **Unbounded waits:** `UINT64_MAX` in `PresentationManager.cpp:122`, `Swapchain.cpp:311`, `MultiGpuManager.cpp:1811`. A wedged GPU (or minimized window on some drivers) hangs the process forever. Use a 5–10 s bound + diagnostic dump (pairs naturally with the planned `VK_KHR_device_fault` work in TODO.md).
8. **Warning suppressions:** `-Wno-unused-parameter -Wno-unused-variable -Wno-missing-field-initializers` (CMakeLists:32-34) and Clang's `-Wno-unused-private-field` are blanket. The C++23 codebase can carry `[[maybe_unused]]` and designated initializers; the missing-field-initializers suppression has real bug-hiding potential (a missed struct member in a 40-entry push-constant array is exactly how you get a silent zero). The scoped `-Wno-array-bounds` (ImageDumper/tinyexr) and `-Wno-deprecated` (UsdLoader) are acceptable but should be narrowed.
9. **No LTO / no math-errno flags:** `CMAKE_INTERPROCEDURAL_OPTIMIZATION` is unset; `-fno-math-errno -fno-trapping-math` absent. The CPU side (scene loading, telemetry, governance) is small but these are free.
10. **CLI parser:** `Config.cpp` is still a 1,553-line if/else ladder with 89 branches; unknown arguments only `Logger::warn` (line 1491) and execution continues — a CI typo like `--scaler fsr3 quality` vs `--scaler fsr quality` silently renders the wrong thing. Exit non-zero on unknown args (behind a `--strict-args` if needed), and consider a table-driven parser.
11. **CMake machine-specific hardcoding (OpenUSD section):** `/usr/lib64/libtbb.so.2`, `/usr/lib/python3.14/site-packages/materialx`, `C:/Users/naoki/Development/USD`, `rav1e|svt-av1|libvmaf` regex filters. This works on the author's box and is fragile everywhere else — including CI (which builds without USD). Move to proper `find_package` hints + an `USD_ROOT` cache var, and drop the per-target `IMPORTED_LOCATION` surgery.
12. **Descriptor inflation:** 11 `VkDescriptorPool`s and `EngineDescriptorManager.cpp:237-255` still writes **512** combined-image descriptors (padding with `m_dummyWhite`) on every scene update, even though `descriptorBindingVariableDescriptorCount` is enabled and the shader uses `nonuniformEXT` indexing. Write only `m_sceneTextures.size()` entries.
13. **`Light` struct is 96 bytes** (`wavefront_common.glsl:149-156`) — 1.5 cache lines per pair; every other light fetch straddles a 128B boundary. Pad to 128B (a `padding` w-component can absorb it) or repack to 64B.
14. **NRC buffers** still `VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT` with `PREFER_DEVICE` (`NRCManager.cpp:65-85`) — mixed-usage allocations that force slower memory types on some drivers; nothing on the CPU reads them.
15. **Test-harness machine residue:** `run_headless_tests.sh` probes `/opt/rocm/core-10.0/bin/amd-smi` and `/home/naoki/.local/bin/amd-smi`, and comments say "limiting threads for Threadripper 3750X" while the dev host is now Strix Halo. Parameterize (`-j$(nproc)`, `command -v amd-smi`).
16. **Visual regression coverage gap (CRIT-11 remainder):** 16 configs, but `run_headless_tests.sh` renders only ~10 of them. `infinity_1spp_raw`, `upways_infinity_1080p`, `upways_2x_infinity_1080p`, `cornell_upways_1080p`, `classroom_4k_*` will "file not found" → `--strict` exit 1 on a clean checkout. Pass `--render` (the flag exists and now works with `--binary`) or add the missing renders.
17. **Windows test parity:** `run_headless_tests.ps1` (167 lines) still omits CTest scene switching, the mgpu suites, and all visual regression. The CI commit `c2e0a51` skips GPU perf/visual suites in CI entirely, so the *only* place these run is the local Linux box — the Windows regression surface is effectively untested.
18. **Root-directory clutter:** 26 `pathways_telemetry_*.json` files, a 3.4 MB `user_repro_upways_sr.png`, `test_stats.json` — all gitignored, but they are artifacts that belong in `output/` (already gitignored) or should be deleted.

---

### C. Performance Optimizations

1. **Push-DMA multi-GPU merge (A1)** — the single highest-value perf item; unblocks full 1.9×+ on ReBAR systems and removes the merge from the latency-critical read path.
2. **Dynamic work-stealing tile assignment.** Checkerboard parity is static 50/50 (`MultiGpuCoordinator::planAndLaunchSecondary`). Path-tracing cost varies ~50× between sky and caustic pixels. A shared atomic tile counter (in the host-zero-copy buffer, which both GPUs already map) letting each GPU pull the next 64×64 tile would convert load imbalance into free scaling — MGPU.md names this as the goal architecture; it is unimplemented.
3. **Descriptor buffers (`VK_EXT_descriptor_buffer`).** 11 pools, 26+ `vkUpdateDescriptorSets` calls per frame, and even/odd descriptor-set ping-pong (`WavefrontPipeline.cpp:820`). RADV supports DB; migrating the wavefront set (36 bindings, 24 of them storage buffers) to a device-written descriptor buffer eliminates the CPU update cost *and* the ping-pong, and is the right RDNA5 posture (prior review FEAT-04; missing from TODO.md — add it).
4. **Push descriptors for post-processing.** `features14.pushDescriptor` is enabled but unused. `tonemap_aces`, `fsr3_*`, `accum_*` are single-image/buffer passes — trivial `vkCmdPushDescriptorSet` candidates that delete 3–4 pools.
5. **`Light` repack (B13)** — ~12–18% fewer cache-line fetches in every NEE loop; cheap win.
6. **UMA (Strix Halo) batch sizing is now correct** (207,360 px, `RayTracingOrchestrator::getTargetBatchPixels`) — good. Verify the discrete tiers (1.5M/2M/8.29M by VRAM) against SPM data; the README still claims "2.0M on APU, 1.0M on discrete," which matches neither the code nor the 74.3%-VRAM-reduction report.
7. **Ray payload is 64B** (4×vec4). `throughput` and `radiance` are the two candidates for FP16 packing (FP16 accumulation and R11G11B10 transfer already exist); that would take the hot SoA queues to 48B/ray — the UMA isolation study measured exactly this. Worth an A/B on gfx1151 where the bus is the bottleneck.
8. **NRC 32B records + wave-cooperative compaction** (NRC.md B4) — only matters if NRC is revived, but it is specified and should be done as part of A2.
9. **Adaptive epsilon (B6)** — correctness-adjacent perf (fewer fireflies → less clamping → fewer wasted samples).
10. **Profiling:** excellent per-bounce GPU timestamps, SPM scripts, RGP/RADV integration. The one gap: no continuous CPU-side frame-pipeline breakdown (command recording time vs submit vs wait). A `PATHWAYS_PROFILE_CPU=1` mode with scoped timers would close the loop.

---

### D. Nice-to-Have Features & SOTA Alignment

1. **ReSTIR PT (full path resampling).** ReSTIR *DI* exists (opt-in, working) and the dead GI/Jacobian functions from the PT design are already in `restir_common.glsl`. This is the single biggest noise-reduction win available for the 1 SPP real-time target (the Veach Ajar benchmark — 26.35 ms pure MC at 4K — is exactly the scene ReSTIR PT was built for). The prior review's FEAT-01 spec is still valid; promote it from "dead code" to a tracked milestone.
2. **Opacity micromaps (`VK_EXT_opacity_micromap`).** The glTF/USD loaders support alpha MASK/BLEND; alpha-tested geometry currently costs shader ALU + RT interruptions. RADV supports OMM; this is a clean, bounded project (bake 2/4-state micromaps from cutout textures at load). Missing from TODO.md — add it.
3. **`VK_KHR_shader_abort` + `VK_KHR_device_fault`** (already in TODO.md §4) — genuinely important *because* of DGC: autonomous GPU dispatch makes hangs/OOB queue indices much harder to diagnose. Keep this priority.
4. **Persistent wavefront scheduler (FEAT-05).** `wavefront_persistent.comp` exists as a 942-line orphan — the design work is half done. A resident workgroup pool with GPU work-stealing across stage queues would eliminate the ~0.45 ms fixed barrier floor (README "Fixed Pipeline Barrier Floor" note) and is the correct RDNA5 posture. Prototype against the tail-megakernel already shipped.
5. **Subsurface scattering** (TODO.md §7) and **Box3D physics** (TODO.md §8) are well-scoped; no changes suggested. For SSS, Approach C (Burley dipole + detached shadow queue) is the right first phase — it reuses the existing queue machinery with zero new BVH machinery.
6. **Descriptor buffers** (see C3) — also belongs here as RDNA5 forward-readiness.
7. **Frame generation** — FSR 3's FG is absent (FSR 3.1 upscaling + RCAS shipped). Given the temporal infrastructure (motion vectors, accumulation), FG is a natural extension, but rank it below ReSTIR PT and OMM for a path tracer.
8. **`vkCmdDispatchIndirect2KHR`** — the TODO.md assessment (byte-identical PM4, negligible savings) is correct; no action.

---

## 5. Architecture & Code Quality Observations

- **The Engine decomposition (commit `ff7daa9`, Oct 4) is a clear win.** Engine.cpp went 9,559 → 2,666 lines; the new subsystems (`RayTracingOrchestrator`, `MultiGpuCoordinator`, `PresentationManager`, `SuperResolutionManager`, …) have clean seams, and the `friend class` + observing-pointer pattern is a pragmatic (if slightly unorthodox) way to avoid a full dependency-injection rewrite. CTest passes 20/20 locally.
- **Watch for the refactor's regression risk window:** the decomposition landed the day before this review. The visual-regression suite is the safety net — make sure it runs green on a clean checkout before the next feature lands (see B16).
- **Vulkan feature adoption is strong:** maintenance4/5/6, sync2, dynamic rendering + local read, BDA, variable/partially-bound descriptors, timeline semaphores, subgroup size control (Wave32 enforced in compute, native in RT), cooperative matrices. Gaps: push descriptors (enabled, unused), descriptor buffers (not enabled), `hostImageCopy` (not enabled — relevant to the UMA zero-copy upload path), `VK_EXT_shader_atomic_float` (needed for the NRC B5 fix).
- **Data layout discipline is good and improving:** 64B `TriangleShadeGPU` (2 per 128B line), 64B `ShadeMaterialGPU`, 4B archetype scalars, oct32/half2 packing, BDA queues. Remaining stragglers: 96B `Light`, 80B NRC records, 64B ray payload (FP16 opportunity).
- **Third-party stack is current:** ImGui 1.93.0 WIP, GLM 1.1.0, VMA 3.4.0, SDL3, cgltf, stb, tinyexr. No legacy baggage.
- **The `scratch/`, `build/`, root telemetry JSONs, and `.agents/` are all properly gitignored** — repo hygiene is fine.

## 6. Documentation Consistency (labels vs function)

The weakest area. Specific mismatches found:

| Location | Docs say | Code says |
|---|---|---|
| README:103, ARCHITECTURE.md:227-240 | `TriangleShadeGPU` = 128B (7×vec4) | 64B, oct32/half2 packed (`static_assert` in `ProceduralScene.hpp:80`) |
| README CLI table | `--mgpu-transfer` default `host`; `staging` valid | default `P2P` w/ auto-fallback; `staging` **throws** (`Config.cpp:671`) |
| README CLI table | `--wavefront-sort` default `dual` | default `Auto` (`Config.hpp:70`) |
| README CLI table | `--sec-sort` default `none` | default `DirectCoherent` (Xiang 2023) (`Config.hpp:75`) |
| README §1 | "2.0M pixels on APU/UMA, 1.0M on discrete" | 207,360 APU; 1.5M/2M/8.29M discrete by VRAM (`RayTracingOrchestrator.cpp:157-176`) |
| README Requirements | "functions on any compliant Vulkan 1.4 driver" | DGC is fatal-required (`VulkanContext.cpp:587`) |
| README §1 | DGC Execution Sets "fully supported via --dgc-execset" | experimental; RADV fails pipeline switching (documented in `--help` text itself) |
| CLI help | `--nrc` "Neural Radiance Caching" | direct-light-only cache, 32-sample optimizer (per own eval doc) |
| `docs/README.md` index | lists `strix_halo_*`, `uma_*` reports | ✅ exist — good |

**Recommendation:** a single doc-sync pass (~half a day) against the current CLI defaults and struct sizes. The benchmark tables in the README are the project's marketing surface; stale architecture claims next to fresh numbers undermine the credible ones.

## 7. Test Suite Assessment

- **20 CTest targets, all passing locally in <2 s.** A meaningful subset is *real* GPU work now (`DgcCompact`, `BlasBatch`, `P2PDirectBar`, `CrossGpuSync` create devices and dispatch) — a big improvement over the old "sizeof-only mocks."
- **Gaps:** no test framework (each test is a hand-rolled `main()` with `std::exit(1)` asserts — fine at this scale, but Catch2 would cut boilerplate and enable name-based triage); no unit tests for the *new* subsystems from the Oct 4 refactor (`MultiGpuCoordinator` planning math, `QualityGovernor` EMA, `TelemetryReporter` JSON schema); the governor and the 2D batch aspect-ratio solver (pure functions, trivially testable) have no direct coverage.
- **The headless matrix (10 suites) is the real quality gate** and it is good — frame luminance/blown-pixel checks, scaling ≥1.65× verification, visual regression with PSNR/MAE, a 21-scene perf matrix. Fix the coverage gap (B16) and Windows parity (B17) and this becomes a genuine CI story. Note CI currently skips all GPU-dependent suites (`c2e0a51`) — acceptable given no GPU runners, but it means the only enforcement point is local discipline.

## 8. UX & Aesthetics

- **Dear ImGui 1.93 WIP with a custom dark-glass theme** (translucent 0.76-alpha panels, rounded corners, cyan/amber/green status colors), **scalable vector font with 2× oversampling for HiDPI**, gamepad + keyboard nav, FPS "hero card" with color-coded frame budget, live histograms, and a full control panel. This is a *modern, functional* dev-tool HUD — not clunky. It is the right framework choice for an engine viewport (native, zero-latency, Vulkan-backed).
- Camera controls (FPS fly, orbit, gamepad with analog curves, cinematic paths) are the deepest UX investment in the codebase and are well-tested.
- Quick wins: enable `ImGuiConfigFlags_NavEnableSetMousePos` for gamepad-driven sliders; persist panel layout (currently `io.IniFilename = nullptr` by design, but a `--ui-persist` opt-in would help); the 60-frame (0.5 s) latency history is short for spotting 1% lows — 300 frames is cheap.

---

## 9. Prioritized Action Plan

**Quick wins (days):**

1. Invert MGPU P2P to push-DMA posted writes (A1) — or at minimum re-default to host zero-copy until it is done.
2. Delete or fix `upways_reconstruct.comp` (A3); delete `wavefront_persistent.comp`/`raytrace_comp.comp` or promote to a tracked experiment (B2).
3. Shader version cleanup: 8× `#version 450` → `460`, drop `core` (B1).
4. Doc-sync pass for the §6 table.
5. `--render` (or explicit renders) in `run_headless_tests.sh` for the 6 missing visual-regression configs (B16); parameterize amd-smi paths / `-j` (B15).
6. Bounded timeouts replacing `UINT64_MAX` (B7); explicit `secPitch` push-constant member (B5).
7. Variable-descriptor-count for scene textures (B12); `Light` → 128B (B13).

**Medium term (weeks):**

8. NRC revival per NRC.md: Option-2 plumbing + Option-1 termination, B5 training topology, 32B records — or relabel the feature honestly (A2).
9. ReSTIR PT integration using the existing dead GI functions (D1).
10. Descriptor buffer migration for the wavefront set + push descriptors for post (C3/C4).
11. Opacity micromaps for alpha-tested geometry (D2).
12. Windows headless-harness parity incl. visual regression (B17); add unit tests for `MultiGpuCoordinator`/`QualityGovernor`/batch solver.
13. Dynamic work-stealing tile assignment for MGPU (C2).
14. CMake: LTO, remove blanket `-Wno-unused-*`, de-hardcode the OpenUSD section (B7/B8/B10).

**Long term (months):**

15. Persistent wavefront scheduler (integrate the orphaned shader; eliminates the ~0.45 ms barrier floor) (D4).
16. `VK_KHR_shader_abort`/`device_fault` diagnostics (TODO.md §4 — keep).
17. SSS Approach C, Box3D physics showcase (TODO.md §7–8 — well scoped).
18. FP16 ray-payload throughput/radiance fields (48B rays) — A/B on Strix Halo first (C7).

**No rewrite needed.** The architecture is sound, current, and aligned with the stated goals; the debt is in the tail (docs, dead code, one multi-GPU dataflow decision, and the NRC gap between name and implementation).

---

### Bottom line

Pathways is in strong shape: the critical bugs from the last review are fixed, the engine decomposition improved maintainability without breaking the test suite, and the core wavefront/DGC/multi-GPU design is genuinely state-of-the-art for its hardware targets. The two things that most need attention are **(1) the P2P merge dataflow — the project's default configuration is the exact PCIe-read pattern its own prior analysis flagged as catastrophic, and the fix is a buffer-ownership inversion the existing plumbing is 90% of the way to** — and **(2) documentation/label integrity**, where the README still describes a 128-byte triangle layout, a `staging` mode that throws, and defaults that no longer match the code. Everything else is well-ordered incremental work, and the project's own docs (NRC.md, MGPU.md, TODO.md) already specify most of the right answers.
