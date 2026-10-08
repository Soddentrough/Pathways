# Pathways — Full Project Review

**Date**: October 7, 2026
**Scope**: Full codebase (`src/` ~43k LOC, 43 GLSL/SPIR-V shaders, `CMakeLists.txt`, CI, docs, tests), git history through HEAD (`b2ae75f`), verified by clean build, full CTest run, live headless engine runs (`--upways`, `--mgpu`), a live Vulkan-Hardware-DB fetch via `scripts/audit_vulkan_api.py`, and an FidelityFX-SDK integration feasibility study.
**Supersedes**: `codebase_review_2026_10_05.md` for status tracking. That review's remediations are re-verified here (Section 3) and held, with two exceptions found this cycle.

**Review host**: AMD Strix Halo APU (Radeon 8060S, RADV, Mesa 26.2.3). Note: the host was running a ~22 GB resident `llama-server` on the same UMA GPU during benchmarking — absolute timings in this document are contaminated; relative comparisons remain indicative.

---

## 1. Executive Summary

Pathways remains a genuinely advanced, well-engineered Vulkan 1.4 path tracer. The October 5 quick wins verifiably landed and survived the Oct 6 branch merge: shaders are uniformly `#version 460`, orphan shaders are gone, `UINT64_MAX` waits are bounded, `Light` is 128 B, the push-constant bit-packing hack is gone, and the engine runs with **0 validation errors** (verified live). The Engine decomposition (2,666 lines from ~9,500) is a real maintainability win, and the data-layout discipline (`static_assert`-guarded 64 B triangles, 128 B lights, FP16-packed 16 B ray state) is better than most production engines. **No rewrite is needed.**

The findings this cycle cluster into one code-vs-documentation inversion, one broken quality gate, and a set of label-vs-function mismatches (FSR 3.1, NRC, `--restir-pt`) that matter precisely because the project's differentiator is verifiable technical claims.

---

## 2. Findings

### A. Critical — Code vs. Documentation

#### A1. DGC Execution Sets: the runtime default is the opposite of what README and `--help` describe, and `--dgc-execset` is a no-op — **FIXED this cycle**
- README §1 claimed production defaults to multi-dispatch indirect and that Execution Sets are an experimental opt-in via `--dgc-execset`.
- Reality (verified live): `DGCManager::initMaterialExecutionSets` enables the exec-set path **unless** `PATHWAYS_DISABLE_MATERIAL_DGC`/`PATHWAYS_DISABLE_DGC_EXECSET` env vars are set. With no flags the engine creates exec sets, logs `WARN "Experimental DGC Material Execution Set active"`, and dispatches material microkernels through `recordMaterialPreprocess/Execute` (`WavefrontPipeline.cpp:868-872`).
- `--dgc-execset` (`Config.cpp:1099`) sets `PATHWAYS_ENABLE_DGC_EXECSET`/`PATHWAYS_ENABLE_MATERIAL_DGC` — **neither env var is read anywhere**. The flag did nothing.
- The multi-dispatch fallback loop in `recordMaterialExecute` is fully implemented, so both paths work.
- Local A/B (Strix Halo, 1080p Cornell, contaminated host): exec-set 3.652 ms vs multi-dispatch 3.776 ms (~3.4% in favor of exec-set; the README's "+22.3%" multi-dispatch claim is RDNA4-attributed and could not be re-verified here).
- **Resolution (Oct 7)**: the flag pair was made real — `--no-dgc-execset` disables, `--dgc-execset` force-enables, and the README/help text now describes the actual default (exec sets when the driver advertises compute execution sets; multi-dispatch otherwise or on request). The "+22.3%" sentence was re-attributed to GPU-autonomous indirect dispatch vs CPU readback. **Action item**: re-run the exec-set vs multi-dispatch A/B on the RDNA4 testbeds and let the winner be the single documented default.

#### A2. The `PerformanceMatrix` CTest gate fails on a clean checkout on the project's own dev host — **OPEN**
- Verified: `ctest` → 19/20, `PerformanceMatrix` FAILED with "+100% to +537%" apparent regressions across all 42 measured configs.
- Root causes: (1) `tests/references/performance_baseline.json` carries **no hardware/driver/date/build metadata** — it was recorded on the dual-RDNA4 testbed (mgpu baselines ≈ half the single-GPU numbers) and compared verbatim on Strix Halo; (2) `--mgpu` on a single-GPU machine silently falls back to single-GPU rendering but the harness still compares against dual-GPU baselines; (3) no system-idle guard — on UMA, co-resident workloads (this review found a 22 GB LLM server on the same iGPU) directly steal LPDDR5X bandwidth.
- A permanently-red gate trains developers to ignore test failures.
- **Fix**: key baselines by device name (+ driver, git hash, build flags); skip `*_mgpu` profiles when <2 qualifying devices exist; add a pre-benchmark idle check (GPU utilization / competing VRAM consumers); enforce `RADV_PROFILE_PSTATE=peak` in the harness; auto-skip with a printed reason on hardware mismatch.

#### A3. "AMD FidelityFX Super Resolution (FSR 3.1)" branding vs. actual implementation — **OPEN (integration work stopped per project owner, Oct 8)**
- The `fsr3_*` shaders are a bespoke temporal-reprojection upscaler with a 5-tap cross Laplacian sharpening pass — not AMD's EASU/RCAS, and the official SDK is not in the tree. The README, `ARCHITECTURE.md`, and the CLI brand it as AMD FSR 3.1; that claim is not accurate under inspection.
- **SDK feasibility study (Oct 7–8)**, recorded for any future revival:
  - The local `FidelityFX-SDK` clone at tag `v2.3.0` is **DX12-only** (its own known-issues table: "Vulkan is currently not supported in AMD FSR SDK 2.3") — unusable for a Vulkan renderer.
  - The **`v1.1.x` tags of the same repository contain the open-source FSR 3.1 Upscaler (`ffx_fsr3upscaler`) with a Vulkan backend** (`sdk/src/components/fsr3upscaler/`, `sdk/src/backends/vk/` incl. `CMakeShadersFSR3Upscaler.txt`), matching AMD's *FSR 3.1 Release Overview and Integration* guide ("Supported graphics APIs are DX12 and Vulkan"). MIT licensed.
  - Verified: the v1.1.4 Vulkan `.glsl` pass sources **compile directly with stock `glslangValidator`** (`-DFFX_GLSL=1 -DFFX_GPU=1 --target-env vulkan1.2 -e CS -S comp`) — AMD's Windows-only `FidelityFX_SC.exe` is not strictly required for the shader step.
  - The real blocker is the host side: using the official component (`ffx_fsr3upscaler.cpp`, 1,509 ln) requires building AMD's ~4,700-line `ffx_vk.cpp` device/resource/pipeline framework alongside Pathways' own `VkDevice`/VMA/command-buffer ownership, plus generating SC-format permutation-blob headers (indirection tables + reflection metadata) for ~10 passes × 64 permutations × wave/16-bit variants, plus single- and multi-GPU barrier-coherency validation. Assessed as a multi-day, high-validation-risk project disproportionate to the technique's priority.
- **Decision (project owner, Oct 8)**: FSR3 work is descoped — neither official integration nor feature removal will be pursued now; it is not a priority. Consequence: the finding stands — the FSR branding describes the custom filter, and published FSR-path benchmark numbers belong to that filter. A naming-only pass (drop AMD/FSR branding from the custom filter, or clearly label it "FSR-style") would close A3 at documentation cost only whenever the owner chooses.

#### A4. NRC still does not implement what its in-app labels promise — **FIXED (labels) this cycle; research gap OPEN**
- Unchanged from Oct 5: training target is direct light only (`targetRad = vec3(secDirectL)`, `wavefront_shade_diffuse.comp:867`); training dispatch is hard-coded to 1 workgroup = 32 samples/frame (`NRCManager.cpp:575`).
- The Oct 5 relabel was applied to the README but not to the surfaces users actually see: `--help` said "Neural Radiance Caching", the GUI header said "Neural Radiance Caching (Wave32 WMMA)", and startup logs said "Neural Radiance Caching Subsystem". **Fixed Oct 7**: all in-app surfaces now read "Neural Direct-Light Caching (experimental)" with a pointer to `docs/NRC.md` for the full-radiance roadmap.
- The functional gap (Option-2 path-tail supervision, B5 hybrid training topology, 32 B query records) remains the highest-value research item, already fully specified in `docs/NRC.md`.

#### A5. README showcases Upways performance numbers the test suite refuses to gate — **OPEN**
- The Veach Ajar showcase advertises "Native 4K Upways Denoising — only 2.47 ms inference overhead" and "`--upways-sr` … 3.3x Speedup". Meanwhile the three Upways visual-regression configs are marked `"experimental": True` and excluded from `--strict`, and the reconstructor had a "destructive albedo squaring" bug fixed only on Oct 6 (`6c8be7a`).
- The main WMMA pass now runs with 0 validation errors at 1080p (verified live); the fallback `upways_reconstruct.comp` still has the descriptor-type mismatch (bindings 7/8/9/12 `image2D` vs `sampler2D`) documented on Oct 5 — only the dispatch geometry was fixed.
- **Fix**: re-measure and re-qualify the showcase numbers on the current build and un-gate the visual configs, or move the Upways rows to an explicitly-labeled "Experimental results" subsection.

---

### B. Minor Bugs, Code Smells & Technical Debt

1. **CLI label mismatches** — **FIXED this cycle**: `--help` claimed `--sec-sort` default `directional` (actual: `direct`/Xiang-2023); `--restir-pt` was a silent alias for `--restir` (DI) — it now warns that PT is unimplemented; `--nrc` help/GUI labels (A4).
2. **Docs drift** — **FIXED this cycle**: `docs/README.md` hub and the `ARCHITECTURE.md` binding table still described the old **128-byte** `TriangleShadeGPU`; README run example referenced nonexistent `scenes/living-room/living_room.glb`; README hardcoded a machine-specific Radeon Tool Suite path; hardware-coverage table lacked a "data as of" date (live check: DGC desktop coverage 48.8% vs the table's 49.3%).
3. **Unresolved numeric inconsistency** (OPEN): README lists Cornell Box 4K as both "7.36 ms (135.9 FPS)" (showcase) and "8.01 ms (124.8 FPS)" (architecture table) — different measurement sessions, no dates shown. The "2.28x super-linear scaling" row needs an evidence footnote or removal. Tables now carry a measurement-provenance note; re-measurement on the RDNA4 testbed should re-cut all numbers in one session.
4. **Warning suppressions (measured)**: removing the blanket flags yields **2,244** `-Wmissing-field-initializers`, **38** `-Wunused-parameter`, **12** `-Wunused-variable` (e.g. `Engine.cpp:1715-1731` refactor leftovers: `groupsX/groupsY/useAsyncComputeMerge`). The missing-field count is Vulkan aggregate-init style — mechanically fixable with designated initializers; the suppression can hide a forgotten push-constant/struct member (a silent zero). Budget a dedicated sweep, then delete the flags (CMakeLists:32-34).
5. **No LTO / math flags**: `CMAKE_INTERPROCEDURAL_OPTIMIZATION` unset; `-fno-math-errno`/`-fno-trapping-math` absent. Free CPU-side wins.
6. **`catch (...) {}` swallowing** in `VulkanContext.cpp:634,703,717,725` — log at debug level at minimum.
7. **Dead/unused**: `restir_common.glsl` PT helpers (`evalUnshadowedTargetGI`, `evalGIJacobian`, `validateReconnectionFootprint`) unreferenced; `VK_EXT_descriptor_heap` enabled on device creation (`VulkanContext.cpp:854`) but used nowhere — enable-and-use or don't enable.
8. **Config/telemetry hygiene**: `MAX_SCENE_TEXTURES = 512` duplicated in 3 headers; telemetry JSON defaults to CWD (`ImageDumper.cpp:202`), which is what scatters `pathways_telemetry_*.json` across the repo root; ~8 undocumented `PATHWAYS_*` env-var knobs (`PATHWAYS_PROFILE_WF`, `PATHWAYS_INSTANCE_DENSITY`, `PATHWAYS_CULL_DISTANCE`, `PATHWAYS_DISABLE_DGC*`, `PATHWAYS_USD_MAX_INSTANCES`).
9. **CMake machine-specific hardcoding** (Oct 5 B11, still open): `C:/Users/naoki/Development/USD`, `/usr/lib64/libtbb.so.2`, `/usr/lib/python3.14/site-packages/materialx`.
10. **Windows harness parity** (Oct 5 B17, still open): `run_headless_tests.ps1` lacks ctest scene-switching, mgpu suites, and visual regression; CI skips all GPU suites (`c2e0a51`) — the only enforcement point is local Linux discipline.
11. **Vendored deps track development snapshots**: Dear ImGui "1.93.0 WIP", GLM 1.1.0 (master; last tagged release 1.0.x). Pin to tagged releases or record exact commit hashes.
12. **`EPSILON 0.0005`** fixed world-space ray offset (`wavefront_common.glsl:20`) — leak/acne across scene scales (Oct 5 B6, open); `1000003u` frame decorrelation (`MultiGpuCoordinator.cpp:143`) unexplained.
13. **FP16 accumulation stall**: the "Welford" running average (misnomer — it is a plain online mean; Welford computes variance) updates with `alpha = 1/N` into RGBA16F history. Beyond N ≈ 1024 the update falls below the FP16 ULP and convergence stalls, so "progressive convergence up to 2048 SPP" is numerically unreachable on the default FP16 path. Fix: auto-promote the accumulation target to RGBA32F above ~512 SPP (the `AccumFormat` switch already exists) or accumulate sum+count with FP32 sum. Renamed "Welford" → "online mean" in docs this cycle; the numeric fix is OPEN.
14. **Tests**: hand-rolled `main()`s, no framework; no direct unit coverage for `QualityGovernor` EMA, `MultiGpuCoordinator` planning math, or the batch solver (all pure functions). The GPU-touching tests (`DgcCompact`, `BlasBatch`, `P2PDirectBar`) are genuinely good.

---

## 3. Verification of the Oct 5 Review Against Current Code

| Oct 5 item | Status (Oct 7, verified) |
|---|---|
| Shader `#version 450` → `460`, drop `core` | ✅ Done (all 43 shader files on 460) |
| Orphan shaders deleted | ✅ Deleted (`3c658dc`); a parallel branch re-synced them to 64 B (`12eaa90`) and the merge kept them deleted — clean resolution |
| Bounded sync timeouts | ✅ Done — zero `UINT64_MAX` waits remain in `src/` |
| `pc.pad` bit-packing → explicit members | ✅ Done (`accum_merge.comp:49-50`, survives merge) |
| `Light` 96 B → 128 B | ✅ Done (`wavefront_common.glsl:190-199`, CPU+GPU in sync) |
| Doc-sync pass (64 B triangle, transfer modes, DGC requirement, NRC README label, sort defaults) | ⚠️ Mostly done; survived in README but **not** in `docs/README.md` hub / `ARCHITECTURE.md` binding table / in-app help — fixed this cycle (A4, B2) |
| Upways fallback dispatch fix | ✅ Dispatch fixed; **descriptor-level mismatch still open** (A5) |
| Visual-regression `--render` + experimental gating | ✅ Done; Upways configs excluded from `--strict` |
| DGC execset "marked experimental" | ⚠️ Doc-only fix — masked the code-side inversion found as A1 |

---

## 4. FSR 3.1 — Feasibility Study Outcome (work stopped per project owner, Oct 8)

A full feasibility study was performed (technical record in A3): the official Vulkan FSR 3.1 Upscaler exists (FidelityFX-SDK `v1.1.x` tags, MIT), its Vulkan `.glsl` passes compile with stock `glslangValidator`, and the local `v2.3.0` checkout is DX12-only. Clean integration would require adopting AMD's heavy VK device/pipeline framework plus SC-format permutation-blob generation and multi-GPU barrier-coherency validation — a multi-day, high-risk effort.

**On October 8 the project owner directed that FSR3 work stop; it is not a priority.** No FSR code, shaders, CLI, or build files were changed this cycle. Consequence to keep in mind: the README/ARCHITECTURE "FSR 3.1" branding continues to describe the custom filter (A3 remains open), and FSR-path benchmark numbers belong to that filter, not to AMD's technique. A future naming-only pass would close this cheaply; nothing else was done.

---

## 5. Performance Optimizations (carried forward + new)

1. **Resolve A1 with RDNA4 data**: benchmark exec-set vs multi-dispatch on the target testbeds; keep the winner as the single documented default.
2. **MGPU work-stealing tile assignment** (open): shared buffer + atomics already exist; static checkerboard parity is the remaining multi-GPU headroom on caustic-heavy scenes.
3. **Descriptor buffers + push descriptors**: `pushDescriptor` enabled but unused; the 36-binding wavefront set with even/odd ping-pong is the textbook client; also the right RDNA5 posture. Drop `VK_EXT_descriptor_heap` until used.
4. **ReSTIR PT**: dead GI-resampling functions + a (now honest-warning) `--restir-pt` flag already exist; Veach Ajar (26 ms pure MC at 4K) is the target scene. Biggest noise win for the 1-SPP real-time goal.
5. **FP16 accumulation stall fix** (B13): promote to RGBA32F above ~512 SPP.
6. **Opacity micromaps** (`VK_EXT_opacity_micromap`) for alpha-MASK glTF/USD geometry — bounded project, RADV support exists; add to TODO.md.
7. **`VK_EXT_shader_object`** evaluation for wavefront microkernel dispatch (complements DGC on the fallback path); `VK_KHR_pipeline_binary` for startup hitch; `VK_EXT_host_image_copy` for UMA uploads.
8. **Ray payload FP16 A/B** (Oct 5 C7): 16 B `RayState` is already FP16-packed; the remaining payload throughput/radiance fields are the next queue-bandwidth cut on gfx1151.
9. **Benchmark hygiene** (fixes A2 structurally): per-device baselines with metadata, idle guards, pinned p-state, one-session re-cut of all README tables.
10. **Adaptive epsilon** (B12).

---

## 6. Nice-to-Have / Roadmap Notes

- Persistent wavefront scheduler (barrier-floor elimination; prototype recoverable from git history), SSS Approach C, Box3D physics showcase — all well-scoped in TODO.md, unchanged.
- `VK_KHR_shader_abort` + `VK_KHR_device_fault` (TODO §4): keep priority — DGC-autonomous dispatch (exec sets on by default on capable drivers, per A1) makes GPU-side fault diagnosis currently near-impossible.
- Test framework (Catch2) + unit tests for governor/coordinator; `--ui-persist` for ImGui layout.
- SOTA alignment: the existing plan covers ReSTIR PT, NRC Option-2, descriptor buffers, OMM, RDNA5 readiness. Two additions: AMD's **FSR Ray Regeneration** / **Radiance Caching** (SDK 2.x, ML-based, currently DX12 — watch for Vulkan) as the eventual evolution of Upways' 16.7 KB KPN; and NIS-line joint SR+denoise formulations. RDNA5 is not shipping hardware; keep "RDNA5-readiness" language aspirational.

---

## 7. What's Right (keep doing it)

- Clean build; engine at 0 validation errors; 19/20 CTest with the sole failure being the A2 gate problem, not the engine.
- Data-layout discipline with `static_assert`s across every hot struct.
- The self-correcting review loop: the Oct 5 review caught and documented its own predecessor's misread (CRIT-03/push-DMA), and its remediations verifiably survived a branch merge. This is the single healthiest signal in the repo.
- Vulkan adoption (maintenance4-6, sync2, dynamic rendering + local read, BDA, subgroup size control, cooperative matrix, DMA-BUF P2P) contains no deprecated APIs; the strict capability gate in `verifyPhysicalDeviceRequirements` is the correct expression of "no backwards compatibility."
- Docs infrastructure (live audit script, dated reports, supersession chain) is genuinely good — the recurring failure mode is README claims outrunning the harness that keeps them honest; the A2/B9 fixes close that loop.

---

## 8. Prioritized Action Plan

**Landed this cycle (Oct 7)**:
1. ✅ A1 — `--dgc-execset`/`--no-dgc-execset` made real; docs describe the actual default; RDNA4 A/B remains as follow-up.
2. ⏹️ A3 — FSR 3.1 official-integration feasibility studied in depth (v1.1.4 Vulkan upscaler confirmed; shaders compile with stock glslangValidator); **work stopped per project owner (Oct 8)**. No FSR code changed; the branding finding remains open (see §4).
3. ✅ A4 (labels) + B1 + B2 — in-app NRC labels honest; `--sec-sort` help fixed; `--restir-pt` warns; docs hub/binding-table 128 B→64 B; broken scene path, hardcoded tool path, coverage-date staleness fixed; "Welford"→"online mean" rename.

**Next (days)**:
4. A2 — per-hardware baselines + idle guards + mgpu device gating (or move matrix out of default CTest).
5. A5 — Upways claims re-qualification or "Experimental" subsection; fix the fallback descriptor mismatch when Upways is revived.
6. B13 — RGBA32F accumulation promotion above ~512 SPP.
7. Re-cut all README benchmark tables in one session on the RDNA4 testbeds (fixes the 7.36/8.01 inconsistency and the 2.28x footnote with dated data).

**Medium term (weeks)**: B4 warning-flag sweep → remove suppressions; B9 CMake de-hardcoding; B10 Windows harness parity; ReSTIR PT; descriptor buffers; MGPU work-stealing; LTO + math flags.

**Long term (months)**: persistent wavefront scheduler; `shader_abort`/`device_fault`; SSS Approach C; Box3D; FP16 ray-payload A/B; NRC Option-2 revival.
