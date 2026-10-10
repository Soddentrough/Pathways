# ReSTIR Implementation Review — Pathways

**Date:** 2026-10-10
**Reviewer:** Assistant (deep static review; no code changes in this report)
**Scope:** `shaders/compute/restir_common.glsl`, the inline resampling blocks in
`shaders/compute/wavefront_shade_diffuse.comp` / `wavefront_shade_complex.comp` /
`wavefront_shade_conductor.comp`, the host-side `src/rt/ReSTIRManager.{hpp,cpp}`,
integration through `RayTracingOrchestrator.cpp`, `WavefrontPipeline.cpp`,
`EngineDescriptorManager.cpp`, `Engine.cpp`, `Config.{hpp,cpp}`, plus docs, CLI/GUI
surface, and tests.

> Note: the reviewed state includes the currently uncommitted working-tree changes
> (the "inline ReSTIR" refactor that purged the standalone `restir_di_temporal/spatial`
> passes). This report describes the code as it exists in the working tree.

---

## 1. Executive Summary

**What exists is a competent, performance-conscious ReSTIR *DI* implementation. What the project *calls* it is wrong, and there is one confirmed math bug, one significant always-on behavior bug, several hygiene problems, and ~500 MB of VRAM allocated for a feature that is off by default.**

The active algorithm is: 4-candidate light sampling (alias/light-tree) → 1-sample RIS
selection → temporal reprojection of a previous-frame light reservoir → wave32
XOR-shuffle spatial reuse across 2 neighbor lanes → final BRDF evaluation with inline
shadow test. That is a legitimate, production-style ReSTIR DI, and the core weight math
for area lights is **correct** (including the non-trivial solid-angle-measure Jacobian,
§4.1). It is emphatically **not** ReSTIR PT: no path-space reservoirs, no GI resampling,
no shift mapping of secondary vertices — every PT-specific function in
`restir_common.glsl` is dead code.

Severity-ordered headline findings:

| # | Severity | Finding |
|---|----------|---------|
| 1 | **High** | Spatial reuse block runs **even when ReSTIR is disabled**, and in that mode it biases selection without weight correction (§4.4) |
| 2 | **High** | Spot-light reuse Jacobian **double-counts 1/d²** — confirmed bias (§4.2) |
| 3 | **High** | Identity crisis: code, `--help`, README, GUI tooltips and log messages tell four different stories about DI vs. PT (§5) |
| 4 | **Medium** | `ReSTIRManager` is a hollow shell: its pipeline is never created, `recordFrame` early-exits, and its ~500 MB (4K) of reservoir buffers are allocated unconditionally and never zero-cleared (§6) |
| 5 | **Medium** | `--restir-m-cap` and `--restir-min-lights` are no-ops; the integrity test built around the latter is now vacuous (§5.2) |
| 6 | **Medium** | Subgroup shuffles execute under divergent control flow — spec-level UB, garbage-tolerant only by luck (§4.5) |
| 7 | **Medium** | Temporal reuse is disabled exactly when it matters (camera movement), despite motion vectors being fetched and reprojected (§4.6) |
| 8 | **Low** | ~300-line copy-paste of the whole ReSTIR block between diffuse/complex shaders; dead struct fields; `age` field stored but never used for any decision (§7) |

---

## 2. What the Implementation Actually Is (Data Flow)

```
Engine (flag 1<<9 in UBO)
  └─ WavefrontPipeline shade dispatches (bounce 0 only)
       binding 32: currentReservoirs[frameSlot]      (write)
       binding 36: historyReservoirs[1-frameSlot]    (read)
       per diffuse/complex primary hit:
         1. M = min(numLights, 4) candidates via alias/light-tree, RIS stream select
         2. temporal: reproject via MV buffer, reconnect prev (lightIdx, uv8),
            re-evaluate p̂, J = q_prev/q_curr, M-cap 16, stream-combine
         3. spatial: XOR-lane {1,2} shuffle, reconnect, re-evaluate, M-cap 8, combine
         4. finalize: full GGX+diffuse BRDF × emission × risWeight (wsum/M/p̂, clamp 500)
         5. inline shadow ray to selected light; if occluded → contribution 0 AND reservoir zeroed
         6. store r → currentReservoirs[pixelIndex]
```

Storage is a compact 32 B `PathReservoirPT` (light index + M, wSum, packed flags/age/oct-dir,
scalar p̂, reconnect dist, packed light normal + 8-bit-per-channel light UV, FP16 radiance).
Double-buffered by frame slot; single-queue ordering makes the history read-after-write safe.
Descriptor plumbing (`EngineDescriptorManager.cpp:286-318`) is correct, and resize rebinds
are handled (`Engine.cpp:2365-2370`).

---

## 3. What Is Right

- **The area-light temporal/spatial math is correct.** This is the part most DIY ReSTIR
  implementations get wrong. Because the source pdf is expressed in *per-vertex
  solid-angle measure* (`lightPdf = d²/(A·cosL)`), the reuse update requires the ratio
  `q_prev/q_curr`, and `evalDirectJacobian` computes exactly
  `(d_p/d_n)²·(cosL_n/cosL_p) = q_p/q_n`. Combined with a p̂ that deliberately omits the
  geometric terms, `weightPrev = w·(p̂_n/p̂_p)·J` reduces to the exact `p̂_n/q_n`. Verified
  by substitution.
- **History M-truncation is done properly**: `wPrevNorm = (wSum/mPrev)·min(mPrev,16)`
  scales the sum when clamping M (`wavefront_shade_diffuse.comp:734-736`) — many
  implementations silently inflate weights here.
- **Zero-weight proposals are counted in M** (candidates rejected by NdotL still
  increment M) — correct RIS bookkeeping.
- **Scalar/luminance target p̂** is a legitimate choice: it is *unbiased* (the p̂ choice
  only affects variance in the finalization `f·wsum/(M·p̂)`), at the cost of color
  variance. The proxy `(albedo+F0+0.04)·lum` is sensible and cheap.
- **Register-only spatial reuse via `subgroupShuffle`** — no LDS staging, no extra
  dispatch, no G-buffer round-trip. Good instinct for a wavefront architecture (but see
  §4.5 for its legality problem).
- **Geometry validity gates** for spatial reuse (normal dot > 0.80,
  world-dist/cam-dist < 0.10) and light-index bounds checks before indexing `lights[]`
  are present.
- Occlusion is applied to the *final selected sample* (inline shadow ray), and the
  occluded reservoir is zeroed so bad samples don't poison history.
- The Oct-5/Oct-7 review cycles already caught and honestly re-labeled `--restir-pt`
  once; the codebase has a functioning docs-vs-code audit culture to plug this report
  into.

---

## 4. Correctness Analysis

### 4.1 Measure/Jacobian (area lights) — correct

See §3. Worth a comment block in `restir_common.glsl` because it is subtle and sits next
to a wrong case (§4.2), which invites "consistency" fixes in the wrong direction.

### 4.2 **Bug: spot-light Jacobian double-counts 1/d²** (bias)

For spots, sampling is delta-direction with `lightPdf = 1` (counting measure over light
index); the `1/d²` and cone factor are folded into `lightEmission` *inside the target*
(`reconnectDirectPath`, `restir_common.glsl:120-126`). The correct reuse factor is
therefore `q_p/q_n = 1`. But `evalDirectJacobian` returns
`clamp((d_p/d_n)², 0.05, 20)` for `SPOT` (`restir_common.glsl:147-150`) on top of a p̂
ratio that already carries `d_p²/d_n²`. Net error factor: `(d_p/d_n)²`.

- Static camera + static geometry: `d_p = d_n`, invisible.
- Spatial reuse: neighbors' distances always differ → systematic spatial bias for spots,
  biased *toward* candidates the neighbor saw at closer range.
- Animated objects under temporal reuse: bias proportional to the squared distance change.

**Fix:** `J = 1` for `SPOT` (and keep `J = 1` for directional). One line, plus a
regression test with an off-center spot and a moving surface.

### 4.3 Bias inventory (mostly acceptable, none documented)

Clamped/forbidden reuse and clamped weights are standard engineering, but they are
undocumented and their compounding is unstudied:

- `clamp(J, 0.05, 20)`, distance-ratio gate `[0.2, 5]` (rejects reuse → truncation bias),
- `risWeight` clamp `[0, 500]` (ReSTIR path) vs `[0, 10000]` (non-ReSTIR path) — inconsistent,
- `cosLightOrig` floored at `0.01` (grazing-angle bias),
- spatial M-cap 8 / temporal M-cap 16 (known, accepted ReSTIR bias),
- light UV stored at **8 bits per channel** — quantization bias for area-light temporal
  reuse; visible as micro-stepping when re-projecting across a large quad light.

### 4.4 **Bug: spatial reuse runs with ReSTIR disabled — and is weight-inconsistent there**

The "In-Register Subgroup Spatial Reuse" block (`wavefront_shade_diffuse.comp:771-858`,
same in `wavefront_shade_complex.comp`) is **not** guarded by `enableRestirPT`, unlike
the temporal block. Consequences:

1. The default (no `--restir`) path pays the shuffle/reconnect cost on every primary
   diffuse hit.
2. Worse: when a spatial candidate wins, it overwrites
   `selectedLightIdx/Dir/Emission/Target`, but the finalization in non-PT mode uses
   `risWeight = totalWeightSum / (M·selectedTarget)` — the fresh-stream weight sum
   combined with a *neighbor's* target and a selection driven by information not present
   in the sum. That is a biased estimator (and `selectedLightSelectPdf` goes stale for
   the MIS term). In PT mode the bookkeeping is consistent (everything accumulates into
   `r.wSum`).

**Fix:** guard the block with `enableRestirPT`, or make non-PT finalization use
`r.wSum/(M_eff·p̂)` consistently. As written, "ReSTIR off" is not actually "ReSTIR off" —
which silently invalidates any A/B measurement of the flag.

### 4.5 Subgroup UB under divergence

`subgroupShuffle(hitPoint…, partnerLane)` etc. run inside per-lane-divergent control
flow (`inRange`, `pathTerminated`, `isDelta`, cutout continuation). When a partner lane
has branched away, the shuffle result is undefined per spec. The validity checks
(`isValidReservoir`, `nbrWSum > 0`, `nbrLightIdx < numLights`) filter garbage most of the
time, but this is luck, not design. Recommended: perform spatial reuse from the
**history reservoir buffer** at screen-space neighbor taps (the standard ReSTIR DI
spatial pattern) instead of shuffling live registers across a divergent wave — this is
fully legal, removes 12 shuffles/pixel, and enables proper random tap offsets.

### 4.6 Temporal policy: disabled exactly where it pays

`historyValid` requires `!(flags & (1<<23))`, and bit 23 is set whenever
`accumReset || cameraMovedLastFrame` (`Engine.cpp:1695-1698`). So during camera motion —
the 1-SPP real-time scenario ReSTIR exists for — the reservoir history is discarded even
though the code faithfully loads motion vectors and reprojects. Meanwhile, with a *static*
camera and progressive accumulation already averaging frames, temporal reuse contributes
comparatively little. The `normalDepth`/`prevNormalDepth` images are plumbed through the
(dead) manager but never used for disocclusion validation. This is the single biggest
quality-per-effort lever in the feature: MV reprojection + depth/normal rejection +
max-age truncation, enabled during camera motion.

Also: the reservoir buffers are **not cleared on allocation**
(`ReSTIRManager::initBuffers`), so after a mid-session resize (frameIndex > 0, fresh GPU
memory) the first reuse frame can interpret uninitialized bits as a valid reservoir. The
valid-bit lottery mostly spares you; zero-fill on allocation is the cheap fix.

### 4.7 Multi-sample (SPP > 1) semantics

Each SPP iteration re-shades bounce 0 and overwrites `currentReservoirs[pixelIndex]`;
all samples in a frame reuse the *same* previous-frame history. Correctness is fine
(last-writer-wins becomes next frame's history), but samples are correlated through
shared history — worth knowing when interpreting variance tests.

### 4.8 Stale-reservoir edge cases

Pixels shaded by the dielectric pass, or by the complex pass when `transmission >= 0.1`,
never write `currentReservoirs` (the complex guard skips both the block *and* its clear
branch). Those slots retain data from two frames ago; a pixel that crosses such a
material boundary during motion can serve stale history with a p̂ that matches neither
surface. A "write invalid reservoir on skip" closes this.

---

## 5. Naming, Docs, and Interface Honesty

### 5.1 Four contradictory stories

- `restir_common.glsl` / `ReSTIRManager.{hpp,cpp}` / GUI tooltip: "**True ReSTIR PT**",
  "path reservoirs", "GRIS", "path-space reservoir resampling"
  (`GuiManager.cpp:1294-1299`, `Config.cpp:913-920`).
- README:289: "`--restir-pt` currently warns and enables DI only — full ReSTIR PT is
  unimplemented" — now *itself* stale, since `--restir-pt` silently enables the same
  thing with an info log claiming PT.
- `docs/ARCHITECTURE.md:323`: "ReSTIR DI".
- Reality: ReSTIR DI with temporal + wave32 spatial reuse.

The previous review cycles (Oct 5/7) fixed exactly this class of issue and the pendulum
swung back — `--restir-di` now *warns that DI was purged* while the actual code is DI.
Given that the project's stated differentiator is verifiable claims, standardize on
"**ReSTIR DI (inline, wave32)**" everywhere and reserve "ReSTIR PT" for the (currently
unimplemented) GI-reservoir path, exactly as the Oct-7 report's roadmap item defines it.

### 5.2 Dead configuration surface

- `--restir-m-cap` (default 30): flows into `ReSTIRManager::recordFrame` push constants
  only — the pipeline is never created (§6). The live caps are hardcoded `16`/`8` in two
  shaders. The GUI slider is decorative.
- `--restir-min-lights`: parsed and discarded (`Config.cpp:931-937`), but README still
  documents auto-activation semantics, and
  `scripts/verify_visual_integrity.py::test_restir_many_lights_variance_and_gating`
  still runs a "bypass" configuration (`--restir-min-lights 100`) that no longer bypasses
  anything — the test passes vacuously.
- `ReSTIRPushConstants.hasLightTree`, `numTriangles`, `passIndex` and the entire
  10-binding descriptor layout of the manager: dead.

### 5.3 Dead PT scaffolding

`evalUnshadowedTargetGI`, `evalPTJacobian`, `validateReconnectionFootprint`,
`updateReservoirPT`, `combineReservoirsPT` have zero call sites (previously flagged
Oct 7, still present). The reservoir's `secondaryRadianceRG/B` fields are written but
never read (`unpackRadiance*` has no call sites — radiance is re-fetched from the light
table via `lightIndex`); `getPathLength`/`getLobe` are never read; `age` is incremented
and capped at 30 but **never used for any decision**. This scaffolding reads as "PT
implemented" to every future reviewer and LLM-assisted pass — that is its main cost.

---

## 6. ReSTIRManager: A Class Whose Only Living Job Is `malloc`

`Engine::createReSTIRResources` passes `{}` as the fused SPIR-V
(`Engine.cpp:1069-1079`), so `m_fusedPipeline == VK_NULL_HANDLE` forever and
`recordFrame` early-exits after a resize check (`ReSTIRManager.cpp:172-177`). The engine
still registers a post-classify callback that calls it every frame
(`Engine.cpp:888-913`) — harmless but noise. Meanwhile:

- Buffers are allocated **unconditionally at startup** even with `--restir` off:
  2 × W·H·32 B ≈ **506 MiB at 4K** (UMA-shared on Strix Halo, but real).
- The header comment describes a dual-slice layout (`[0..N-1]` shading, `[N..2N-1]`
  bounce-1 candidates) that the allocation and all live code contradict — a stale design
  doc in disguise.
- `recordFrame` has a 19-parameter signature; the class owns descriptors/pool/layout for
  a pipeline that cannot exist.

**Recommendation:** lazy allocation (first frame `enable_restir` is true), zero-fill on
(re)allocation, delete the pipeline/descriptor/recordFrame machinery (keep a slim
buffer-owner or fold the two buffers into `WavefrontPipeline`, which already owns their
bindings), and delete the callback call site.

---

## 7. Engineering Quality

- **Duplication**: the ~300-line ReSTIR block is copy-pasted between
  `wavefront_shade_diffuse.comp` and `wavefront_shade_complex.comp` (diffed: identical
  except clearcoat/sheen and two predicates). The Jacobian bug (§4.2) exists twice. This
  belongs in a shared include next to `restir_common.glsl`.
- **Fallback binding hazard**: when no ReSTIR buffer is supplied, bindings 32/36 alias
  `m_queueCounters` (`WavefrontPipeline.cpp:388-392`). Currently unreachable (flag 9
  requires the manager), but one refactor away from the shader scribbling reservoirs over
  queue counters. A dedicated small dummy buffer costs nothing.
- **No tests**: no unit test of `evalDirectJacobian` (which would have caught §4.2
  against a closed-form spot case), no statistical gate beyond a lenient
  `max(wall, floor)` Laplacian-variance comparison in the integrity script, and no bias
  check against a high-SPP reference. The Oct-7 report's "verifiable claims" standard
  applies here first.
- **No debug views**: no M / W / reservoir-valid viewport, which makes every tuning
  question (caps, gates) a blind A/B render.

---

## 8. Position vs. SOTA

Against 2022-2025 production ReSTIR DI (Lin et al. 2022; Bitterli & Eisemann;
Unreal/Blender implementations), this is roughly "correct core, minimal feature set":

| Capability | Status |
|---|---|
| Solid-angle-measure RIS with correct q-ratio (area) | ✅ |
| Temporal reuse, M-truncated, MV-reprojected | ✅ (but disabled during camera motion) |
| Spatial reuse | ⚠️ 2 XOR-neighbors, fixed pattern, no random offsets, no disocclusion guard, UB under divergence |
| Visibility in target / shadow re-validation for reused samples | ❌ (target is unshadowed; acceptable, but final-sample-only shadowing + reservoir zeroing is a heuristic mix) |
| Disocclusion detection (depth/normal) | ❌ images plumbed, unused |
| Bias documentation | ❌ |
| ReSTIR GI / PT reservoirs | ❌ (dead scaffolding only) |
| Hybrid: ReSTIR for glossy (bounced) light | ❌ conductors/dielectrics use a separate 1-stage RIS with no reservoir |

For the project's stated 1-SPP real-time goal on Strix Halo, the honest framing is: this
is a good DI variance reducer for many-light scenes, not the "ReSTIR PT" that the TODO
lists as the big Veach-Ajar win.

---

## 9. Prioritized Recommendations

**Days (correctness + honesty):**

1. Fix `evalDirectJacobian`: return 1.0 for `SPOT` (§4.2). Add a closed-form unit test.
2. Guard the spatial-reuse block with `enableRestirPT` or make non-PT finalization use
   `r.wSum` consistently (§4.4).
3. Rename everything to "ReSTIR DI (inline wave32)"; fix README:289-291,
   `Config.cpp:288/913-920`, GUI tooltip, and the startup log line; delete or clearly
   mark the PT scaffolding (§5, §5.3).
4. Make reservoir allocation lazy + zero-filled (§6).
5. Fix the vacuous integrity test (re-implement min-lights gating or drop the assertion);
   document the clamp/gate bias list.

**Weeks (quality + headroom):**

6. Enable temporal reuse during camera motion with depth/normal disocclusion checks
   (the gbuffer images already exist); use stored `age` for max-age truncation (§4.6).
7. Replace divergent-wave shuffles with history-buffer neighbor taps (§4.5); add random
   tap offsets; dedupe diffuse/complex blocks into one include; write invalid reservoirs
   on skipped material paths (§4.8).
8. Wire `--restir-m-cap` into the shade pass push constants; add M/W debug views and a
   variance-reduction CI gate against the existing `many-lights` scene.
9. Shrink the reservoir to the fields actually read (remove the dead FP16 radiance
   fields; store `cosLightOrig` and the primary hit distance directly) — smaller footprint
   and less bandwidth.

**Strategic (the actual goal: ReSTIR PT):**

10. Keep "ReSTIR PT" as a separate, properly-scoped roadmap item: GI/path reservoirs on
    bounce ≥ 1 with **path replay**. The current `PathReservoirPT` secondary-vertex fields
    are not a starting point — correct GI reuse requires either full path storage/replay
    or a first-vertex reconnection scheme with an X1-context buffer captured at bounce 0
    (position, normal, BSDF params, ωo, source pdf), reuse evaluated at bounce 1, replay
    visibility rays, and careful estimator decomposition so the 2-bounce contribution is
    counted exactly once. The wavefront's material-sorted queues make this a multi-day
    design effort; the fix list above (shared include, history-tap reuse, honest naming)
    is the foundation it should be built on.

---

## 10. Overall Assessment

**B− as engineering** (correct core math where it's subtle, one real bias bug, one real
always-on bug, weak test coverage), **D as communication** (the naming/documentation
problem is the project's own worst-enemy pattern, re-introduced after being fixed once).
The fix list is short and mostly mechanical; the algorithm deserves it.
