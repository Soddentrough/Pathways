# ReSTIR PT Implementation Review — 2026-10-11

**Scope:** Full review of the current ReSTIR PT implementation at `c7cd2cf`
(Phase 1 DI + Phase 2a/2b GI + Phase 3 path-database groundwork, including
upstream commits `d5e4fab`, `5d35634`, `165c804`, `198bd5b`, `1f69b5d`,
`3892255`, `c7cd2cf`).

**Re-assessment note (post-`c7cd2cf`):** §2 was re-verified after the
upstream merge — the infinity-mirror regression is **fixed upstream** and the
original root-cause attribution in this report was *wrong* (see §2.1). P2 was
also resolved upstream. All other findings were re-checked against `c7cd2cf`
and still stand unless marked otherwise.

Files reviewed: `shaders/compute/restir_common.glsl` (873 lines),
`shaders/compute/wavefront_shade_diffuse.comp` (DI §795+, GI §493+),
`shaders/compute/wavefront_shadow.comp`, `shaders/compute/wavefront_common.glsl`
(Hit atom §343+), `shaders/compute/wavefront_intersect.comp` (hit-atom packing),
`src/rt/ReSTIRTypes.hpp`, `src/rt/ReSTIRManager.{hpp,cpp}`,
`src/scene/Material.hpp` (rho_max packing), `src/core/Engine.cpp` (lazy alloc,
flags), `docs/RESTIR_PT_DESIGN.md`, `tests/test_restir_pt_math.cpp`.

---

## 1. Verdict

The estimator math is **sound**. I re-derived the GI finalization, the DI
reuse Jacobians, the M-truncation, and the MIS weights against the SNIS
form; all are correct (details §3). The architecture is clean: inline
resampling, double-buffered grids, history-only spatial taps, shadow-queue
replay visibility, lazy allocation, zero-fill discipline, and a passing 10-test
math suite.

The problems are at the **edges**: (a) one dead gate that makes an advertised
safety feature a no-op, (b) Hit-atom reconstruction assumptions that are
untrue for transformed instances and spheres, (c) Phase 3 buffers allocated
eagerly with no call sites (~272 B/pixel), and (d) several stale/contradictory
comments and doc claims. The one **critical adjacent regression** tracked
during this review cycle (infinity-mirror darkening) was **resolved upstream**
by `3892255`/`c7cd2cf`; §2 records the corrected root cause (Mesa RADV DGC
execution-set driver defect, not a sampler bias).

---

## 2. Infinity-mirror darkening — RESOLVED upstream (diagnosis corrected)

**Symptom (historical):** infinity-mirror mean brightness 12119 (`62b8141`)
→ ~8990 at `198bd5b`, −26%, legacy path (ReSTIR off).

### 2.1 Corrected root cause

The original version of this report attributed the darkening to a bias in
Dual mode's coherent clustered sampling (`sampleGGXCoherent`). **That was
wrong.** The true chain is:

1. `d6fd5ea` added the load-time `getDivergentAreaRatio()` cache in
   `applyLoadedScene()` (before host triangles are freed), so
   `--wavefront-sort auto` began resolving to **Dual** for this scene
   (previously the ratio was computed after the triangles were freed, always
   cached `0.0`, and `auto` silently meant **None**).
2. Dual sort uses a different DGC dispatch configuration, which triggered the
   **Mesa RADV `VkIndirectExecutionSetEXT` driver defect**: dynamic compute
   pipeline switching stalls non-diffuse material queues, dropping their rays
   to 0. That — not the sampler — is why emissive/conductor bounce-1
   contributions vanished.
3. `3892255` fixed the rendering by making the execution set **opt-in**
   (`Config::dgcExecSetRequested()` now defaults to false; enable via
   `--dgc-execset` or `PATHWAYS_DGC_EXECSET=1`), keeping the standard
   multi-dispatch path as default.

**Verification at `c7cd2cf`** (256², 2 spp, 120 f, `--max-bounces 4`):

| sort | mean |
|---|---|
| auto (=Dual) | 13653 |
| dual | 13647 |
| none | 13991 |

All modes are now bright and statistically indistinguishable → no coherent-
sampling bias exists; the "investigate `sampleGGXCoherent`" follow-up from the
first draft is **dropped**.

### 2.2 Residual notes

- The load-time `getDivergentAreaRatio()` cache (d6fd5ea) is now *correct*
  behavior — `auto → Dual` works and should be kept.
- The DGC execution-set default (off) is a driver-workaround policy decision;
  revisit when Mesa RADV fixes the pipeline-switching defect. The workaround
  is documented in `Config.cpp`.
- The `--scene-bounces` opt-in added in `c7cd2cf` (default off) correctly
  reverts the automatic `sceneDefaultMaxBounces` adoption from `3892255`,
  preserving the default 4-bounce budget for non-opted-in scenes.

---

## 3. Estimator correctness (verified)

All of the following were checked by derivation, not just by reading:

- **GI finalization** `giW = wSum / (M · albedoBar · lum(L̂2w))` is exactly the
  self-normalized importance-sampling form `C = (wsum/M) · f(ŷ)/p̂(ŷ)` with
  integrand `f(ω) = (1/π)·cosθ1·L̂2(ω)` and target
  `p̂ = albedoBar·(1/π)·cosθ1·lum(L̂2)`. The `(1/π)cosθ1` factors cancel
  identically, so the denominator *should* be `albedoBar·lum` — the code is
  right, and the apparent "missing factor" is not a bug.
- **Self-candidate consistency:** with only the fresh candidate,
  `wsum = p̂/q̂ = albedoBar·lum` (q̂ = cosine pdf), `M = 1` → `W = 1`
  exactly, i.e. the GI path reduces to the plain NEE estimator. ✓
- **DI reuse Jacobian** for area lights `J = (d_src/d_new)²·(cosL_new/cosL_src)`
  matches `q_src/q_new` for `q(ω) = d²/(A·cosL)·selectPdf`; spot/directional
  correctly return 1.0 (1/d² folded into the stored target). The
  "do not harmonize the branches" regression note is accurate.
- **M-truncation** scales `wSum` by `mTap/mSrc` consistently in both DI and GI
  tap paths. ✓
- **No feedback loops:** spatial taps read `giHistoryReservoirs` (previous
  frame) only; the *temporal* reservoir saved to the current grid is the
  pre-spatial one. DI has the same discipline. ✓
- **MIS** (DI finalization): balance heuristic against the path's own BSDF
  sampling (`bsdfPdf = (1−specProb)·diffPdf + specProb·specPdf`), applied in
  both ReSTIR and legacy modes; the comment correctly records the +9%
  double-count regression it prevents. ✓
- **Replay visibility** reuses the generic `PackedShadowRay` queue; the shadow
  shader decodes origin/maxDist/dir/pixel/radiance with no GI-specific code —
  clean, and it restores 100%-SIMD traversal for winner replays. ✓
- **rho_max packing** into `tex_flags.w` bits [15:8] collides with nothing:
  bits [7:0] (`matType`) and [31:16] (fp16 normalScale, read via
  `unpackHalf2x16().y` only) are the sole other consumers. ✓
- **Numerics:** no NaN/Inf in 60-frame `--restir` cornell-box render; all 10
  `test_restir_pt_math` tests pass (Jacobian reciprocity, G(a,b) symmetry,
  rho_max quantization bound, ReservoirPT streaming unbiasedness).

---

## 4. Correctness findings (ranked)

### C1 — Glossy-receiver gate is dead code for the diffuse path (High)

`giApplyTap` rejects reconnection when the *receiver* is glossy
(`roughness < 0.15` → `validFootprint = false`), which is the right guard for
a pure-cosine X1 context. But the X1 context is written with a **hardcoded
roughness of 1.0** in the diffuse shader:

```glsl
// wavefront_shade_diffuse.comp:1166
x1Contexts[pixelIndex] = makeX1Context(hitNormal, vec3(diffuseColor), V,
                                       dot(vec3(F0), vec3(0.3333333)), 1.0);
```

The complex shader passes the real `roughness` (line 944). So the gate fires
for complex primaries but never for plain-diffuse primaries — exactly the
surfaces (roughness 0.15–0.5 diffuse with non-negligible F0) where a
cosine-lobe-only context is least valid. **Fix:** pass the actual roughness
in the diffuse write (the variable is in scope).

### C2 — Hit-atom reconstruction ignores instance transforms (High, latent)

`reconstructHitPosition()` fetches `vertexPositions[3·primitiveId + i]` from
the world-space baked position array (binding 40) and **never applies the
TLAS instance transform** — yet `wavefront_intersect.comp` carries
`hitW2o`/`hitO2w` per instance, i.e. the engine's tracing path *does* support
non-identity instances. Any scene using transformed instances would store hit
atoms whose reconstructed positions are in the wrong space, silently biasing
the GI Jacobian, the disocclusion gates, and the replay visibility rays.
All current procedural scenes bake world-space geometry, so this is latent —
but the design doc's invariant ("bit-for-bit exact under rigid or skinned
mesh animation", §3.2) is **false** as written. **Fix options:** (a) assert
identity instances at scene upload when ReSTIR GI is enabled, (b) store the
instance transform in the atom (breaks the 16-byte budget), or (c) document
the restriction prominently in `restir_common.glsl` and the design doc.

### C3 — Sphere hit atoms lose their normal (Medium)

- `wavefront_intersect.comp:211-215` packs sphere hits as
  `instanceId/primitiveId/baryPacked = floatBitsToUint(spPos.xyz)` and
  `flags = 1u` (frontFace bit dropped).
- The `Hit` struct comment claims `baryPacked` holds "oct32 normal for
  spheres" — it does not; it holds `spPos.z` as float bits.
- `reconstructHitPositionAndNormal()` returns `norm = (0,1,0)` for spheres.

Consequence: when a sphere is the winning x2 (e.g. the chrome sphere in
infinity-mirror), `cosRcNew = |dot(n2c, −dirNew)|` in `giApplyTap` uses a
wrong normal → wrong reconnection Jacobian (clamped to [0.33, 3], so bounded
but biased). **Fix:** either decode the true normal (store oct32 in the
spare flag bits is not enough — consider `frontFace`-sign + recompute from
center/radius via a sphere lookup) or reject sphere x2 candidates in
`giApplyTap` (cheapest, and spheres are usually glossy receivers anyway).

### C4 — Legacy NEE candidate count silently reduced (Low, pending)

`wavefront_shade_diffuse.comp:790` (and complex:579):
`M = (enableRestirPT && bounce == 0u && numLights > 1u) ? min(numLights,4) : 1`.
With ReSTIR off, legacy NEE dropped from M=4 to M=1. Both are unbiased (W
normalizes), but M=1 has measurably higher variance. A working-tree fix
(restoring unconditional M=4 for legacy) exists in stash `bisect probes`
alongside debug markers that must **not** be committed — triage the stash.

### C5 — Emissive-on-hit gate for diffuse/complex paths (Low, needs a decision)

`d6fd5ea` added `&& (bounce == 0u || isSpecularPath)` to the emissive
accumulation in the diffuse (line 402) and complex (line 316) shaders. The old
code accumulated `throughput·emissive` on hit unconditionally. Pure-emissive
archetype meshes are unaffected (their pipeline has no gate), so the impact is
limited to *mixed* emissive-diffuse materials at bounce ≥ 1 on diffuse paths —
but it is an undocumented behavior change that darkens exactly those cases.
Either document the rationale (if it was an anti-double-count measure) or
revert to unconditional.

---

## 5. Performance / resource findings

- **P1 — Phase 3 buffers allocated eagerly with no call sites (High).**
  `ReSTIRManager::initBuffers()` allocates, per pixel: PT reservoirs 2×32 B,
  PathSample 160 B, ShadeRequest 32 B, ShadeResult 16 B = **272 B/pixel with
  zero consumers** (553 MB at 1080p; ~1.1 GB at 4K) on top of the live
  DI/GI/X1 grids (144 B/pixel). The lazy-allocation design that kept idle
  VRAM out of the default path is undermined the moment `--restir` is used.
  **Fix:** allocate the Phase 3 buffers only when a Phase 3 feature flag is
  set (or when the first consumer pass exists).
- **P2 — 32 spatial GI taps per pixel. RESOLVED upstream (`c7cd2cf`):** the
  tap loop is capped at 4 (matching DI), mitigating incoherent vertex-fetch
  pressure. The per-frame random rotation still well-mixes the 4-offset
  kernel. If convergence quality regresses, an 8/16-tap A/B against the
  energy/stability test is the follow-up knob.
- **P3 — Stale size comment.** `Engine.cpp:1691` says the lazy allocation
  keeps "~2 x W*H*28 B" out of the idle footprint; the real total is 416
  B/pixel. Update the comment when P1 is fixed.
- **P4 — Inline rayQuery still in the shade kernel (intentional, but the doc
  overclaims).** The GI *self*-candidate shadow test
  (`traceShadowRayInline` at diffuse:637) must stay synchronous (the reservoir
  must store the shadowed L̂2). The design doc's Phase 2b "elimination of
  inline rayQuery from shading kernels" is therefore only true for winner
  replays. Mark Phase 2b as "partial" in the doc.

---

## 6. Documentation / consistency findings

- **D1 — `giTargetAt` header comment contradicts the code.** The comment says
  albedo is *omitted* from the target ("prevents extreme target-ratio jumps
  across high-frequency texture boundaries"), but the function multiplies by
  `albedoBar`. The code is the right choice (consistent with the
  finalization derivation in §3); fix the comment.
- **D2 — Design doc claims `VK_KHR_ray_tracing_position_fetch`.** The
  implementation reconstructs positions manually from the binding-40 position
  buffer. Either adopt the extension (it would also fix C2's transform
  problem) or correct the doc.
- **D3 — Hardware references.** The doc targets "RDNA 4 / gfx1201 / Radeon AI
  PRO R9700" throughout; the project targets AMD Strix Halo (RDNA3.5, UMA).
  The 128 B "vector cache line" alignment arguments should be re-based or
  labeled as forward-looking.
- **D4 — `--restir-m-cap` help text** says default 30; `Config.hpp` default is
  60.
- **D5 — `Hit.baryPacked` comment** ("oct32 normal for spheres") is wrong —
  see C3.
- **D6 — `GI_WEIGHT_CLAMP_MAX = 500.0`** is defined but never used (the live
  clamp is the literal `2.8` in the diffuse shader). Use the constant or
  delete it.
- **D7 — `ptUpdate()` does not maintain `W`** (`weightSum`/`M` are updated,
  `W` is not). No call sites yet, but the struct comment asserts
  `W = w_sum/(M·targetProxy)`; add a "caller must refresh W" note or compute
  it in `ptUpdate`.
- **D8 — `GiReservoir` bit-layout comments** in `ReSTIRTypes.hpp` vs GLSL
  accessors: verified consistent ([8:1] age, [18:9] M, [20:19] hitFlags,
  [28:21] cosSrc). ✓ (no action)

---

## 7. Suggested next steps (priority order, post-`c7cd2cf`)

1. ~~Infinity-mirror fix~~ — **done upstream** (`3892255` + `c7cd2cf`); §2.1
   documents the corrected cause. Re-run the visual integrity suite to close
   it out.
2. **C1:** pass real roughness into `makeX1Context` in the diffuse shader;
   extend `test_restir_pt_math` with a glossy-receiver rejection case.
3. **P1:** defer Phase 3 buffer allocation behind a flag.
4. **C2/C3:** document or guard the Hit-atom reconstruction assumptions
   (identity instances; reject sphere x2 in `giApplyTap`).
5. **C4/C5:** triage the `bisect probes` stash (commit the M restoration,
   decide on the emissive gate, discard the ×50 markers).
6. **D1–D7:** comment/doc cleanup pass (small, single commit).
7. **MGPU coverage:** the ReSTIR grids are now bound on the secondary node
   (`c7cd2cf`), but there is no automated dual-GPU ReSTIR test; add one to the
   integrity suite when a dual-GPU target is available.
