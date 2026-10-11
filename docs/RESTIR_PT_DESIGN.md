# ReSTIR PT Design — Path-Space Reservoir Resampling with Replay

**Status:** Phase 2a IMPLEMENTED (2026-10-11): single-vertex GI reservoir with X1 context,
winner replay visibility, and bounce-1 estimator replacement. Phase 2b (multi-vertex path
replay) remains planned. The Phase 1 foundation — the inline ReSTIR DI pipeline, the
shared `restir_common.glsl` core, history-tap reuse, and the honest naming — landed on
2026-10-10 (`docs/reports/restir_review_2026_10_10.md`).

### Phase 2a as built (deviations from the plan below)

- **Scope:** GI reservoirs apply at bounce 1 only, for paths whose continuation at the
  primary hit was a diffuse cosine REFLECTION lobe (transmitted and specular lobes write
  an invalid X1 context and keep legacy shading). x2 on complex/dielectric/emissive
  surfaces invalidates the GI slot (no candidate).
- **X1Context is 24 B** (n1 oct32, fp16 albedo, oct32 wo, fp16 scalar F0, valid flag).
  `wo`/`F0` are needed because the bounce-0 continuation folds `(1-F)·albedo/(1-specProb)`
  into the carried throughput; the finalization therefore multiplies the winner by the
  Fresnel ratio `(1-F_w)/(1-F̂)` to stay unbiased (without it: ~+9% on cornell-box).
- **GiReservoir is 32 B**: x2 world position fp32×3, wSum, p̂, fp16 L̂2, fp16 d_src, and
  `flagsM = valid | age[8:1] | M[24:9]`. The tap weight keeps the `(d_src/d_new)²`
  solid-angle Jacobian; `q` itself is not stored (cosine lobe ⇒ recoverable).
- **L̂2 is stored SHADOWED** (inline shadow x2→light traced in the same pass; the deferred
  shadow queue cannot feed the reservoir). The winner replay ray covers x1↔x2 only when
  the winner is a reused tap (self-winner needs no ray: the path proved visibility).
- **Finalization identity:** `C2 = T2·L̂2_w · wsum/(M·albedoBar·l̄um_w) · (1-F_w)/(1-F̂)`;
  with a single candidate this reduces exactly to the legacy `T2·L̂2` accumulation
  (verified: frame-1 images bit-comparable, 512-frame energy within 1.7% of plain PT).
- **Activation:** UBO flag bit 24 (`restir_gi`, implied by `--restir`, off via
  `--no-restir-gi`) AND material-sort secondary pipelines AND hardware ray queries.
  The unified (non-material-sort) pipeline never runs GI — grids stay invalid there.
- **Bindings:** 37 = X1 contexts (single-buffered, same-frame producer/consumer),
  38 = GI current, 39 = GI history (double-buffered like DI).
**Goal:** Resample *indirect* illumination (bounce ≥ 1) with the same reservoir machinery
that already serves direct light, targeting the 1-SPP real-time goal on Strix Halo and the
Veach-Ajar stress scenes.

---

## 1. Why the old `PathReservoirPT` scaffolding was deleted

The purged struct implied that path-space reuse was one struct away. It was not. Correct
GI reuse requires three things the wavefront did not have:

1. **The BRDF context of the reuse vertex x1** at the time reuse is evaluated. In a
   material-sorted wavefront, the bounce-1 shade pass knows x2 (its hit) but has lost x1's
   material, albedo, and outgoing BSDF pdf. Reconnecting a neighbor's candidate *without*
   re-evaluating `f_r(x1, ωo, ω2')` is not reuse — it is stale caching.
2. **A consistent estimator decomposition.** If a reused candidate supplies the 2-bounce
   contribution, the path's own pixel contribution for that same bounce must be *replaced*,
   not added, or the 2-bounce term is double-counted.
3. **Path replay for deeper bounces.** If the reservoir winner is not the path's own
   candidate, the continuation beyond x2 must originate from the *winner's* x2', which
   requires sampling the BSDF at x2' — i.e. its material — which the wavefront cannot do
   from the reservoir alone.

## 2. Architecture

### 2.1 X1-Context buffer (new, per pixel, written at bounce 0)

The bounce-0 diffuse/complex shade pass already knows everything the reuse vertex needs.
It writes one record per pixel per frame alongside the GI reservoir:

```
struct X1Context {              // 32 B
    vec3  position;             // x1
    vec3  normal;               // n1 (shading normal)
    vec3  albedo;               // diffuse color
    vec3  wo;                   // -primary ray direction (toward camera)
    float roughness;
    float metallic;
    uint  packed;               // fp16 F0.x proxy, valid bit, frame stamp
    float pad;
};
```

Cost: 32 B/px/frame write + random reads at bounce 1. On UMA (gfx1151) this is L2-friendly
since taps are spatially local.

### 2.2 GI reservoir (new, per pixel, double-buffered like DI)

```
struct GiReservoir {            // 32 B
    uint  idxM;                 // [15:0] candidate tag, [31:16] M
    float wSum;
    float targetPdf;            // p̂ at storing pixel (scalar, solid-angle measure at x1)
    uint  flags;                // [7:0] age, [8] valid, [9] lobe, [25:10] oct16(x2 normal)
    float x2Dist;               // |x2 - x1| at storage time
    uint  dirPacked;            // oct16(ω2) — direction x1 -> x2
    uint  radRG;                // fp16x2  L̂2 (unshadowed direct light + emissive at x2, previous frame)
    uint  radB_misc;            // fp16 B + fp16 cos(theta at x2, stored config)
};
```

`L̂2` is the **previous frame's** unshadowed direct radiance computed at x2 during bounce-1
shading (exactly the `secDirectL` the DI stage already computes there — no new probe pass).

### 2.3 Estimator (bounce-1 shade, diffuse x1 paths only, initially)

For the ray currently hitting x2 with origin x1 and direction ω2:

- **Fresh candidate:** `(ω2, |x2-x1|, n2, L̂2 := unshadowed direct at x2 computed this pass)`.
  Source pdf `q(ω2)` = the cosine-lobe pdf used at bounce 0 (`cosθ1/π`, recorded in the
  X1-context at generation time so it survives the queue reshuffle).
- **Target:** `p̂(x1, candidate) = max(lum(albedo·L̂2/π), ε) · cosθ1` — computable for *any*
  candidate from the X1-context + reservoir fields.
- **Taps:** same 1 temporal + 4 spatial rotated-cross pattern as DI, reading the previous
  frame's GI grid. Reconnection keeps the candidate's **x2 fixed** and re-aims from the
  current x1: `ω2' = (x2 - x1_new)/d'`, `d' = |x2 - x1_new|`, with
  - geometry gates: `cosθ1' > 0`, `cosθ2' > 0`, distance-ratio and x1-normal/depth gates
    (same constants as DI),
  - measure factor: solid-angle `q_src/q_curr` (the p̂ above is per-solid-angle at the
    vertex; radiance is ray-constant so no 1/d² appears in p̂ — the q-ratio carries it,
    analogous to the DI area-light case),
  - **replay visibility:** one inline ray query x1 → x2' (`traceShadowRayInline` already
    exists in the shade passes); a blocked candidate contributes weight 0 and is dropped.
- **Finalization (the double-count fix):** the winning candidate's contribution
  `f_r(x1, ωo, ω2') · cosθ1' · L̂2 · W` **replaces** this ray's usual pixel accumulation of
  `T1 · direct(x2)`. The path still continues to bounce 2+ as today, contributing only
  3+-bounce transport; its own 2-bounce pixel term is emitted solely through the reservoir
  estimator. To keep the fresh candidate's L̂2 identical to what the path would have added,
  the winner's L̂2 for the *self* candidate is the full direct+emissive term of this pass.

### 2.4 Known bias sources (must be documented in-code, like DI §4.3)

- `L̂2` is one frame stale → lag on animated lights/materials (bounded by max-age truncation).
- Deeper-than-2-bounce transport of *reused* candidates is lost (their paths are not
  replayed beyond x2); mitigated by keeping the self-path continuation as the unbiased
  3+-bounce estimator.
- M-truncation caps (temporal/spatial), geometry gates, replay-visibility rejection
  weighting.
- Specular x1 lobes are excluded initially (q is undefined for them from the context
  record); they fall back to plain path continuation.

## 3. Implementation sequence

1. **X1-context buffer**: struct, allocation (lazy, same pattern as DI reservoirs),
   binding (next free slot in the wavefront descriptor layout), bounce-0 writes.
2. **GI reservoir grid**: extend `ReSTIRManager` to own a second pair of buffers;
   zero-fill on (re)alloc (same `recordClearIfNeeded` pattern).
3. **Bounce-1 reuse block** in `wavefront_shade_diffuse.comp` (secondary path section)
   calling new `restirGi*` functions in `restir_common.glsl` — mirroring the DI function
   shapes (`restirGiFreshCandidate`, `restirGiApplyTap`).
4. **Estimator switch**: gate the replacement of the bounce-1 pixel accumulation behind
   the ReSTIR flag; A/B validate energy conservation on Cornell Box (2-bounce reference).
5. **Validation**: extend `scripts/verify_visual_integrity.py` with a GI variance-reduction
   + bias gate (many-lights + cornell-box vs. high-SPP reference images); add the
   M/W debug view for both DI and GI.
   *(Done 2026-10-11: `test_restir_gi_energy_and_stability` gates energy + fireflies; the
   M/W/light/coverage debug views ship as `--debug-view` + GUI combo.)*

## 4. Explicit non-goals for Phase 2

- Full multi-vertex path replay (needs per-sample path storage; separate milestone).
  Phase 2a deliberately does NOT redirect the bounce-2 continuation toward the reservoir
  winner: the continuation always follows the path's own x2, so deeper bounces stay
  unbiased and no path storage is needed. Winner-consistent continuation is Phase 2b.
- ReSTIR for conductors/dielectrics (their shade passes keep their 1-stage RIS).
- Light trees for GI candidate proposals (reuse the alias table first).
- GI on transmitted diffuse lobes or on complex-material x2 surfaces (context invalid →
  legacy shading).
