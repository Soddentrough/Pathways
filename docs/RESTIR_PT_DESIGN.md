# ReSTIR PT Design — Path-Space Reservoir Resampling & Decoupled Architecture

**Status:**
- **Phase 1 (Completed 2026-10-10):** Inline ReSTIR DI subsystem, shared `restir_common.glsl`, temporal + spatial-tap reuse for analytical lights (`docs/reports/restir_review_2026_10_10.md`).
- **Phase 2a (Completed 2026-10-11):** Path-space secondary-vertex ReSTIR GI at bounce 1, X1 context buffer, continuous random 2D spatial rotation, target ratio and weight clamping (`commit d5e4fab`).
- **Phase 2b (Active Development):** Decoupled ReSTIR PT — elimination of inline `rayQuery` from shading kernels, stream-compacted visibility queues, and 16-byte `Hit` atom with `VK_KHR_ray_tracing_position_fetch`.
- **Phase 3 (Long-Term Architectural Target):** Fully Inverted / Deferred Gather Path-Database Architecture (pure geometry/visibility resampling with end-of-frame deferred material closure evaluation).

---

## 1. Executive Architectural Summary

ReSTIR PT (*Path-Space Reservoir Spatio-Temporal Importance Resampling*) resamples light transport paths across space and time. However, how the resampling pipeline is structured determines whether the engine achieves real-time frame rates or collapses under GPU hardware penalties.

Two major decoupled architectures exist for production-grade ReSTIR PT in Vulkan 1.4:

1. **Approach A: Request / Result Wavefront Ping-Pong (Lin et al. 2022, GRIS)**
   - **Premise:** Geometry kernels trace rays and evaluate shift mappings; when they reach a reconnected vertex, they emit a lightweight `ShadeRequest`. Specialized material kernels evaluate BSDF $(f, \text{pdf})$ binned by material class, and a resolve pass computes exact Jacobians and MIS weights.
   - **Key Advantage:** Resampling operates on the **exact** target function $\hat{p}$ and true BSDF lobes at every step, maximizing convergence and maintaining mathematical unbiasedness.

2. **Approach B: Fully Inverted / Deferred Gather Path-Database (Bitterli et al., Benty et al. 2023)**
   - **Premise:** Treat the renderer as a **path-sampling database with a deferred material evaluator**. Reservoirs store pure geometric path topology and sampling provenance (seeds, vertices, hints). Resampling, shifting, and visibility-validation operate with **zero material code**, using geometric target proxies. Materials and spectral closures are evaluated **only on the surviving paths at the end of the frame**.
   - **Key Advantage:** Maximum architectural purity and theoretical peak SIMD efficiency. Resampling kernels never load textures, evaluate closures, or ping-pong with material passes.

Below is the complete specification of both paradigms, followed by Pathways' progressive adoption path.

---

## 2. The Core Problem: Why Shading and Geometry Must Never Mix

In a naive or prototype ReSTIR implementation (including Pathways Phase 2a), visibility testing and path shifting are performed inline inside the material shading shader (`wavefront_shade_diffuse.comp`) via `traceShadowRayInline()` (`rayQueryEXT`).

On modern SIMD/SIMT architectures (such as AMD RDNA 4 Wave32 and NVIDIA Ada/Blackwell), this creates three severe, compounding hardware penalties:

### 2.1 SIMD Traversal Lockstep Penalty
- In a 32-lane wave, threads that retain their own candidate skip the inline ray query, while threads selecting a neighbor candidate execute it. The inactive lanes sit idle.
- For active lanes, every thread traces toward a different 3D secondary vertex in arbitrary directions. Lane 0 may hit an occluder on step 1, while Lane 1 traverses 40 BVH nodes deep into a mesh. **The entire 32-lane wave is stalled until the single slowest lane finishes traversal.**

### 2.2 VGPR Pressure and Occupancy Collapse
- Shading kernels require significant Vector General-Purpose Registers (VGPRs) for BSDF evaluation, VNDF sampling, textures, PRNG state, and reservoir arithmetic.
- Incorporating `rayQueryEXT` forces the shader compiler to allocate traversal state registers, ray descriptor registers, and stack scratch space inside the same shader.
- On RDNA 4 (`gfx1201`), exceeding register limits drastically cuts occupancy (waves per Compute Unit), crippling the GPU's ability to hide memory latency during texture and reservoir fetches.

### 2.3 L0 / L1 Vector Cache Thrashing
- Dedicated ray traversal kernels fetch only BVH nodes and triangle descriptors into cache lines.
- Mixing traversal into shading forces the cache hierarchy to juggle BVH bounding boxes and triangle data alongside material properties, normal maps, and albedo textures, causing severe cache line evictions.

**Rule:** *Tracing and shifting are geometry problems; shading is a material problem. The two never live in the same kernel.*

---

## 3. Long-Term Architecture: Fully Inverted / Deferred Gather (Approach B)

The Fully Inverted architecture is the highest-performance, most complete long-term paradigm for ReSTIR PT. It treats the path tracer as a **geometric path database**:

```
[1. Primary VisBuffer / G-Buffer]
       ↓
[2. Seed Path Generation] ───────────────► Emits raw geometric PathSkeletons (no BSDF/colors)
       ↓
[3. Initial Reservoir Construction (R0)]
       ↓
[4. Temporal Path Resampling] ───────────► Reconnection / Shift logic (pure geometry)
       ↓
[5. Spatial Path Resampling] ────────────► Reconnection / Shift logic (pure geometry)
       ↓
[6. Batch Visibility Verification] ──────► 100% SIMD-utilized rayQuery / traversal pass
       ↓
[7. MIS & Reservoir Combination] ────────► Pure ALU resolve (Jacobian |J| + geometric proxy)
       ↓
[8. Path Reconstruction] ────────────────► Extract winning path keys per pixel
       ↓
[9. Deferred Material Evaluation] ───────► Evaluate BSDF closures, textures, emission
       ↓
[10. Final Accumulation & Denoiser]
```

### 3.1 Separation of Concerns Matrix

| System Layer | Question Answered | Knows Materials? | Uses Ray Tracing? |
|---|---|---|---|
| **Layer 1: Path Topology & Sampling** | Which geometric subpaths exist and how were they sampled? | No (Geometry + Seeds only) | Yes (Skeleton tracing) |
| **Layer 2: Shift & Visibility Validation** | Can a shifted path connect validly without occlusion? | No (Topology + Sidedness only) | Yes (Visibility-only rays) |
| **Layer 3: Material / BSDF Shading** | What spectral radiance $f / p$ does the transport produce? | **Yes** (Closures, textures, BRDFs) | **No** (Pure compute gather) |

### 3.2 Data Structures for Approach B

#### Hit Record (The 16-byte Atomic Currency)
```glsl
struct Hit {            // 16 bytes
    uint instanceId;    // TLAS instance custom index
    uint primitiveId;   // Triangle index in BLAS geometry
    uint baryPacked;    // 2 x unorm16 (u, v barycentrics)
    uint flags;         // Front/back face, isLight, alphaTested, isDelta hint
};
```
*Note:* World positions and normals are **never** stored in the reservoir. They are reconstructed on demand in whatever kernel needs them using `rayQueryGetIntersectionTriangleVertexPositionsEXT` or BDA vertex streaming. This guarantees that temporal shifts remain bit-for-bit exact even under dynamic rigid or skinned mesh animation.

#### Path Vertex & Path Sample
```glsl
struct PathVertex {
    Hit   hit;
    uint  woOct;         // Outgoing direction towards previous vertex (octahedral 32-bit)
    uint  materialId;    // Integer material ID (for deferred stage routing only)
    uint  vertexType;    // CAMERA, SURFACE, LIGHT, SPECULAR_DELTA
};

struct PathSample {              // Stored as Structure of Arrays (SoA)
    uint       vertexCount;      // Effective length of subpath (e.g. 2 to 6)
    PathVertex vertices[6];      // Compact fixed-capacity prefix/suffix
    uint       rngSeedPrefix;    // Replay seed for vertices before reconnection
    uint       rngSeedSuffix;    // Replay seed for vertices after reconnection
    uint       strategyId;       // Sampling provenance (unidirectional, NEE, BSDF)
};
```

#### ReSTIR PT Reservoir (SoA Layout)
```glsl
struct ReservoirPT {             // Packed SoA streams
    float weightSum;             // Sum of resample weights
    float W;                     // Unbiased contribution weight
    float M;                     // Confidence counter (capped)
    float targetProxy;           // Cached geometric target value
    uint  pathIndex;             // Index into PathSample SoA buffer
    uint  lastFrameAge;          // Age tracking for temporal expiry
};
```

### 3.3 Decoupled Target Function: Geometric Proxies
Because full BSDFs and textures are not evaluated during resampling, the target function $\hat{p}(X)$ is computed via **conservative geometric upper bounds**:
$$\hat{p}(X) \approx \prod_{i=1}^{k} \rho_{\text{max}}(\text{material}_i) \cdot G(v_{i-1}, v_i) \cdot L_{\text{emit-bound}}(v_k)$$
where:
- $\rho_{\text{max}}$ is a precomputed 8-bit scalar upper bound of the material's hemispherical reflectance.
- $G(v_{i-1}, v_i) = \frac{|\cos\theta_{i-1}| \cdot |\cos\theta_i|}{\|v_{i-1} - v_i\|^2}$ is the geometric coupling factor.
- $L_{\text{emit-bound}}$ is the flux or bounding emission of the terminal light/emitter.

This decouples the entire resampling loop from bindless texture fetches, material branching, and closure evaluation.

---

## 4. Intermediate Architecture: Request / Result Wavefront (Approach A)

While Approach B represents the ultimate theoretical ceiling, **Approach A (Lin et al. 2022 / GRIS)** provides an immediate, pragmatic transition because it matches Pathways' existing **material-sorted DGC pipelines**:

```
[Wavefront Bounce Loops] ──► Writes Initial PathSample (rcVertex, seeds, throughput)
                                    │
                                    ▼
[Shift-Trace Kernel] ──────► Traces Reconnection Visibility (x_{k-1} ↔ rcVertex)
                             Emits ShadeRequests to transient queue:
                             struct ShadeRequest { Hit hit; uint wi, wo; uint slot; uint matId; }
                                    │
                                    ▼
[DGC Material Pipelines] ──► Consumes ShadeRequests binned by materialId
                             Evaluates true BSDF f and pdf (zero ray queries!)
                             Emits ShadeResult: { f16vec3 f; float16_t pdfFwd, pdfRev; }
                                    │
                                    ▼
[ReSTIR PT Resolve] ───────► Evaluates exact Jacobian |J| & pairwise MIS
                             Updates reservoirs: w_i = m_i · p̂ · W · |J|
```

### Key Advantage for Pathways
Pathways already possesses:
- `m_materialIndexQueue` and material sorting logic.
- Separate `shadeDiffuse`, `shadeComplex`, `shadeDielectric`, and `shadeEmissive` pipelines.
- DGC (`vkCmdExecuteGeneratedCommandsEXT`) execution sets for indirect material execution.

Instead of writing a completely new material system, Approach A reuses Pathways' existing material pipelines to evaluate `ShadeRequest` records, achieving zero ray divergence with exact BSDF targets.

---

## 5. Vulkan 1.4 Native Capabilities & Implementation Rules

Both architectures rely on standard, cross-vendor Vulkan 1.4 features:

1. **`VK_KHR_ray_query`**: Primary traversal engine inside compute shaders.
2. **`VK_KHR_ray_tracing_position_fetch`**: Reconstructs world-space triangle positions directly from `(instanceId, primitiveId, bary)` without touching vertex buffers.
3. **Buffer Device Address (BDA) & `buffer_reference`**: Accesses SoA reservoir buffers and request queues without descriptor set churn.
4. **`float_controls2` & Strict Determinism**:
   - Compiling all shift, trace, and resolve shaders with `SignedZeroInfNanPreserve` and uniform FMA contraction rules to ensure that geometric Jacobians and reconnection tests match bit-for-bit.
5. **Synchronization 2 (`VK_KHR_synchronization2`)**: Global compute-to-compute buffer memory barriers (`COMPUTE_SHADER → COMPUTE_SHADER`) gating each pass.

---

## 6. Phased Implementation Roadmap for Pathways

### Phase 2b.1 (Immediate): Reconnection Visibility Decoupling
- **Goal:** Eliminate `traceShadowRayInline` and `rayQueryEXT` from [`shaders/compute/wavefront_shade_diffuse.comp`](file:///home/naoki/Development/Pathways/shaders/compute/wavefront_shade_diffuse.comp).
- **Implementation:**
  - When `!giShadeWinnerSelf`, append a reconnection ray to a dedicated stream-compacted visibility queue (or extend `PackedShadowRay`).
  - Execute traversal in a pure shadow/visibility compute kernel.
  - Accumulate winner radiance only for unoccluded rays.
  - **Outcome:** Shading kernel VGPR usage drops significantly, wave occupancy doubles, and SIMD lockstep stalls are completely eliminated.

### Phase 2b.2: 16-Byte `Hit` Atom & Position Fetch
- Replace the 32-byte world-position representation in `GiReservoir` with the 16-byte `Hit` struct.
- Reconstruct $x_2$ on demand via `VK_KHR_ray_tracing_position_fetch` or BDA.
- **Outcome:** Eliminates geometric drift on animated geometry; cuts reservoir bandwidth in half.

### Phase 2b.3: Multi-Bounce Shift-Trace & Material Request Queues (Approach A)
- Implement `ShadeRequest` and `ShadeResult` transient buffers in `ReSTIRManager`.
- Dispatch Shift-Trace $\to$ DGC Material Pipelines $\to$ ReSTIR Resolve.
- Implement exact geometric Jacobian $|J|$:
  $$\|J\| = \frac{\cos\theta_{\text{rc}}'}{\cos\theta_{\text{rc}}} \cdot \frac{\|x_{k-1} - x_{\text{rc}}\|^2}{\|x_{k-1}' - x_{\text{rc}}\|^2}$$
- **Outcome:** Full multi-bounce ReSTIR PT with exact BSDF targets and pairwise MIS.

### Phase 3: Fully Inverted Path-Database Architecture (Approach B)
- Transition reservoir storage to pure `PathSample` SoA records with RNG seeds and topology hashes.
- Replace mid-frame material evaluation with geometric target proxies $\hat{p}$.
- Defer all material BSDF closure evaluation to an end-of-frame gather pass.
- **Outcome:** Ultimate theoretical ceiling of performance and architectural purity.
