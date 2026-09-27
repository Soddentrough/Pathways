Ideas...

# Expert Review: NRC on RDNA 4 — Answers by Section

Before the questions: two of your five failures (WG0-only weight updates, direct-light-only targets) mean **you have not yet evaluated NRC — you've evaluated a broken approximation of it**. The 18% regression and blotchiness are real, but the quality conclusions are contaminated. Fix Attempts 2 and 4 before deciding the paradigm question in Section C. Also, delete the orphaned 12.58 MB hash buffer — it's not just wasted VRAM; the host mapping and barriers pollute your synchronization graph and may serialize passes that could overlap.

---

## SECTION A: Algorithmic Foundations

### A1. Routing multi-bounce radiance back to training records

You don't need path history buffers or global sync. The original NRC design (Müller et al. 2021) solves this two ways; both map cleanly to wavefront tracing:

**Option 1 — Self-training bootstrap (recommended, and what the paper actually does).** Training paths do *not* trace to full termination. They trace a short suffix (2–4 extra bounces) and terminate *into the cache itself*: the target is
```
target = Σ (throughput_k · L_direct_k) + throughput_terminal · NRC(x_term, ω_term)
```
This turns multi-bounce propagation into a fixed-point iteration (analogous to temporal-difference learning) — the cache bootstraps its own multi-bounce estimate over frames. This eliminates the need for downstream bounces to "find" their record: the suffix is short and bounded.

**Option 2 — Scatter-accumulation via record index.** Carry a 4-byte `trainRecordIdx` plus current RGB throughput in the ray payload (you already carry throughput). At every downstream shading event, each training ray does one fixed-point atomic add of `throughput · L_emitted_or_NEE` into `trainingTargets[trainRecordIdx]`. Your existing pass boundaries are the only synchronization needed — targets are complete when the wavefront loop drains, before the training dispatch. At 3% of rays this is trivial bandwidth (~60K paths × 12 B × a few bounces).

The best design is Option 2's plumbing with Option 1's termination rule. Also store per-vertex records for *every* vertex of the training suffix (paper's trick): one training path yields N training samples, multiplying your effective training set ~4× for free.

### A2. Is online self-training viable at 1 SPP without a denoiser?

**No — and it was never designed to be.** NRC reduces the variance of the *path tail* (bounces 3+); the dominant noise at 1 SPP is direct lighting and first-bounce indirect, which NRC doesn't touch. The published NRC results all sit behind temporal accumulation/denoising.

What *is* fixable is your adaptation-speed complaint, which is currently an artifact of the batch-size-32 bug. With that fixed:

- **Exponential moving average of network weights** for inference (paper uses this): train weights θ, infer with θ_EMA. Kills temporal flicker during adaptation.
- **Relative L2 loss** (normalize by predicted luminance² + ε): prevents bright pixels from dominating gradients — essential for HDR stability.
- **Confidence-gated blending during cold start**: track running training loss; if loss > threshold (camera cut, newly revealed geometry), fall back to traced continuation for a larger fraction of paths and down-weight the cache.

Even then: budget for a denoiser. The correct mental model is *NRC + denoiser vs. pure MC + denoiser*, where NRC's contribution is a dramatically less noisy denoiser input for the GI tail.

### A3. Lobe segregation and energy conservation

Don't use a pure roughness threshold — use the **path-spread heuristic** from the paper: terminate into the cache when the accumulated ray footprint (approximated from BSDF sampling PDFs along the path) exceeds a multiple of the cache's effective spatial resolution at the first vertex. This naturally lets mirrors and glass propagate physically (their footprint stays small) while diffuse and rough-specular lobes terminate quickly. A practical hybrid: hard-deny cache queries for roughness < ~0.2 or any transmission event, use the spread heuristic above that.

Energy conservation rules:

1. The cache predicts **full outgoing radiance L_o(x, ω)** at the query vertex, including direct lighting *at that vertex*. Therefore: **do not also evaluate NEE at the cutoff vertex** for cache-terminated paths, or you double-count. (This interacts with your Attempt 4 bug — your targets must include direct + indirect at the terminal vertex, consistently with what queries assume.)
2. **Demodulate albedo**: train on `L_o / albedo` (with ε-clamped albedo), re-modulate at query time. This removes texture detail from the network's burden, improves convergence ~an order of magnitude, and guarantees the cache can't emit energy on black surfaces.
3. The blend with specular is just path splitting: cache handles the diffuse/rough lobe's continuation weighted by its sampling probability; specular continuation carries `1 - P(diffuse)` throughput physically. No additional normalization needed if the lobe selection PDF is respected.

---

## SECTION B: GPU Architecture

### B4. Killing the 80-byte query record traffic

Three-pronged attack:

**Compress the record to 32 bytes (two `uvec4` stores):**
- Position: 3× fp32 (12 B) — or 3× fp16 relative to a per-tile anchor if scene scale permits
- Direction + normal: octahedral-encoded, 2× snorm16 each (8 B total)
- Albedo: R9G9B9E5 or 10:10:10:2 (4 B)
- Roughness + flags: 2 B; pixel index: 4 B; padding: 2 B

Two aligned 16-byte stores are coalesced-friendly on RDNA 4 and cut traffic to ~64 MB/frame at 2M rays — but more importantly:

**Wave-cooperative compaction instead of per-thread atomics.** Per-thread `atomicAdd` on the queue counter is your serialization point. Use `subgroupBallot` on the "emits query" predicate, `subgroupBallotBitCount` for the wave's total, one `atomicAdd` per wave (32× contention reduction), then `subgroupBallotExclusiveBitCount` for intra-wave offsets. Stage records in LDS and burst-write per wave for fully coalesced 128-byte transactions.

**Question the queue's existence at high coherence.** At 1 SPP with mostly diffuse scenes, nearly every pixel emits a query — compaction buys nothing, and a *dense* per-pixel record buffer (visibility-buffer style, fixed offsets, perfectly coalesced, no atomics at all) is strictly faster. Use dense layout when the previous frame's query rate exceeded ~60%, compacted queue below. Your 8-SPP overflow bug (Failure A) also disappears with the dense layout + per-pixel sample loop, or by sizing the queue as `W × H × SPP` with your CRIT-04 fallback retained.

### B5. Mini-batch Adam reduction topology

**Hybrid two-level: LDS reduction intra-workgroup, fp32 global atomics inter-workgroup, then a tiny Adam kernel.** Neither pure option is right:

- Pure two-pass (partials buffer + reduction dispatch) costs an extra dispatch + barrier and writes `#weights × #workgroups` partials — for a 4×64-wide MLP (~17K weights) × 256 WGs that's 17 MB of traffic. Wasteful.
- Pure per-thread global atomics create pathological contention (thousands of threads hammering the same weight's gradient).

Correct topology:
1. Each thread accumulates its local gradient contributions in registers (fp32; if activations/WMMA are fp16, apply loss scaling ~128–1024 and unscale at accumulation).
2. `subgroupAdd` within each Wave32, then cross-subgroup reduction through LDS → one gradient vector per workgroup.
3. One `atomicAdd` (fp32, `VK_EXT_shader_atomic_float` — supported on RDNA 4) per weight per *workgroup* into a global gradient buffer. Contention = 256 workgroups max per weight, and a 68 KB gradient buffer lives entirely in L2. This is fast.
4. A separate tiny dispatch (`ceil(numWeights/256)` workgroups) applies Adam: fp32 master weights, fp32 m/v moments, then quantizes weights to fp16 for the WMMA inference/forward buffers.

This kills both the WG0 pathology and scratch spills, and gives you true 4K–16K batches. Non-determinism from float atomics is irrelevant for SGD. Keep **master weights in fp32** — fp16 Adam moments will stall convergence.

### B6. Indirect dispatch vs. static full-screen dispatch

`vkCmdDispatchIndirect` eliminates the *empty-workgroup* cost (launching 500K WGs to retire immediately still burns command-processor and SPI launch throughput), and yes, you should do it. Details:

- The indirect args must contain **workgroup counts**, not thread counts. Do the ceil-divide in the same shader that finalizes the counter (one thread), or in a 1-thread fixup dispatch — don't read a raw ray count.
- The DISPATCH_INDIRECT packet read on AMD's command processor is cheap (~µs); small indirect dispatches themselves are not the problem. **The real bubble is the barrier**: `VK_ACCESS_INDIRECT_COMMAND_READ_BIT` dependency on the counter write typically compiles to a pipeline drain between the queue-fill pass and inference. Mitigate by (a) overlapping the gap with independent work (your resolve pass, denoiser history reprojection) on the same queue, or (b) running training on the **async compute queue** — it has no dependency on the current frame's queries if you train on last frame's records (one frame of training latency is harmless for a cache).
- Alternative that sidesteps indirect entirely: **persistent-threads inference** — launch a fixed grid (`#WGPs × waves-per-WGP`), each wave loops popping 32-query tiles off the counter until empty. No indirect args, no fixup pass, ideal WMMA utilization.

Separately: 1.25 ms for inference is too slow regardless of dispatch shape. A 64-wide, 4–5 hidden layer MLP at ~2M queries should run ~0.3–0.5 ms on an R9700 if all layers are fused into one kernel with weights resident in LDS and activations never touching VRAM. If you're dispatching per-layer, that's your first fix.

---

## SECTION C: Architectural Paradigms

### C7. Per-scene pre-baked cache

Viable and underrated — but you have a better option **because you have two GPUs**. The killer NRC deployment on your hardware is:

**Asynchronous cross-GPU training.** GPU 0 renders and runs frozen-weight inference only (~0.3–0.5 ms once optimized, zero training cost in the frame). GPU 1 continuously trains on training records streamed over PCIe (60K records × 32 B ≈ 2 MB/frame; weights back are ~35–70 KB). Update inference weights every N frames with EMA smoothing. This gives you *continuous* adaptation to dynamic lights and objects (with 3–10 frames of latency, visually indistinguishable for diffuse GI) while removing training entirely from the render GPU's frame budget. It converts your 18% regression into a probable net win.

Pure pre-bake trade-offs, if you go that route instead: frozen weights = baked-lightmap semantics. Moving emitters, doors opening, TOD changes all go stale; dynamic objects receive stale irradiance and *contribute nothing* to GI. Mitigations: retrain triggers on light-transform hashes, blend-over-N-frames weight swaps, or conditioning inputs (sun angle) trained across the parameter range at bake time. Acceptable for archviz/static walkthroughs; wrong for gameplay.

### C8. Zero-shot pre-trained coordinate model

**Not feasible as posed, and this is fundamental, not an engineering gap.** Outgoing radiance at (x, ω) is a global functional of the entire scene — geometry, materials, emitters, visibility. A coordinate-only MLP with frozen weights has zero mutual information with an unseen glTF's light transport; two scenes can share identical local inputs (position, normal, albedo, roughness) and have arbitrarily different radiance. The per-scene online/baked training *is* the scene conditioning.

What is feasible in the zero-shot regime: **image/screen-space operators** — networks conditioned on G-buffers and noisy radiance that predict/complete indirect illumination (deep-shading-style GI prediction, neural denoisers). These generalize because their input already encodes scene-specific transport samples. Scene-conditioned transformers over geometry/probe tokens are an active research direction but are nowhere near your latency budget. Practically: zero-shot lives in image space, not coordinate space.

### C9. Pivot to ReSTIR GI + neural denoiser?

For a 1-SPP interactive viewport, **yes for shipping quality — but the framing "either/or" is wrong; production pipelines use both.**

- **ReSTIR GI** excels exactly where NRC is weakest: the first indirect bounce, which carries most GI energy and all sharp detail (contact shadows, glossy interreflection). It's zero-training, temporally stable under motion (reservoirs reproject naturally — solving your camera-motion complaint), and reuses your ReSTIR DI infrastructure. Memory: ~32–48 B/pixel of reservoirs + temporal history, comparable to your query traffic, without the training pipeline.
- **NRC** excels exactly where ReSTIR GI is weakest: the bounce-3+ tail, which is low-frequency, low-energy, and expensive to sample. This is why Cyberpunk-class pipelines pair ReSTIR DI/GI with a radiance cache for tail termination.
- A spatio-temporal denoiser is required in **both** designs at 1 SPP; it is not a differentiator.

**Recommended sequencing:** (1) Ship ReSTIR GI + denoiser as the GI foundation — it fixes your motion-quality problem with proven, tuning-light machinery. (2) Fix the two NRC correctness bugs (A1 targets, B5 reduction) and re-benchmark NRC *purely as a tail cache at the ReSTIR GI continuation vertex*, with cross-GPU async training per C7. In that role its query volume drops by an order of magnitude (only continuation rays, not full viewport), its 0.42 ms queue and 1.25 ms inference costs shrink proportionally, and your Attempt 5 comparison becomes fair: NRC's break-even point is bounces 3–16 amortized, not bounce 3 alone.

----

# Expert Analysis: Neural Radiance Caching on RDNA 4 

This is a systems-level diagnosis. The majority of your regression is not a neural network capacity problem, but a *data movement and synchronization problem*. For bounce depths of 2–4, physically traversing the BVH is cheaper on modern ray tracing GPUs than materializing, serializing, and scheduling a GPU-side learning workload. 

## SECTION A: Algorithmic Foundations and Supervision

### A.1 Routing multi-bounce ground truth back to earlier bounce training records

The core supervision bug (training on `secDirectL` only) is the standard difficulty of NRC: the "target" for a query at bounce 2 is not known at bounce 2. In a streaming wavefront path tracer, shading at different path depths executes in different compute dispatches, compacted against different ray queues, so there is no parent pointer available on the GPU by default.

Do not store full path histories. Store a small reverse link between a continuation path and its originating NRC query.

#### Recommended approach: Query ID + a training payload buffer

1. When emitting an NRC query at bounce 2, assign it a monotonic `queryId` and write an `NRCTrainingRecord` into a GPU buffer indexed by `queryId` (containing `position`, `normal`, `viewDir` or `incomingDirection`, `roughness`, `baseColor`, `albedo`, `materialMask`, `throughputPath` initialized to the path throughput up to and including bounce 2, and `targetRadiance = (0,0,0)`, `accumWeight = 0`).
2. Also write the query payload for inference into a separate queue. The path that was stochastically selected for "neural supervision" (the 3% of rays) carries `queryId` and `trainThroughput = pathThroughput` with it down the wavefront in its ray payload (not in a global structure).
3. When that path terminates (miss, Russian roulette, or a maximum bounce), add `trainThroughput * Le_terminal_or_accumulated` to `trainingRecords[queryId].targetRadiance` using a 64-bit or floating point atomic add. If the path continues, multiply `trainThroughput` by the BSDF * cosine / pdf of the sampled direction and propagate `queryId` to its child rays.
 
This is known in production implementations of NRC (NVIDIA's Neural Radiance Cache for path tracing) as "path continuation attribution" or "query attribution". The memory overhead is O(number of in-flight NRC queries), not O(number of rays in flight). No per-pixel path ring buffer is necessary.

Two important details:

- **Use atomic adds to the target.** Multiple continuation samples are almost never generated for a single query, however, if you ever use multiple training continuations or Russian roulette splitting changes, this still holds. For a single continuation path, atomics are still cheap and race-free. 
- **Do not blend direct lighting into the target.** The NRC is meant to predict the *incoming radiance from bounces >= cutoff+1 integrated along the diffuse lobe* (the indirect tail). The renderer should add direct lighting, emissives at the shading point, and specular lobe contributions physically. Training the cache to predict direct lighting doubles or triples the learning task and directly creates the "looks like local illumination" artifact that you observed.
- **Track throughput, not radiance.** Propagate the path throughput `beta` with the `queryId`. The unbiased ground truth is $L_{tail} = \mathbb{E}[\beta_{>=cutoff} \cdot L_e(\text{path end})]$. Storing `beta * L$ at termination and accumulating into the record yields that value.

This removes the need for a global barrier between bounce passes. Attribution happens at path termination time, which already occurs inside your shading/closest-hit miss wavefronts.

### A.2 Is online 1-SPP adaptation viable during camera navigation?

Not in its current "from scratch per frame" streaming form for a cold or rapidly changing view. It is viable only under strict constraints.

The optimization problem is non-stationary. Every camera translation, rotation, object transformation, light toggle, or visibility change invalidates a large portion of the 5D (position, normal, direction, and material) domain. A first-order optimizer such as Adam with a small replay of 1k–8k samples cannot "track" high-frequency view changes at 1 SPP. This matches your observation of blotchiness and slow adaptation.

There are three practical regimes:

| Strategy | Adaptation | Stability | Dynamic Lights/Objects | Notes |
|---|---|---|---|---|
| Train-from-scratch online (your current design) | Slow at 1 SPP | Low on camera motion | Theoretically best | Almost always requires a temporal or spatio-temporal denoiser for viewport navigation. |
| Freeze weights between camera cuts and train during still frames | Fast convergence while still | High while moving | Poor for moving lights and rigid/dynamic meshes | Very effective for architectural walkthroughs. Detect camera velocity, `prevViewProj` delta, and scene dirty flags. |
| Per-scene pre-training on load | None or minimal | Highest | Poor unless retrained | Discussed in C.1. |
| Cross-scene pretrained encoder + scene latent | Partial | Medium | Unknown | Discussed in C.2. |

A critical implementation detail for online NRC: use a **sample replay buffer** (a circular buffer of recent `(features, target)` pairs) rather than training exclusively on the samples generated in the current frame. Training on a mixture of the last N frames (e.g., 4–16 frames) decorrelates the extremely coherent 1-SPP ray distribution and greatly reduces temporal flickering. Your current design effectively has a replay buffer of ~0–1 frame because only Workgroup 0 updates weights and the "effective batch" collapsed to 32 samples (Attempt 2), which is the single largest optimizer starvation issue.

Also, do not measure convergence in epochs. Measure it in *million training samples*. Many NRC references converge on the order of 10^6 to 10^7 supervised samples for a scene, not 60–120 frames.

### A.3 Specular lobes, roughness split, and energy conservation

NRC should not cache the entire incoming radiance. It should cache the radiance of **rough diffuse and glossy-diffuse indirect transport**. High-frequency specular transport has very low spatial and directional autocorrelation, which is precisely what coordinate MLPs with low-frequency encodings and small networks generalize poorly at, and is also what path tracing and ReSTIR excel at.

#### Cutoff and lobe gating

Apply the decision at the ray that would spawn an NRC query (commonly at the first diffuse bounce or at bounce 2):

- Evaluate the surface BRDF roughness $\alpha$ (GGX) or the lagomorph/Disney roughness. 
- Query NRC only if the bounce is on a **diffuse or low-glossy** lobe and the path vertex is not a perfect or near-perfect mirror. A common heuristic is to cache when $\alpha \ge \alpha_{threshold}$ (e.g. 0.25–0.5) or when the BSDF has a significant Lambertian component. 
- For $\alpha < \alpha_{threshold}$ (glossy and specular), **do not query the cache. Continue tracing physically** for at least one more bounce or up to the specular chain limit. Alternatively, cap the specular bounce depth separately.
- Never cache transmission through thin dielectrics or perfectly specular caustic-carrying paths unless you explicitly partition them. These frequently benefit little from a slow, view-agnostic radiance cache.

#### Directional input

A large source of error in NRC is what direction to condition on. Predicting radiance *incident* over the hemisphere as a function of $(x,n,\omega_o)$ is important. Using only position and normal (a "radiance field" similar to a light field probe in 5D reduced form) will leak view-dependent glossy indirect lighting. At a minimum, encode the outgoing direction $\omega_o$ or the reflection direction hemisphere. Instant-NGP coordinate-based networks for radiance still condition on direction for non-Lambertian appearance, however, for a **diffuse-only cache target**, the ground truth integrated over the diffuse lobe becomes much less view-dependent and is significantly easier for a tiny MLP to learn.

#### Energy conservation and blending

1. **Split by transport.** $L = L_{direct} + L_{specular\ physical} + L_{diffuse\ neural\ or\ physical}$.
2. **Do not overwrite specular.** The neural prediction must never replace BSDF-sampled specular paths. Blend at the path throughput level: when NRC returns $L_{tail,diffuse}$, multiply by the diffuse reflectance and the cosine-weighted integration factor for that vertex and add it to the path radiance, terminating the diffuse continuation beyond the cutoff.
3. **Clamp and preserve albedo scale.** Predict irradiance-like or reflected radiance and ensure $L_{pred} \ge 0$. Apply firefly clamping to training targets (e.g., percentile or $\min(target, \tau \cdot \text{mean})$) before the loss, as rare high-energy light paths will dominate the L2 gradient and destabilize Adam.
4. **Avoid double-counting.** The 3% of rays used for training must trace the true physical tail. The remaining 97% may use the neural prediction. These must use the same cutoff, same lobe mask, and the same Russian roulette policy.
5. **Validation by path length.** A correct implementation should converge visually towards the reference as training samples accumulate, not towards bounce 2 direct illumination. Your Attempt 4 directly violates this. 

## SECTION B: GPU Architecture, WMMA, and Memory Bandwidth

### B.1 Reducing 80-byte query VRAM traffic

Writing per-query structs of 80 bytes for millions of rays is likely the largest contributor to your frame time increase. On RDNA, global memory writes from ray tracing and shading wavefronts are highly sensitive to coalescing. Queries generated by incoherent diffuse bounces are among the worst case for this.

Use a **generate-then-compact-then-batch** pipeline and move as much data as possible to registers and workgroup memory.

| Technique | Expected Impact | Implementation Notes |
|---|---|---|
| SoA query layout | High | Do not use an array of structs. Split into `pos[Q]`, `nor[Q]`, `dirWo[Q]`, `material[Q]`, `albedo[Q]`, `roughnessMetal[Q]`, `queryId[Q]`, etc. This drastically improves write coalescing. |
| Local tile buffering | High | Shade a tile of pixels or a compacted wave and buffer queries for that tile in LDS before flushing a contiguous block to VRAM. |
| Encode materials | High | Pack BSDF parameters into 16–32 bytes. 80 bytes strongly suggests storing auxiliary debugging data. For inference alone, 40–56 bytes is often sufficient; `queryId` and training flags can be in separate arrays. |
| Deferred query emission | Medium–High | Only emit NRC queries after the material evaluation and after passing the roughness and path probability tests. Your current pipeline appears to emit them at bounce 2 for many rays that should physically continue instead. |
| Hash or reuse by spatial cache first | High (scene dependent) | Before enqueueing a query, test a short-lived spatial temporal cache (e.g., a screen-space or world-space hash of `(quantized pos, quantized nor, material class)`) and reuse a previously predicted value. Many NRC implementations get large hit rates in static lighting and smooth surfaces. |
| Ray payload carry for training | Medium | Carry `queryId` and `trainThroughput` in the secondary ray payload (which already exists) instead of expanding the global query record. |

Do not attempt to "cache queries in registers across the entire screen". The divergence of diffuse rays prevents that. Tile-based LDS accumulation is realistic for a Vulkan compute + ray tracing wavefront renderer.

### B.2 Stable multi-workgroup Adam without single-workgroup bottleneck or LDS/scratch spills

Your structural bug (only Workgroup 0 updates weights) is mathematically incorrect. Discarding gradients from 255 workgroups reduces your stochastic gradient to a batch of ~32. The fix is a **parallel gradient reduction**. Atomic floating point additions to global gradient accumulator buffers are the correct approach for a weight count of the size used by NRC (commonly 10k–150k parameters for a tiny MLP).

#### Recommended reduction topology (two-level)

1. **Per-thread & per-workgroup reduction.** Each invocation computes gradients $\nabla_\theta L$ for its assigned training samples. Reduce gradients within the workgroup into LDS for the parameters assigned to that workgroup (parameter sharding) or reduce to a workgroup partial buffer.
2. **Global reduction.** Add workgroup partial sums into a global `gradBuffer` using `atomicAdd` on `float` (or `float16` with care). On AMD GPUs, 32-bit floating point atomics to VRAM are supported and for ~10^5 parameters and ~10^3–10^4 samples distributed across workgroups are far cheaper than serializing all work to WG 0. 
3. **Optimizer step.** Dispatch a separate, small compute pass that iterates over all parameters (or over parameter tiles) once per training frame and applies Adam ($m_t, v_t, \hat{m}, \hat{v}, \theta \leftarrow \theta - \alpha \hat{m}/(\sqrt{\hat{v}}+\epsilon)$) using the averaged gradients (`grad / batchSize`). This pass should be fully parallel across workgroups and across parameters.

#### Alternatives and notes

- **Parameter tiling by workgroup:** Shard weight matrices by rows or by output features so that each workgroup updates or reduces a disjoint subset of parameters, eliminating most atomic contention.
- **Avoid per-sample large local arrays.** Your spill to 267 KB was caused by storing intermediate activations and per-sample Jacobians on the stack. For reverse-mode MLP training in a compute shader, use **checkpointing by layer**, recompute activations, or store activations for the mini-batch in LDS and in structured buffers by layer. Do not have an array of the entire computation graph per thread. Cooperative matrix usage for the forward and backward GEMMs should also reduce register pressure considerably on RDNA 4.
- **Do not use subgroup reductions to cross workgroups.** Subgroups (waves) cannot reduce across workgroups. Only LDS, global atomics, and a second dispatch can.
- **Batch size target.** With correct cross-workgroup reduction you can realistically use 2048–8192 training samples per frame, which is what is required for online NRC to track lighting changes. 32 samples is numerically closer to stochastic gradient descent with extreme noise than to mini-batch Adam.

Also, verify that your Adam moments ($m, v$) are stored in VRAM in the same precision layout as weights and that they persist across frames. They must not be re-initialized.

### B.3 `vkCmdDispatchIndirect` for inference

Yes, driving inference from the atomic query counter is the correct architectural change, but **indirect dispatch alone will not recover all of the overhead** on AMD's command processor (CP).

#### Why it helps

At 4K, a static fullscreen dispatch launches on the order of $ceil((W*H)/256)$ workgroups regardless of how many NRC queries exist. If only 5%, 10%, or 25% of pixels produce diffuse queries, the vast majority of those waves execute a branch that immediately returns. Wave launch, register allocation, shader fetch, and barrier setup for those empty waves are still work for the GPU's scheduler.

`vkCmdDispatchIndirect` with `DispatchIndirectCommand { groupCountX = (queryCount + WG_SIZE - 1)/WG_SIZE, 1,1 }` will launch approximately only the workgroups required to process the enqueued queries.

#### AMD-specific caveats

- **Small dispatches are expensive.** If `queryCount` is very small (e.g., < 256–1024), an indirect dispatch can be less efficient than letting a larger persistent kernel run due to CP dispatch latency and pipeline bubbles. A hybrid is common: if `queryCount == 0` skip, else if `queryCount < threshold` use a small static dispatch or a persistent thread (workgroup persistent) loop, else use indirect.
- **Counter readback hazard.** The query counter written by the ray tracing/shading passes must be made visible to the indirect buffer consumer. Use a correct pipeline barrier between the queue write/compaction pass and `vkCmdDispatchIndirect` (transfer or compute dependency on the buffer with `VK_ACCESS_SHADER_WRITE_BIT` -> `VK_ACCESS_INDIRECT_COMMAND_READ_BIT`).
- **Pair with compaction.** Indirect dispatch over a non-compacted list (dispatching over the screen and testing a flag) gives far less benefit than dispatching over a compacted array of query indices. Compact the query list using a prefix sum (scan) over tiles or a single-pass workgroup scan followed by stream compaction. This also directly enables the SoA layout from B.1.
- **Two-pass is normal.** `clear counter -> trace/shade -> build indirect + compact -> barrier -> inference indirect -> atomic resolve -> training` is a standard wavefront reformulation.

For RDNA 4 specifically, also profile the `vkCmdDispatch` count and the "wave starts" counter in Radeon GPU Profiler (RGP). Empty fullscreen dispatches tend to show up very clearly there.

## SECTION C: Alternative Architectural Paradigms

### C.1 Abandon online training and pre-bake the neural cache per scene?

A **frozen, scene-pre-trained NRC (or a neural irradiance cache)** is a very reasonable engineering trade-off and, for many real-time applications, visually superior to under-trained online NRC at 1 SPP.

#### Trade-offs

| Aspect | Online (continuous) | Pre-baked on scene load/camera cut |
|---|---|---|
| Dynamic geometry | Good in principle | Poor (requires partial or full retrain) |
| Dynamic emissive & time-of-day | Adapts slowly | Stale until retrained |
| Static scenes | Converges over time | High quality in 1–5 s of GPU training before interaction |
| 1-SPP navigation | Often noisy/blotchy | Stable, low temporal variance |
| GPU frame budget | Consumes 0.6–1.9 ms every frame for backprop + Adam | Consumes 0 ms during interaction, budget spent at load |
| Complexity | Highest (AD, Adam, atomics across WG, supervision, replay) | Medium (training loop, but no frame-time training scheduler, races, and batch starvation) |
| Correctness risk | High (your supervision bug) | Lower (can train against a high-quality path-traced target) |

#### Recommendation

Implement a **hybrid "train while idle"** model first, not an irreversible abandonment:

- Detect scene or lighting dirtiness and camera motion (velocity magnitude).
- While the camera is stationary and the scene is static, train the network in the background using rays generated by the renderer (self-supervised against physical continuation paths).
- While the camera is moving or a dynamic entity moves, **freeze network weights** and either use the last trained network, fall back to physical tracing for the diffuse tail for a subset of pixels, or reduce the NRC coverage.
- On scene load, run a short **warm-up training phase** (1–3 seconds, uncapped by the 16.67 ms frame budget by spreading across frames or by a compute-heavy training mode) to obtain a usable initial cache.

This preserves support for dynamic content while solving the 1-SPP navigation problem that you are measuring. For a portfolio or research renderer with mostly static glTF scenes, full pre-baking is extremely compelling.

### C.2 Zero-shot, cross-scene pretrained radiance model?

In general, **no, a coordinate-only zero-shot MLP is not practically feasible for arbitrary unseen glTF assets** in the form that Instant-NGP style NRC uses.

#### Why

NRC networks are, overwhelmingly, *scene-overfit coordinate functions*. The mapping $f_\theta(x,n,\omega_o, m) \to L_{indirect}$ implicitly encodes:

- Global visibility and occlusion (which emitters are visible from which regions)
- Light source positions, intensities, colors, and indirect light transport relationships
- Scene scale, unit conventions, and material response
- Interreflections specific to that mesh topology

A network that takes only world-space position, surface normal, outgoing direction, and a small material code has **no explicit global scene representation**. Two completely different scenes can share identical $(x,n,\omega_o)$ yet require completely different radiance values.

#### Where zero-shot could be plausible

Research directions exist, but they materially change the architecture:

- **Scene-conditioned networks:** Encode the scene with a triplane, a sparse voxel feature grid, a latent vector from a scene encoder, or CLIP/geometry embeddings. This increases inference cost and VRAM significantly.
- **Generalizable neural rendering priors:** Methods related to neural light transport, GPNR, TransNeRF variants, or recent neural radiosity generalization are still far from arbitrary, unprepared glTF scenes running at real-time path tracing frame rates with tiny MLPs and WMMA.
- **Foundation models for rendering:** Not production-ready for dynamic, path-traced, unbiased supervision.

Conclusion for this project: **do not pursue dataset-wide zero-shot pretraining**. Pursue **per-scene online or load-time training**. If generalization is a goal, feature-grid + MLP per scene (the NRC formulation) is the correct design space.

### C.3 Pivot to ReSTIR GI + a spatio-temporal denoiser?

For a real-time 1-SPP renderer, in most cases, **yes, ReSTIR GI is more robust, more predictable in performance, and easier to reach a high-quality image than a GPU-trained MLP** given the bugs and overhead that you have cataloged. This does not mean NRC is a dead end, but it means the implementation order is likely inverted.

#### Direct comparison

| Criterion | Neural Radiance Cache (online) | ReSTIR GI (+ temporal + spatial denoiser) |
|---|---|---|
| Ground truth correctness | Hard (attribution across wavefront passes, AD) | Unbiased in expectation (reservoirs resample path samples) |
| 1-SPP convergence | Slow, needs many frames to learn domain | Leverages temporal reuse across frames + spatial reuse |
| Camera motion | Prone to ghosting, blur, underfitting | Handles with temporal rejection (normal, depth, roughness, motion vectors) |
| Dynamic lights & objects | Requires retraining | Reuses valid reservoirs with validity checks |
| GPU cost | Queue (80B) + static/full inference + backward + Adam | Reservoir buffers (typically 5–7x 16–32 bytes/pixel), neighbor rays, resampling, no backpropagation |
| Memory bandwidth | High VRAM traffic + weight gradients + moments | Mostly screen-space buffers |
| Debuggability | Opaque (network weights) | Inspectable (reservoir WI, M, p_hat, hit position) |
| AMD RDNA 4 fit | Cooperative matrices used but memory-bound by traffic | Ray tracing + compute heavy, better matches existing ReSTIR DI |
| Specular/glossy | Difficult to partition | Can be path depth or roughness limited similarly |

#### Important architectural note

ReSTIR GI does **not** remove the need for tracing indirect rays. It drastically **reduces how many full multi-bounce paths** need to be evaluated by reusing and combining indirect illumination samples between pixels and across frames. NRC attempts to **replace** the traversal and shading of bounces 3–16 with a single MLP evaluation. On RDNA 4, for shallow cutoffs, replacement has proven more expensive than reuse for your implementation.

Furthermore, you already implement ReSTIR DI and temporal accumulation. ReSTIR GI shares reservoir data structures, motion vectors, normal/depth reprojection, and history validation. Adding a modern spatio-temporal denoiser on top pairs extremely naturally with a 1-SPP + reservoir renderer. An online-trained neural network introduces an entirely new class of correctness bugs (automatic differentiation, optimizer stability, and supervision) that you have now encountered in five attempts.

#### Suggested hybrid (best of both)

A very successful modern direction is **ReSTIR GI to provide training targets or to provide the cache**, not MLPs first: for example, use ReSTIR GI for diffuse indirect and use NRC only in regions with poor reservoir history (disocclusions, fast motion) or cache temporally stable irradiance. Alternatively, **Neural Irradiance Caching** research can also be fed by reservoir samples. I would not recommend combining both from the start.

## Prioritized Action Plan (in order of highest return)

Ranked against your measured 12.06 ms -> 14.28 ms regression and functional defects.

| Order | Task | Resolves | Effort |
|---|---|---|---|
| 1 | Fix multi-bounce supervision using `queryId` + throughput propagation + atomic target accumulation (A.1). Stop training on `secDirectL`. | Fundamentally incorrect images | Medium |
| 2 | Fix cross-workgroup Adam (parallel gradient reduction + global grad atomics + separate optimizer dispatch). Eliminate "WG 0 only" (B.2). Increase effective batch to >= 2048. | Network never learns | High |
| 3 | Switch query storage to SoA, tile-buffer in LDS, pack to smallest size, and only enqueue queries passing lobe/roughness mask (A.3, B.1). | Largest bandwidth cost | Medium–High |
| 4 | Replace fullscreen inference dispatch with `vkCmdDispatchIndirect` + prefix-sum compaction of query list (B.3). | 1.25 ms static dispatch | Medium |
| 5 | Add training replay buffer + camera-motion freeze + firefly target clamping + diffuse-only cache (A.2–A.3). | 1-SPP flicker and instability | Medium |
| 6 | Evaluate ReSTIR GI against continuing NRC using a profiler (RGP) with the above fixes. If NRC still > physical bounce 3 cost, pivot to C.3. | Architectural decision | Research/Profiling |

## Top-Level Answer to the Core Hypothesis

Your core hypothesis is **correct**. 

> On modern high-throughput ray tracing architectures (RDNA 4), the serialized memory traffic of queuing secondary ray queries and launching GPU compute passes (inference, atomic resolve, backward propagation, and Adam updates) exceeds the traversal latency of physical rays for shallow bounce depths (bounces 2 through 4).

The evidence supports this quantitatively: tracing bounce 3 is `0.27 ms`, whereas NRC query writes + static inference + training sum to approximately `0.42 + 1.25 + 0.63 = 2.30 ms` of added GPU work in that breakdown. The algorithmic value of caching bounces 3–4 is, in many indoor and simple scenes, too small to pay for a general-purpose online optimizer on the GPU. The value of caching **deep** indirect (bounces 5–16) would be much larger relative to their BVH and shading cost, however, obtaining unbiased deep supervision through a streaming wavefront tracer is also proportionally harder.

**My strongest single recommendation:** before rewriting more of the neural training system, fix supervision (A.1) and the effective batch size (B.2). With those two defects, the current visual results do not measure NRC's algorithmic potential at all. After measuring a corrected build in Radeon GPU Profiler, make a data-driven decision between a frozen/load-time NRC and **ReSTIR GI + spatio-temporal denoising**. For a 1-SPP real-time viewport renderer, the latter matches your already-existing ReSTIR DI investment far better than runtime backpropagation does.

---

This is a genuinely well-diagnosed postmortem — several of your "attempts" have already isolated the actual bugs (unrouted ground truth in Attempt 4, the single-workgroup update bug in Attempt 2) rather than just symptoms. I'll go through each question with that in mind, since a few of your Section A/B questions have direct fixes given what you've already found.

## Section A: Algorithmic Foundations and Supervision

**A1 — routing continuation radiance back without global sync or path-history buffers**

You don't need a backward pass or a persistent history buffer at all. The recursive rendering equation `L_i = Le_i + f_i·L_{i+1}` can be accumulated *forward*, incrementally, entirely within the per-thread payload your wavefront architecture already carries across dispatches.

Extend the existing continuation-ray payload (which already tracks throughput and pixel index across bounce dispatches) with two extra scalars: `trainThroughput` (starts at 1.0 when the path is selected for training at the cutoff bounce) and `trainRadianceAccum` (starts at 0.0). At each subsequent bounce dispatch:

```
trainRadianceAccum += trainThroughput * (Le_bounce + NEE_direct_bounce)
trainThroughput *= bsdf_weight / pdf
```

When the path terminates (miss, emitter hit, Russian-roulette kill, or a hard depth cap — cap it, e.g. 8 bounces past cutoff, so training paths can't run open-ended and stall a dispatch), a single write lands `trainRadianceAccum` into the training-record slot that was allocated at spawn time. No backward walk, no barrier, no per-vertex history array — the "history" is two floats living in a payload you're already carrying. This is essentially what the original NRC paper's self-training scheme does under the hood; it's cheaper than what you built in Attempt 4, not more complex.

This directly fixes your Attempt 4 bug: right now the record is stamped with `secDirectL` and never touched again, so the network is being trained to reproduce bounce-2 direct lighting — a completely different (and much easier, much dimmer) target than integrated multi-bounce indirect radiance. That mismatch alone would explain a lot of the quality problems downstream of Attempt 4, independent of everything else.

**A2 — is online self-training viable for 1-SPP viewport nav without an external denoiser**

No, and it's worth separating why. NRC removes variance from the *truncated tail* (bounces beyond cutoff) by replacing it with a network regression. It does nothing for the variance in the bounces you still trace normally (0 through cutoff). At 1 SPP your direct lighting and first 1-2 indirect bounces are still raw Monte Carlo noise — NRC was never designed to denoise those, only to cap how deep you have to trace. So "no external denoiser" was never really on the table; NRC is complementary to a spatiotemporal denoiser, not a substitute for one.

Separately, the cold-start problem you're seeing during camera motion is real and structural: the network needs on the order of hundreds of frames of stable gradient signal to converge on newly revealed geometry, and every camera cut or fast pan invalidates a chunk of what it's learned. That transient shows up as exactly the blotchy/boiling artifact you described — it's inherent to online self-training under a real-time budget, not a tuning issue.

**A3 — roughness thresholds and energy conservation at the cutoff bounce**

Gate NRC strictly to diffuse and rough-glossy lobes above a roughness threshold (commonly ~0.3–0.5 roughness in published implementations). Below that, continue full path tracing recursively until you hit a sufficiently rough vertex — never query the cache at a near-specular vertex. The reason is representational, not just aesthetic: your encoding (positional + directional) is a smooth, band-limited function approximator, and a sharp specular lobe is close to a delta function in direction space. Forcing the network to fit that either blurs the highlight into a diffuse smear or the network just fails to converge on that region of input space at all.

For energy conservation, use reflectance factorization: train the network to predict incident radiance (or radiance divided by the local albedo/BSDF term) rather than raw outgoing radiance, then multiply the local BSDF·cosine/pdf term back in at query time, in your regular shading code. This keeps the material response living entirely in your existing BSDF evaluation path — the network only ever predicts a lighting quantity, so you don't have energy-conservation bugs baked into network weights that are opaque to debug.

## Section B: GPU Architecture, WMMA, and Memory Bandwidth

**B1 — decoupling query generation from inference to cut the 80–160 MB/frame traffic**

Three things, roughly in order of impact:

- **Shrink the record.** 80 bytes is generous for what NRC actually needs: quantized position (fp16×3), octahedral-encoded direction (fp16×2), a compact material/roughness field, and throughput (fp16×3), plus a pixel index for scatter-back — that's closer to 24–32 bytes. Halving-to-thirding the record size directly halves-to-thirds the bandwidth, independent of anything else.
- **Switch AoS → SoA.** An 80-byte-stride array-of-structs means every lane's read/write straddles cache lines. Restructure as struct-of-arrays (positions array, directions array, etc.) so a workgroup's coalesced access pattern actually hits full 128-byte lines. This is the same class of problem as your Attempt 1 hash-grid stalls, just showing up in the query buffer instead of the encoding table.
- **Wave-level compaction before the write.** Use ballot + prefix-sum within the generating wave so queries are already contiguous before they hit the global counter/buffer, instead of every lane doing an independent atomic-indexed scattered write. This also reduces contention on the counter itself.

If you want to go further, fusing generation and inference into one dispatch (persistent-thread style, each workgroup owns a tile and infers on its own queries without a VRAM round-trip) eliminates the record traffic entirely, but that trades off batch size — WMMA throughput wants large, uniform batches, and a tile-local batch is small and irregular. Given you're already fighting batch-size starvation (Attempt 2), I'd fix B2 and A1 first before attempting a fused pipeline.

**B2 — reduction topology for a 4K–16K sample Adam update without single-workgroup fallback or scratch spills**

Two-pass hierarchical reduction, not global atomics for the bulk sum:

1. **Pass 1 (local):** each workgroup owns a tile of, say, 256–1024 samples, does forward + backward entirely in registers/LDS (no per-sample scratch arrays — this is what caused your original 267 KB spill), and tree-reduces gradients across its own threads via subgroup ops + LDS. Output: one partial-gradient vector per workgroup, not per sample. 16,384 samples at 256/workgroup collapses to 64 partials.
2. **Pass 2 (global):** a small second dispatch sums those ~64 partials into the final gradient and performs the actual Adam moment updates (m, v, weight write), reading the previous frame's weights and writing the ones inference will use next.

This is also the direct fix for your Attempt 2 bug — right now only workgroup 0 updates weights and everyone else's gradient is discarded, capping effective batch size at 32 regardless of configuration. Pass 2 makes every workgroup's contribution land in the final update legitimately, with no cross-workgroup barrier required during Pass 1. Keep Adam's moment accumulators and master weights in fp32 even if the forward/backward math runs in fp16 through WMMA — standard mixed-precision practice, avoids the update silently vanishing into fp16 rounding at small gradient magnitudes.

**B3 — `vkCmdDispatchIndirect` from the query counter vs. static full-viewport dispatch**

Yes, do this — it's a clear win, not a marginal one. Launching 500K+ workgroups to cover a 4K viewport when only a fraction produced a diffuse query means most of those workgroups pay wave-launch and register-allocation cost for nothing, which can suppress occupancy for the waves that *do* have real work. Indirect dispatch sized from the GPU-written query counter is the standard pattern for this (same idea as GPU-driven culling or compacted shading passes, which you likely already have precedent for in your ReSTIR DI pipeline).

Two things to watch:
- Barrier the counter write before the indirect dispatch reads it — classic hazard if the generation pass and the indirect-dispatch-args write aren't properly ordered.
- Round the dispatched count up to a wave multiple, and consider a minimum-batch threshold (defer training a frame or two if the accumulated query count is tiny) so the fixed per-dispatch overhead doesn't dominate on quiet frames.

Small indirect dispatches do carry some fixed command-processor overhead, but it's on the order of sub-microsecond to low-microsecond — nowhere near the cost of hundreds of thousands of genuinely empty workgroups.

## Section C: Alternative Architectural Paradigms

**C1 — freeze-trained per-scene cache instead of continuous online training**

This is a legitimate and commonly used middle ground, and given your Attempt 5 numbers (0.63 ms/frame just for Adam training), it's probably worth adopting regardless of what you do with A1/B2. Pros: removes training cost from the interactive frame budget entirely, removes the cold-start blotchiness since the viewer never sees an unconverged network, and gives you seconds instead of a 16 ms slice to do the bake — much more forgiving for debugging the batch/reduction issues above. Cons: it only holds for static or quasi-static lighting — any moving light, animated mesh, or time-of-day change invalidates the frozen cache until the next bake, camera cuts either show a 2–3 s pop-in or need to fall back to full path tracing during the bake window, and you'd need a real invalidation heuristic (dirty-region tracking, not "rebake on every cut") or you pay the bake cost too often. It's also, honestly, a different design point than "true" online NRC — closer to a learned probe volume than a live cache — but that's a reasonable trade if it buys you stability.

**C2 — zero-shot pretrained forward model across arbitrary unseen glTF scenes**

Theoretically interesting, not practically available as a drop-in today. The core problem is that visibility/occlusion is a *global* property of the scene's geometry, and a small coordinate-conditioned MLP (what NRC/Instant-NGP-style networks are) has no mechanism to know what's blocking what in a scene it's never seen — there's no positional-encoding trick that lets weights trained on scene A's occlusion generalize to scene B's completely different geometry. Getting real zero-shot behavior would require conditioning the network on an actual per-scene geometric representation (a voxel/SDF feature volume, similar to what Brixelizer already builds for AMD's SDF pipeline, or point-cloud features) fed alongside the coordinate query — which is a materially bigger architecture and an active academic research area, not a small per-scene MLP that trains in milliseconds. I wouldn't budget engineering time toward this for a shipping renderer right now.

**C3 — pivot from NRC to ReSTIR GI + spatio-temporal denoiser**

Given you already have ReSTIR DI and temporal accumulation, and given your own Attempt 5 numbers, I think this is the stronger near-term direction. ReSTIR GI is a natural extension of infrastructure you've already validated (same reservoir resampling machinery), needs no training pipeline at all (no Adam, no backprop, no WMMA batching headaches), and the published results are strong: Ouyang et al.'s ReSTIR GI paper reports a path sampling algorithm for indirect lighting built on ReSTIR's spatio-temporal resampling that resamples multi-bounce paths obtained by path tracing, sharing information about important light paths across both time and pixels, with MSE improvements ranging from 9.3× to 166× over plain path tracing at one sample per pixel per frame in their test scenes, reaching high-quality path-traced global illumination at real-time frame rates when paired with a denoiser.

On the denoiser name — AMD's actual FidelityFX SDK ships a Denoiser module providing denoising compute shaders for reflection and shadow rendering, usable with either ray-traced or rasterized content, plus a separate Brixelizer GI technique built on a compute-based sparse distance field. Brixelizer GI is SDF-based, not ReSTIR/reservoir-based, so it's not a drop-in denoiser for a ReSTIR GI reservoir buffer. "Upways" doesn't match anything I can find on AMD's or a third party's side — worth double-checking that name. If you want a portable, ReSTIR-aware spatiotemporal filter, NVIDIA's NVIDIA Real-time Denoisers (NRD) is open source and vendor-neutral, and is what ReSTIR GI is commonly paired with in practice.

Trade-off worth naming honestly: ReSTIR GI's gains come from spatial reuse, which introduces its own bias-correction burden (Jacobian MIS weights, boiling/light-leak artifacts through thin geometry) — that's real complexity, just complexity of the same kind you already fought through for ReSTIR DI, versus the largely new-to-you complexity of GPU-side backprop and optimizer reduction.

## Net read

Your Attempt 4 and Attempt 2 findings aren't tuning problems — they mean the network has been training on the wrong target with an effectively fixed tiny batch. That alone makes the current NRC quality/performance numbers not a fair verdict on the algorithm. But even granting a clean implementation, your own Attempt 5 breakdown (query writes + inference + training costing more than the 0.27 ms it takes to just trace bounce 3 directly) is consistent with how NRC was originally framed: a win for *deep* multi-bounce paths where plain tracing is prohibitively expensive, not a general replacement for bounces 3-4. If your target workloads are genuinely shallow (≤4 bounces), ReSTIR GI is the structurally better fit given what you already have built. If you have workloads that need real deep-bounce GI, it's worth fixing A1/B2 first (they're bounded, well-understood fixes) and pairing the result with the freeze-and-bake strategy in C1 rather than continuous online training.
