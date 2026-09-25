# Pathways Engine Architecture Specification
## Vulkan 1.4 Real-Time Path Tracing with Autonomous Device-Generated Commands, Wavefront Microkernels, and Multi-GPU Scaling

- **Engine Baseline**: Vulkan Core 1.4 (1.4.341+)
- **Primary Hardware Targets**: AMD RDNA 4 (`gfx1201` / Radeon AI PRO R9700) and RDNA 3 architectures
- **Design Philosophy**: GPU-autonomous Device-Generated Commands (DGC), decoupled wavefront microkernels, zero-copy host memory multi-GPU scaling, and standard cross-vendor Vulkan 1.4 without proprietary vendor extensions.

---

## 1. Architectural Philosophy & Engine Foundations

Real-time path tracing at 4K (3840×2160) within an interactive 8.33 ms (120 FPS) or 16.66 ms (60 FPS) frame budget requires structured pipeline design and register management. Traditional real-time renderers deploy **monolithic megakernels** that package ray generation, BVH traversal, material evaluation, light sampling, and secondary ray recursion into a single execution unit.

While conceptually straightforward, monolithic ray tracing pipelines encounter several hardware bottlenecks on modern wide SIMD architectures:
1. **Intra-Wave Branch & Path-Length Divergence**: SIMD execution units serialize across divergent execution paths when adjacent lanes evaluate differing materials or terminate at differing path lengths.
2. **Vector General-Purpose Register (VGPR) Bloat**: Monolithic shaders must preserve registers and samplers for all supported BSDFs simultaneously, demanding 72–128+ VGPRs per thread and capping hardware wave occupancy to 25%–50%.
3. **Cache Locality Thrashing**: Adjacent threads access disparate textures, material buffers, and BVH nodes, degrading L0/L1 vector cache hit rates.

Pathways replaces the monolithic megakernel approach with a **GPU-driven Wavefront Architecture** using standard Vulkan 1.4 **Device Generated Commands (`VK_EXT_device_generated_commands`)**.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   PATHWAYS WAVEFRONT PIPELINE STAGES                                  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  [ Primary Ray Generation ]
             │
             ▼
  [ Hardware BVH Traversal ] ──► VK_KHR_ray_query / VK_KHR_ray_tracing_pipeline
             │
             ▼
  [ Ray Classification & Binning ] ──► 3D Spatial Morton + Material Composite Binning
             │
             ▼
  [ GPU-Autonomous DGC Dispatch ] ──► vkCmdExecuteGeneratedCommandsEXT
             ├──────────────────────┬──────────────────────┬──────────────────────┐
             ▼                      ▼                      ▼                      ▼
     [ shade_diffuse ]     [ shade_dielectric ]   [ shade_conductor ]    [ shade_complex ]
     (24 VGPRs, 100% Occ)   (40 VGPRs, 100% Occ)   (48 VGPRs, 100% Occ)   (64 VGPRs, 50% Occ)
             │                      │                      │                      │
             └──────────────────────┴──────────────────────┴──────────────────────┘
                                    │
                                    ▼
                       [ Shadow Occlusion Test ] ──► Inline Ray Query
                                    │
                                    ▼
                       [ Progressive Accumulation ] ──► Running Average / Welford
                                    │
                                    ▼
                       [ Temporal Super-Resolution ] ──► AMD FSR 3.1 / Upways ML
```

---

## 2. Wavefront Decomposition & GPU-Autonomous DGC

Pathways decomposes light transport into decoupled, specialized compute microkernels:

### 2.1 The Wavefront Stages
1. **Ray Classification (`shaders/compute/wavefront_classify.comp`)**:
   - Inspects surface intersection records produced by BVH traversal.
   - Evaluates surface material archetype (Diffuse, Dielectric, Conductor, Complex, Emissive, Passthrough) directly via the 4-byte scalar `materialArchetypes` buffer (binding 34), bypassing full material structure loads into VGPRs.
   - Computes 16-bit composite sorting keys combining material archetype and quantized 3D Morton spatial codes.
   - Atomically stages rays into dedicated Structure-of-Arrays (SoA) ray queues via 64-bit Buffer Device Addresses (BDA).
2. **Ray Traversal & Intersection (`shaders/compute/wavefront_intersect.comp`)**:
   - Evaluates fixed-function hardware BVH traversal using inline ray queries (`rayQueryEXT`).
   - Retrieves triangle shading attributes from the 128-byte cache-line aligned `TriangleShadeGPU` buffer (binding 2).
   - Conditionally evaluates tangent frames and object-to-world transforms exclusively for `COMPLEX`, `CONDUCTOR`, and `DIELECTRIC` archetypes, skipping tangent attribute loads and matrix math for diffuse and emissive surfaces.
3. **GPU-Autonomous Command Synthesis (`shaders/compute/wavefront_classify.comp`)**:
   - The classifier kernel synthesizes dual execution command streams into device-local memory without host readbacks:
     - A standard `VkDispatchIndirectCommand` stream (16-byte stride per archetype) consumed by the hardware Command Processor's native multi-dispatch loop (`vkCmdDispatchIndirect`). This production path achieves maximum throughput (+22.3% faster) with zero driver preprocessing latency and hardware 0-workgroup dispatch skipping.
     - A 16-byte `DGCCommand` stream (`ExecutionSet` token + `Dispatch` token) for `VkIndirectExecutionSetEXT` execution sets when `--dgc-execset` is selected.
4. **Material Microkernel Dispatch (`vkCmdDispatchIndirect` / `vkCmdExecuteGeneratedCommandsEXT`)**:
   - Dispatches only the exact workgroup counts needed for each material queue.
   - High-frequency diffuse and primary shading kernels query the compact 64-byte `ShadeMaterialGPU` buffer (binding 35), fetching packed albedo, emissive/specular, PBR parameters, and texture flags at 2 materials per 128B vector cache line.
   - Secondary diffuse shading (`#if !IS_SECONDARY_BOUNCE`) bypasses normal map texture sampling and TBN perturbation, preserving vector registers and memory bandwidth for indirect diffuse GI.
   - Vector register pressure is tailored to each physical lobe:
     - `shade_diffuse.comp`: Lambertian diffuse reflection. Operates at **24–32 VGPRs** (up to **100% Wave32 hardware occupancy** in pure shading mode; 52 VGPRs / 56.2% occupancy on secondary bounces).
     - `shade_dielectric.comp`: Snell's law refraction, Total Internal Reflection (TIR), and volumetric Beer-Lambert absorption. Operates at **40–42 VGPRs** with **up to 100% occupancy** (62.5% with dispersion).
     - `shade_conductor.comp`: Anisotropic GGX specular microfacets with Fresnel-Conductor physics. Operates at **48 VGPRs** with **100% occupancy**.
     - `shade_complex.comp`: Layered clearcoat, transmission, and sheen BSDFs. Operates at **61–64 VGPRs** with **50% occupancy**.
5. **Shadow Occlusion (`shaders/compute/wavefront_shadow.comp`)**:
   - Evaluates direct lighting visibility using binary inline ray queries (`rayQueryConfirmIntersectionEXT`), bypassing hit shader overhead.
6. **Accumulation Resolve (`shaders/compute/accum_running_avg.comp`)**:
   - Numerically stable progressive HDR accumulation using online Welford updates, preventing highlight blowout and floating-point accumulation drift.

### 2.2 Coarse-Batch Wavefront Partitioning & 2D Macro-Tile Decomposition ($O(1)$ Memory Scaling)
Staging all ray queues simultaneously across a native 4K UHD viewport ($3840 \times 2160 = 8.29\text{M pixels}$) consumes approximately **2,721 MB of VRAM** for double-buffered ray and state queues. On unified memory architectures (APUs/UMA) where CPU and GPU share memory bandwidth, cycling this volume of memory each frame induces severe memory unit stalls (>60%).

Pathways solves this with **GPU-Autonomous Coarse-Batch Wavefront Partitioning**:
1. **Target Batch Pixel Budget**: Ray queues are sized strictly for an on-chip pixel budget ($2.0\text{M pixels}$ on APU/UMA platforms, $1.0\text{M pixels}$ on discrete GPUs) via `getTargetBatchPixels()`.
2. **2D Aspect-Ratio Grid Decomposition**: Rather than 1D horizontal strips (which sever vertical spatial neighbors and cause up to a 35% cache-locality regression on textured assets), the viewport is partitioned into a 2D tile grid ($2 \times 1, 2 \times 2, 4 \times 2, 3 \times 3$) matching screen aspect ratio:
   $$\text{TileWidth} = \left\lceil \frac{W}{G_x} \right\rceil, \quad \text{TileHeight} = \left\lceil \frac{H}{G_y} \right\rceil$$
   This preserves 2D spatial texture and BVH cache locality across both primary and secondary rays.
3. **Adaptive Secondary Ray CU Occupancy Capping**: In multi-bounce diffuse GI scenes (`max_bounces > 2`), dividing the screen into too many batches causes Compute Unit starvation on Bounces 2–3 due to diminishing active ray counts. Pathways dynamically caps auto-batches to $\le 4$, ensuring sufficient ray volume to saturate all SIMD execution units while keeping queue memory locked at **~699 MB at 4K (-74.3% reduction)**.
4. **True $O(1)$ Memory Scaling**: Rendering at 8K ($7680 \times 4320$) simply schedules a $4 \times 4$ macro-tile grid without expanding ray queue memory beyond the configured batch budget.

Configurable via `--macro-tiles <count>` (or `--batches <count>`). For empirical SPM cache analysis, see [docs/reports/wavefront_batching_head_to_head.md](reports/wavefront_batching_head_to_head.md).

---

## 3. Spatial, Morton & Directional Ray Coherence

To preserve cache locality across secondary ray bounces:

### 3.1 3D Spatial-Morton + Material Dual-Binning
Sorting rays purely by material restores ALU wave efficiency but disperses spatial locality across scene geometry. Pathways uses a dual-binning key:
$$\text{Key}_{16} = \big(\text{ArchetypeID} \ll 12\big) \;\vert\; \text{Morton3D}(x, y, z)_{12}$$
This clusters rays by material archetype while preserving spatial L0/L1 cache locality for texture and geometry fetches.

### 3.2 Producer-Side Directional Octant Queuing
Secondary rays scattered by rough surfaces are partitioned at emission time across 8 directional octant bins using Wave32 ballot leader-election loops:
```glsl
// Subgroup ballot leader election for octant binning
uint octant = (rayDir.x >= 0.0 ? 1 : 0) |
              (rayDir.y >= 0.0 ? 2 : 0) |
              (rayDir.z >= 0.0 ? 4 : 0);
uvec4 ballot = subgroupBallot(true);
```
Rays traversing downstream BVHs share coherent ray frustums, boosting BVH node hit rates and eliminating post-hoc sort passes.

### 3.3 Secondary Ray Radiance Clamping
To eliminate high-energy Monte Carlo fireflies caused by specular-diffuse-specular paths, secondary rays enforce scene-invariant radiance clamping:
$$\mathbf{L}_{\text{clamped}} = \min\left(\mathbf{L}_{\text{secondary}}, \frac{L_{\text{max}}}{\max(1.0, \|\mathbf{p} - \mathbf{x}\|)}\right)$$
Configurable via `--indirect-clamp` (default `10.0`), suppressing caustic fireflies with zero energy loss on primary surfaces.

---

## 4. Real-Time Multi-GPU Scaling Architecture

Pathways provides unlinked multi-GPU scaling across dual discrete GPUs (e.g. 2x AMD Radeon AI PRO R9700 32GB) over standard PCIe slots without requiring proprietary hardware bridges.

```
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                              ZERO-COPY HOST MEMORY MULTI-GPU TOPOLOGY                                  │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  [ Secondary GPU (GPU 1) ]
       │
       ▼  Renders Alternate 64x64 Checkerboard Tiles
  [ Device VRAM ]
       │
       ▼  CP DMA High-Speed Posted Writes (~25 GB/s PCIe Bus Line Rate)
  [ Pinned Host RAM ] ──► VK_EXT_external_memory_host
       │
       ▼  GPU-Autonomous DMA / Pinned Host Buffering
  [ Primary GPU (GPU 0) ]
       │
       ▼  Direct Import & Tile Composite (<0.12 ms)
  [ Final 4K Swapchain Presentation ]
```

### 4.1 Zero-Copy Host Memory Streaming (`VK_EXT_external_memory_host`)
- **Default Mode (`--mgpu-transfer host`)**: Secondary GPU streams completed $64\times 64$ checkerboard tiles into pinned host memory via CP DMA posted writes at PCIe 4.0/5.0 bus line rate (~25 GB/s, latency <0.5 ms).
- **Direct Primary Import**: Primary GPU imports the host pointer and composites alternate tiles in ~0.12 ms without PCIe bus contention, maintaining consistent frame pacing.

### 4.2 Linux DMA-BUF Direct P2P (`--mgpu-transfer p2p`)
- Uses `VK_EXT_external_memory_dma_buf` and `VK_KHR_external_semaphore_fd` for direct cross-device memory sharing on hardware with coherent inter-GPU links (e.g. Infinity Fabric).

### 4.3 Fine-Grained 2D Checkerboard Tiling
Screen space is subdivided into $64\times 64$ alternating tiles (2,040 tiles at 4K). Dual GPUs execute balanced spatial and shading workloads across alternating tiles, scaling framerates by **$1.72\times$ to $1.93\times$** over single-GPU performance.

---

## 5. Super-Resolution & Neural Reconstruction

Pathways provides two super-resolution options:

### 5.1 AMD FidelityFX Super Resolution 3.1 (FSR 3.1)
- Integrated via modular `Fsr3Upscaler` compute pipeline (`shaders/compute/fsr3_*.comp`).
- Processes jittered low-resolution color, inverted depth, and motion vectors through Lanczos accumulation and Robust Contrast Adaptive Sharpening (RCAS).
- Multi-GPU checkerboard tile reprojection ensures jitter-free temporal stability across alternating GPU frames.

### 5.2 Pathways Upways Neural Reconstruction & Continuous Super-Resolution (Wave32 WMMA)
Pathways integrates a native in-engine neural reconstructor and continuous super-resolution pipeline (`src/rt/UpwaysPipeline.cpp`, `shaders/compute/upways_reconstruct.comp`) accelerated directly on hardware tensor cores via Vulkan `VK_KHR_cooperative_matrix`:
- **Wave32 WMMA Tensor Architecture**:
  - Targets AMD RDNA hardware matrix instructions (`v_wmma_f32_16x16x16_f16`) with native subgroup size 32.
  - Features a 4-layer fully connected topology: `FC1` ($32 \to 64$), `FC2` ($64 \to 64$), `FC3` ($64 \to 64$), and `FC4` ($64 \to 16$).
  - **Zero VRAM Traffic for Inference**: All intermediate activations (`s_acc`, `s_fc1_out`, `s_fc2_out`, `s_fc3_out`, `s_fc4_out`) execute entirely within Local Data Share (LDS) shared memory across 16-pixel workgroups, completely eliminating memory bus bandwidth overhead during inference.
- **Physical Demodulation & Invertible Logarithmic Compression**:
  - Compresses input radiance ($0$ to $10,000+$ nits) into an invertible logarithmic space:
    $$\mathbf{y} = \text{sign}(\mathbf{x}) \cdot \log\big(1 + \mu \|\mathbf{x}\|\big)$$
    preventing high-energy specular fireflies from destabilizing temporal history.
  - Demodulates smooth irradiance from base albedo and surface roughness before inference, reconstructing pin-sharp high-frequency texture details at target display resolution.
- **Dual-Stream Temporal Reprojection & Confidence Gating**:
  - Warps independent temporal histories using surface motion vectors ($\mathbf{v}_{\text{surface}}$) for diffuse GI and virtual hit specular vectors ($\mathbf{v}_{\text{specular}}$) with planar depth-aware disocclusion confidence gating, eliminating ghosting during rapid camera motion.
- **Continuous 2.0x Super-Resolution (`--upways-sr`)**:
  - Ingests fractional subpixel coordinate phase offsets ($\text{fract}(\text{inCoordF})$ in channels 30–31) to reconstruct high-frequency geometric edges from lower-resolution render targets (e.g. 1080p $\to$ 4K in **8.62 ms / 116 FPS** on Veach Ajar).
- **Compacted Weight Buffer & Embedded Fallback**:
  - Uses an ultra-lean **16,704-byte** serialized FP16 SSBO binary (`data/models/upways_weights.bin`), automatically packaged by CMake (`cmake/PackageUpwaysWeights.cmake`) from the adjacent `~/Development/Upways` repository when present, with an embedded compiled-in fallback C++ header (`src/rt/upways_default_weights.hpp`) for 100% self-contained standalone execution.
- **Hardware Timestamp Profiling**:
  - Instrumented with dedicated Vulkan GPU query timestamps (`qBase + 4/5`), exposing microsecond-accurate inference latency (`m_lastUpwaysMs`) in `EngineStats` and telemetry JSON dumps.
- **Unified Scaler CLI (`--scaler`)**:
  - Consolidated `--scaler <mode> [ratio|res]` syntax supporting interchangeable scaler engines (`upways`, `fsr`, `fsr1`, `none`), standard presets (`native`, `quality`, `balanced`, `performance`, `ultra`), or arbitrary internal rendering resolutions (e.g. `--res 4k --scaler upways 1080` or `--scaler upways quality`). Legacy `--denoiser upways` and `--upways-sr` remain fully supported as seamless aliases.

---

## 6. Reservoir Spatiotemporal Importance Sampling (ReSTIR DI)

Pathways implements ReSTIR Direct Illumination (`src/rt/ReSTIRManager.cpp`) to handle many-light environments:
- **Temporal Reservoir Reuse**: Projects light candidate reservoirs across consecutive frames via motion vectors, evaluating temporal visibility confidence.
- **Spatial Reservoir Reuse**: Exchanges light candidates across neighboring pixels within a spatial radius, significantly reducing direct lighting variance with a single shadow ray evaluation.

---

## 7. OpenUSD Stage Ingestion & High-Density Instancing

Pathways ingests complex VFX and CAD production assets via OpenUSD (`UsdLoader.cpp`):
- **High-Density Point Instancing (`UsdGeomPointInstancer`)**: Evaluates scenes with tens of thousands of instances and tens of millions of expanded triangles (e.g. `PointInstancedMedCity.usd` with 40,001 instances and ~49M triangles) without geometry flattening.
- **Prototype BLAS Deduplication**: Instanced geometries reference compact deduplicated prototype BLAS acceleration structures, reducing VRAM footprint by up to 90%.
- **Dynamic Real-Time TLAS Build**: Builds top-level acceleration structures in <2.0 ms per frame on RDNA 4 hardware.

---

## 8. Hardware Data Structures & Cache-Line Alignment

On modern GPU architectures such as AMD RDNA 4 (`gfx1201`), vector cache lines ($L0$ and $L1$) are strictly **128 bytes**. When memory transactions access unaligned data or structures that straddle 128-byte boundaries, the memory subsystem issues two memory requests instead of one—incurring a 100% bandwidth penalty ("split cache-line penalty"). Pathways structures all geometry and material buffers to ensure optimal cache-line alignment and minimal memory bandwidth.

### 8.1 128-Byte Geometry Shading Buffer (`TriangleShadeGPU`)

In traditional rasterization and path tracing engines, triangle vertex positions, normals, texture coordinates, and tangents are interleaved into a single fat vertex structure (e.g. 160+ bytes per triangle). In a wavefront path tracer, however:
1. **Hardware BVH Traversal** requires only vertex positions during acceleration structure building (`VkAccelerationStructureGeometryTrianglesDataKHR`). Traversal itself runs in fixed-function ray tracing hardware.
2. **Shading Kernels** need surface normals, texture coordinates, tangents, and material IDs to evaluate BSDFs and sample textures; they do *not* require vertex positions because world-space hit points are computed from ray origins, direction, and ray query hit distances $t$ ($\mathbf{p} = \mathbf{o} + t \cdot \mathbf{d}$).

Pathways segregates positions from shading geometry into two decoupled buffers:
- **`m_positionBuffer`**: A contiguous array of 16-byte `glm::vec4(x, y, z, 1.0f)` positions used strictly for hardware BLAS builds.
- **`m_triangleShadeBuffer` (`TriangleShadeGPU`)**: An aligned 128-byte structure containing exclusively the attributes required during shading:

| Field | GLSL / C++ Type | Byte Size | Description |
| :--- | :--- | :---: | :--- |
| `normal0_u0` | `vec4` / `glm::vec4` | 16 | Vertex 0 normal (`xyz`), Vertex 0 UV $u$ coordinate (`w`) |
| `normal1_u1` | `vec4` / `glm::vec4` | 16 | Vertex 1 normal (`xyz`), Vertex 1 UV $u$ coordinate (`w`) |
| `normal2_u2` | `vec4` / `glm::vec4` | 16 | Vertex 2 normal (`xyz`), Vertex 2 UV $u$ coordinate (`w`) |
| `tan0_v0` | `vec4` / `glm::vec4` | 16 | Vertex 0 tangent (`xyz`), Vertex 0 UV $v$ coordinate (`w`) |
| `tan1_v1` | `vec4` / `glm::vec4` | 16 | Vertex 1 tangent (`xyz`), Vertex 1 UV $v$ coordinate (`w`) |
| `tan2_v2` | `vec4` / `glm::vec4` | 16 | Vertex 2 tangent (`xyz`), Vertex 2 UV $v$ coordinate (`w`) |
| `tanSigns` | `vec4` / `glm::vec4` | 16 | Tangent handedness signs (`x: tan0.w, y: tan1.w, z: tan2.w, w: unused`) |
| `materialId` | `uint32_t` | 4 | Scene material index |
| `padding[3]` | `uint32_t[3]` | 12 | Alignment padding to guarantee 16-byte / 128-byte boundary |
| **Total** | | **128 Bytes** | **Exactly 1 RDNA 4 Vector Cache Line (0 Split Cache-Line Penalty)** |

```cpp
struct alignas(16) TriangleShadeGPU {
    glm::vec4 normal0_u0; // xyz: normal0, w: uv0.x
    glm::vec4 normal1_u1; // xyz: normal1, w: uv1.x
    glm::vec4 normal2_u2; // xyz: normal2, w: uv2.x
    glm::vec4 tan0_v0;    // xyz: tan0,    w: uv0.y
    glm::vec4 tan1_v1;    // xyz: tan1,    w: uv1.y
    glm::vec4 tan2_v2;    // xyz: tan2,    w: uv2.y
    glm::vec4 tanSigns;   // x: tan0.w, y: tan1.w, z: tan2.w, w: 0.0f
    uint32_t materialId;
    uint32_t padding[3];
};
static_assert(sizeof(TriangleShadeGPU) == 128, "TriangleShadeGPU must be exactly 128 bytes (1 L0 cache line)");
```

### 8.2 Compact 64-Byte Shading Material Buffer (`ShadeMaterialGPU`, Binding 35)

Full physical material descriptions (`MaterialGPU`, 208 bytes) contain extended parameters for clearcoat, transmission, dispersion, sheen, and thin-film iridescence. In high-frequency shading (diffuse, primary ray dispatch, and shadow queries), the vast majority of hits require only core albedo, emissive, metallic-roughness, IOR, and texture handles.

Pathways introduces a compact 64-byte material representation (`ShadeMaterialGPU`) bound at descriptor binding 35:
- **Cache Packing**: Exactly **two** `ShadeMaterialGPU` entries fit into a single 128-byte vector cache line.
- **Bandwidth Reduction**: Accessing material state incurs 64 bytes instead of 208 bytes—a **69.2% reduction in material read bandwidth**.

| Field | Type | Size | Packed Contents |
| :--- | :--- | :---: | :--- |
| `albedo` | `vec4` | 16 | Base color factor (linear RGBA) |
| `emissive_spec` | `vec4` | 16 | `xyz`: Emissive factor (linear RGB), `w`: Specular factor |
| `pbrParams` | `vec4` | 16 | `x`: Roughness, `y`: Metallic, `z`: IOR, `w`: Diffuse transmission factor |
| `tex_flags` | `uvec4` | 16 | `x`: `albedoTex \| (normalTex << 16)`<br>`y`: `mrTex \| (emissiveTex << 16)`<br>`z`: `diffTransTex \| (specularTex << 16)`<br>`w`: `(matType & 0xFFFF) \| (packHalf16(normalScale) << 16)` |
| **Total** | | **64 Bytes** | **Packs 2 materials per 128B cache line (`alignas(16)`)** |

### 8.3 4-Byte Scalar Material Archetype Buffer (`materialArchetypes`, Binding 34)

During ray classification (`wavefront_classify.comp`) and ray intersection (`wavefront_intersect.comp`), the engine must identify each surface hit's BSDF archetype to route the ray to the correct queue or decide whether to evaluate tangents:
$$\text{Archetype} \in \{\text{Diffuse}, \text{Dielectric}, \text{Conductor}, \text{Complex}, \text{Emissive}, \text{Passthrough}\}$$
Instead of reading 64-byte or 208-byte material records, Pathways binds a dedicated array of 32-bit scalars (`uint materialArchetypes[]`) at binding 34:
- Direct index lookup: `uint archetype = materialArchetypes[matId];`
- **Zero Structure Cache Bloat**: Keeps classification and intersection inner loops lean, saving 60–204 bytes of memory fetch per ray hit and preserving VGPR registers.

### 8.4 Secondary Bounce Tangent & Normal Map Bypass

Secondary indirect bounces in diffuse environments carry low-frequency global illumination. Evaluating high-frequency normal maps and loading tangent frames on secondary diffuse bounces introduces significant ALU and memory overhead for imperceptible visual difference.

Pathways deploys a two-tier tangent bypass:
1. **Intersection Bypass (`wavefront_intersect.comp`)**:
   Tangent vectors (`tan0_v0`, `tan1_v1`, `tan2_v2`, `tanSigns`) and object-to-world tangent matrix multiplications are executed **only** if the hit archetype is `COMPLEX`, `CONDUCTOR`, or `DIELECTRIC`:
   ```glsl
   if (archetype == MATERIAL_ARCHETYPE_COMPLEX || 
       archetype == MATERIAL_ARCHETYPE_CONDUCTOR || 
       archetype == MATERIAL_ARCHETYPE_DIELECTRIC) {
       tanDir = getTriangleTangent(tri, closestBary);
       tanDir = normalize(mat3(hitO2w) * tanDir);
       tanSign = getTriangleTangentSign(tri);
   }
   ```
   Diffuse and emissive hits bypass all 4 tangent `vec4` loads and $3\times 3$ matrix transforms.
2. **Shading Bypass (`wavefront_shade_diffuse.comp`)**:
   Compile-time specialization (`#if !IS_SECONDARY_BOUNCE`) strips normal map texture reads, unpack half operations, and TBN orthonormalization from secondary bounce shaders:
   ```glsl
   #if !IS_SECONDARY_BOUNCE
   if (hitType == 0u && normalTex > 0u && normalTex <= 512u) {
       vec3 normalMap = SAMPLE_SCENE_TEXTURE(normalTex, hitUv).rgb * 2.0 - 1.0;
       normalMap.xy *= normalScale;
       normalMap = normalize(normalMap);
       geomTan = normalize(geomTan - dot(geomTan, hitNormal) * hitNormal);
       vec3 geomBitangent = cross(hitNormal, geomTan) * tanSign;
       mat3 tbn = mat3(geomTan, geomBitangent, hitNormal);
       hitNormal = normalize(tbn * normalMap);
   }
   #endif
   ```
   This drops secondary diffuse shader register pressure from 85 VGPRs to **52 VGPRs**, lifting occupancy to **56.2% (9 waves/SIMD)** on RDNA 4 hardware.

### 8.5 Descriptor Set Binding Table

The primary wavefront and compute pipelines bind scene data through descriptor set 0:

| Binding | Resource Type | Name | Stride / Size | Purpose |
| :---: | :--- | :--- | :---: | :--- |
| **0** | `storageImage` | `accumImage` | RGBA32F / RGBA16F | Progressive accumulation & render target |
| **1** | `uniformBuffer` | `cameraUBO` | 256 B | Camera projection, view matrices, resolution |
| **2** | `storageBuffer` | `triangles` | **128 B** | Cache-line aligned shading geometry (`TriangleShadeGPU`) |
| **3** | `storageBuffer` | `spheres` | 32 B | Procedural analytic spheres (`SphereGPU`) |
| **4** | `storageBuffer` | `materials` | 208 B | Full glTF 2.0 extended PBR materials (`MaterialGPU`) |
| **5** | `storageBuffer` | `lights` | 64 B | Analytical & directional scene lights |
| **6** | `accelerationStructure` | `topLevelAS` | — | Hardware Top-Level Acceleration Structure (TLAS) |
| **7** | `combinedSampler` | `environmentMap` | — | HDR/EXR equirectangular environment texture |
| **8** | `combinedSampler[512]` | `sceneTextures` | — | Bindless/descriptor array of glTF/USD textures |
| **9** | `storageBuffer` | `inStates` | 32 B | Input ray state queue (throughput, seed, radiance) |
| **10** | `storageBuffer` | `outStates` | 32 B | Output ray state queue for next bounce |
| **11** | `storageBuffer` | `queueCounters` | 64 B | Atomic workgroup and ray queue counters |
| **20–22** | `storageBuffer` | `nrcQueue/Counters` | Variable | Neural Radiance Caching query, train, and counter buffers |
| **23–24** | `storageImage` | `motionVectors / normalDepth` | RG16F / RGBA16F | Temporal motion vectors and G-Buffer normal/depth |
| **25** | `storageBuffer` | `lightTree` | Variable | Hierarchical 3D Light Tree nodes |
| **26–29** | `storageImage` | `albedoRough / specMetal / mlDiff / mlSpec` | RGBA16F | Denoising feature maps & separated diffuse/specular buffers |
| **30** | `storageBuffer` | `instances` | 64 B | Scene mesh instance transforms and material offsets |
| **31** | `storageImage` | `causticImage` | RGBA16F | Forward photon-traced caustic splat target |
| **32** | `storageBuffer` | `restirReservoirs` | Variable | ReSTIR DI spatio-temporal light candidate reservoirs |
| **34** | `storageBuffer` | `materialArchetypes` | **4 B** | Compact scalar BSDF archetype lookup buffer |
| **35** | `storageBuffer` | `shadeMaterials` | **64 B** | Compact cache-line aligned shading material buffer |
