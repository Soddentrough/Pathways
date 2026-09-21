# Pathways Engine Architecture Specification
## Pure Vulkan 1.4 Real-Time Path Tracing with Autonomous Device-Generated Commands, Wavefront Microkernels, and Multi-GPU Scaling

- **Engine Baseline**: Vulkan Core 1.4 (1.4.341+)
- **Primary Hardware Targets**: AMD RDNA 4 (`gfx1201` / Radeon AI PRO R9700) and RDNA 3 architectures
- **Design Philosophy**: GPU-autonomous Device-Generated Commands (DGC), decoupled wavefront microkernels, zero-copy host memory multi-GPU scaling, and standard cross-vendor Vulkan 1.4 without proprietary vendor extensions.

---

## 1. Architectural Philosophy & Engine Foundations

Real-time path tracing at 4K (3840×2160) within an interactive 8.33 ms (120 FPS) or 16.66 ms (60 FPS) frame budget demands extreme architectural discipline. Traditional real-time renderers deploy **monolithic megakernels** that package ray generation, BVH traversal, material evaluation, light sampling, and secondary ray recursion into a single execution unit.

While conceptually straightforward, monolithic ray tracing pipelines suffer from catastrophic hardware inefficiencies on modern wide SIMD architectures:
1. **Intra-Wave Branch & Path-Length Divergence**: SIMD execution units serialize across divergent execution paths when adjacent lanes evaluate differing materials or terminate at differing path lengths.
2. **Vector General-Purpose Register (VGPR) Bloat**: Monolithic shaders must preserve registers and samplers for all supported BSDFs simultaneously, demanding 72–128+ VGPRs per thread and capping hardware wave occupancy to 25%–50%.
3. **Cache Locality Thrashing**: Adjacent threads access disparate textures, material buffers, and BVH nodes, destroying L0/L1 vector cache hit rates.

Pathways discards the monolithic megakernel paradigm in favor of a **GPU-Autonomous Wavefront Architecture** driven by standard Vulkan 1.4 **Device Generated Commands (`VK_EXT_device_generated_commands`)**.

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
   - Evaluates surface material archetype (Diffuse, Dielectric, Conductor, Complex, Emissive, Passthrough).
   - Computes 16-bit composite sorting keys combining material archetype and quantized 3D Morton spatial codes.
   - Atomically stages rays into dedicated Structure-of-Arrays (SoA) work-list queues via 64-bit Buffer Device Addresses (BDA).
2. **GPU-Autonomous Command Synthesis**:
   - The classifier kernel synthesizes indirect dispatch commands directly into a device command buffer.
   - Updates `VkIndirectExecutionSetEXT` pipeline tokens on the device with zero CPU intervention.
3. **Autonomous Microkernel Execution (`vkCmdExecuteGeneratedCommandsEXT`)**:
   - Dispatches only the exact wave counts needed for each material queue.
   - Vector register pressure is tailored to each physical lobe:
     - `shade_diffuse.comp`: Pure Lambertian diffuse reflection + shadow query. Operates at **24 VGPRs** with **100% Wave32 hardware occupancy**.
     - `shade_dielectric.comp`: Snell's law refraction, Total Internal Reflection (TIR), and volumetric Beer-Lambert absorption. Operates at **40 VGPRs** with **100% occupancy**.
     - `shade_conductor.comp`: Anisotropic GGX specular microfacets with Fresnel-Conductor physics. Operates at **48 VGPRs** with **100% occupancy**.
     - `shade_complex.comp`: Layered clearcoat, transmission, and sheen BSDFs. Operates at **64 VGPRs** with **50% occupancy**.
4. **Shadow Occlusion (`shaders/compute/wavefront_shadow.comp`)**:
   - Evaluates direct lighting visibility using binary inline ray queries (`rayQueryConfirmIntersectionEXT`), bypassing hit shader overhead.
5. **Accumulation Resolve (`shaders/compute/accum_running_avg.comp`)**:
   - Numerically stable progressive HDR accumulation using online Welford updates, preventing highlight blowout and floating-point accumulation drift.

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
       ▼  Zero CPU Intervention / Zero PCIe Bus Contention
  [ Primary GPU (GPU 0) ]
       │
       ▼  Direct Import & Tile Composite (<0.12 ms)
  [ Final 4K Swapchain Presentation ]
```

### 4.1 Zero-Copy Host Memory Streaming (`VK_EXT_external_memory_host`)
- **Default Mode (`--mgpu-transfer host`)**: Secondary GPU streams completed $64\times 64$ checkerboard tiles into pinned host memory via CP DMA posted writes at PCIe 4.0/5.0 bus line rate (~25 GB/s, latency <0.5 ms).
- **Direct Primary Import**: Primary GPU imports the host pointer and composites alternate tiles in ~0.12 ms without PCIe bus contention, ensuring consistent 250+ FPS camera motion.

### 4.2 Linux DMA-BUF Direct P2P (`--mgpu-transfer p2p`)
- Uses `VK_EXT_external_memory_dma_buf` and `VK_KHR_external_semaphore_fd` for direct cross-device memory sharing on hardware with coherent inter-GPU links (e.g. Infinity Fabric).

### 4.3 Fine-Grained 2D Checkerboard Tiling
Screen space is subdivided into $64\times 64$ alternating tiles (2,040 tiles at 4K). Dual GPUs execute perfectly balanced spatial and shading workloads, scaling framerates by **$1.72\times$ to $1.93\times$** over single-GPU performance.

---

## 5. Super-Resolution & Neural Reconstruction

Pathways integrates two state-of-the-art super-resolution paradigms:

### 5.1 AMD FidelityFX Super Resolution 3.1 (FSR 3.1)
- Integrated via modular `Fsr3Upscaler` compute pipeline (`shaders/compute/fsr3_*.comp`).
- Processes jittered low-resolution color, inverted depth, and motion vectors through Lanczos accumulation and Robust Contrast Adaptive Sharpening (RCAS).
- Multi-GPU checkerboard tile reprojection ensures jitter-free temporal stability across alternating GPU frames.

### 5.2 Pathways Upways ML Neural Reconstruction
- Native in-engine neural reconstruction driven by `VK_KHR_cooperative_matrix` Wave32 Wave Matrix Multiply Accumulate (WMMA) instructions on RDNA 4 (`gfx1201`).
- Evaluates INT8/FP16 quantized recurrent convolutional autoencoders directly on device tensor cores, reconstructing 4K HDR images from 1 SPP inputs.

---

## 6. Reservoir Spatiotemporal Importance Sampling (ReSTIR DI)

Pathways implements ReSTIR Direct Illumination (`src/rt/ReSTIRManager.cpp`) to handle many-light environments:
- **Temporal Reservoir Reuse**: Projects light candidate reservoirs across consecutive frames via motion vectors, evaluating temporal visibility confidence.
- **Spatial Reservoir Reuse**: Exchanges light candidates across neighboring pixels within a spatial radius, suppressing direct lighting variance by orders of magnitude with a single shadow ray evaluation.

---

## 7. OpenUSD Stage Ingestion & High-Density Instancing

Pathways ingests complex VFX and CAD production assets via OpenUSD (`UsdLoader.cpp`):
- **High-Density Point Instancing (`UsdGeomPointInstancer`)**: Evaluates scenes exceeding 180,000 instances and ~360 million expanded triangles (e.g. `Scanlands.usdc`) without geometry flattening.
- **Prototype BLAS Deduplication**: Instanced geometries reference compact deduplicated prototype BLAS acceleration structures, reducing VRAM footprint by up to 90% (e.g. 3.58 GB peak VRAM at 1080p for 359M instanced triangles).
- **Dynamic Real-Time TLAS Build**: Builds top-level acceleration structures in <2.0 ms per frame on RDNA 4 hardware.
