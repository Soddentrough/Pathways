# Pathways - Engineering Roadmap & TODO

## Vulkan Specification Alignment & Next-Gen Pipeline Modernization (Spec 1.4.341 → 1.4.363+)

Following the review of the Vulkan 1.4 specification updates (through 1.4.363), the following architectural upgrades and optimizations are planned once the host SDK is updated beyond 1.4.341.0.

---

### 1. Direct Buffer Device Address Commands (`VK_KHR_device_address_commands`)
- [ ] **Status:** Evaluated / Deprecated / Partially Redundant (Ratified in Vulkan 1.4.346)
- [ ] **Target Component:** `src/rt/DGCManager.cpp`, `src/rt/WavefrontPipeline.cpp`
- [ ] **Objective:** Evaluate `vkCmdDispatchIndirect2KHR` across indirect compute dispatch paths.
- [ ] **Architectural Assessment & Notes (September 2026):**
  - **DGC Redundancy**: `DGCManager` already uses native 64-bit GPU device addresses (`genInfo.indirectAddress` and `genInfo.preprocessAddress`) via `VK_EXT_device_generated_commands`. DGC does not use `VkBuffer` handles for execution.
  - **Indirect Fallback**: Replacing `vkCmdDispatchIndirect` with `vkCmdDispatchIndirect2KHR` in the non-DGC fallback path generates byte-for-byte identical hardware PM4 packets on AMD Command Processors; CPU recording savings are negligible (< 0.001 ms).
  - **ReSTIR / NRC Clarification**: `ReSTIRManager` and `NRCManager` execute direct compute dispatches (`vkCmdDispatch`), not indirect dispatches. Their buffer bindings use descriptor sets (`VkDescriptorBufferInfo`). Eliminating buffer handles in ReSTIR/NRC requires Buffer Device Address (BDA) via Push Constants or `VK_EXT_descriptor_buffer`, not `VK_KHR_device_address_commands`.
- [ ] **Revised Tasks:**
  - [ ] If desired for API consistency once SDK > 1.4.346 is installed, swap `vkCmdDispatchIndirect` fallback calls in `DGCManager.cpp` and `WavefrontPipeline.cpp` to `vkCmdDispatchIndirect2KHR`.

---

### 2. Compute Latency Hiding via Split Barriers (`VK_EXT_shader_split_barrier`)
- [ ] **Status:** Evaluated / Rejected for Wavefront Classify (Introduced in Vulkan 1.4.351)
- [ ] **Target Component:** Multi-wave compute filters / Denoiser spatial passes with shared LDS
- [ ] **Objective:** Overlap inter-subgroup synchronization with independent math in multi-subgroup compute workgroups.
- [ ] **Architectural Assessment & Notes (September 2026):**
  - **Causal Dataflow Violation in Classify**: In `wavefront_classify.comp`, 2D Morton Z-curve mapping is the input coordinate generator for primary ray direction, BVH query traversal, and hit evaluation. SoA ray queue stores (`outGeoms`, `outHits`, `outStates`) happen after traversal. It is physically impossible to overlap queue writes with Morton ALU because Morton ALU must finish hundreds of cycles before ray traversal can even launch.
  - **Single-Wave Workgroup (Wave32)**: `wavefront_classify.comp` uses `local_size_x = 8, local_size_y = 4` (32 threads). On AMD RDNA 3.5 / RDNA 4 (`gfx1151`, `gfx1201`), this maps to a single Wave32 wave where all threads execute in SIMD lockstep. There are zero inter-subgroup barriers within the workgroup; all inter-lane sharing uses hardware subgroup intrinsics (`subgroupBallot`, `subgroupShuffle`).
  - **Control vs. Memory Barriers**: `VK_EXT_shader_split_barrier` splits workgroup control barriers (`OpControlBarrierArriveEXT` / `OpControlBarrierWaitEXT`). It cannot split global memory barriers (`memoryBarrierBuffer()`). Global stores are already non-blocking fire-and-forget instructions handled by hardware L0/L1 write buffers.
- [ ] **Revised Scope & Tasks:**
  - [ ] Drop `wavefront_classify.comp` from split barrier optimization.
  - [ ] Re-evaluate split barriers only if authoring wide multi-wave filter kernels ($\ge 256$ threads per workgroup) utilizing shared memory (LDS).

---

### 3. Modular Material Pipeline Libraries (`VK_KHR_pipeline_library_group_handles`)
- [ ] **Status:** Proposed / Pending SDK Upgrade (Introduced in Vulkan 1.4.362)
- [ ] **Target Component:** `src/rt/RTPipeline.cpp`, `src/rt/WavefrontPipeline.cpp`
- [ ] **Objective:** Support dynamic composition of material evaluation microkernels using pipeline libraries without invalidating Shader Binding Table (SBT) entries.
- [ ] **Tasks:**
  - [ ] Enable `VK_KHR_pipeline_library_group_handles` on device creation.
  - [ ] Separate individual material BSDF lobes (`shade_diffuse`, `shade_dielectric`, `shade_conductor`, `shade_complex`) into modular pipeline libraries.
  - [ ] Query bitwise-identical group handles directly from pipeline libraries for dynamic SBT assembly.
  - [ ] Evaluate hot-reloading performance for material shaders during interactive sessions.

---

### 4. GPU-Side Assertions & Telemetry Diagnostics (`VK_KHR_shader_abort` & `VK_KHR_device_fault`)
- [ ] **Status:** Proposed / Pending SDK Upgrade (Introduced in Vulkan 1.4.347)
- [ ] **Target Component:** `src/core/Logger.hpp`, `src/vulkan/VulkanContext.cpp`, `shaders/compute/*`
- [ ] **Objective:** Implement reliable GPU hang diagnosis and shader assertions during autonomous DGC execution.
- [ ] **Tasks:**
  - [ ] Enable `VK_KHR_device_fault` and configure `VkDeviceFaultAddressInfoEXT` dump handler in `VulkanContext`.
  - [ ] Enable `VK_KHR_shader_abort` in debug builds to trap out-of-bounds queue indices and invalid NaN/Inf radiance vectors directly on the GPU.
  - [ ] Integrate structured fault logs (faulting memory address, pipeline, shader stage) into Pathways telemetry.

---

### 5. Embedded Sampling Tables via Constant Data (`VK_KHR_shader_constant_data`)
- [ ] **Status:** Evaluated / Rejected for Sampling Tables (Introduced in Vulkan 1.4.347)
- [ ] **Target Component:** `src/vulkan/VulkanContext.cpp` (prerequisite for `VK_KHR_shader_abort`)
- [ ] **Objective:** Utilize `OpConstantDataKHR` for static diagnostic strings and debug assertions.
- [ ] **Architectural Assessment & Notes (September 2026):**
  - **Microarchitectural Inversion (VGPR Bloat)**: Embedding multi-kilobyte constant tables (Sobol matrices, Halton permutations) into shader binaries causes severe VGPR pressure. Because sampling indices vary per pixel and per bounce (divergent across the wave), the compiler cannot keep tables in scalar registers (SGPRs); it must load them into VGPRs or spill to private scratch memory, degrading Wave32 CU occupancy.
  - **Pathways Baseline is Already Zero-Descriptor & Zero-Memory**: Pathways uses an analytical PCG random number generator (`pcg_hash` in `wavefront_common.glsl`), which requires 0 descriptor slots, 0 bytes of VRAM, and exactly 1 `uint` state register in VGPR. Embedding constant tables would strictly regress performance and register usage.
  - **QMC Best Practice**: If low-discrepancy sampling (e.g. PMJ02bn or Cranley-Patterson rotated Sobol) is adopted in the future, standard practice is binding a small $64 \times 64$ or $128 \times 128$ tileable blue-noise / scramble texture or compact 32-element direction array via existing bindless textures, utilizing hardware L0/L1 texture caches with high spatial locality.
- [ ] **Revised Scope & Tasks:**
  - [ ] Retain `VK_KHR_shader_constant_data` solely as a dependency for `VK_KHR_shader_abort` to embed compile-time assertion messages and diagnostic string tables.
  - [ ] Do not embed divergent numerical sampling matrices into shader constants.

---

### 6. Command Buffer Lifecycle Optimization (Vulkan 1.4.361 / Issue 2782)
- [ ] **Status:** Proposed / Pending SDK Upgrade
- [ ] **Target Component:** `src/vulkan/VulkanContext.cpp`, `src/mgpu/MultiGpuManager.cpp`
- [ ] **Objective:** Streamline command buffer re-recording without explicit `vkResetCommandBuffer` calls.
- [ ] **Tasks:**
  - [ ] Update frame submission rings to call `vkBeginCommandBuffer` directly from the `RECORDING` state.
  - [ ] Verify compatibility across both dual AMD Radeon AI PRO R9700 devices under ROCm/RADV.

---

## Physical Material System & Light Transport Roadmap

### 7. Subsurface Scattering (BSSRDF & Volumetric Light Transport)
- [ ] **Status:** Proposed / Architectural Design Phase
- [ ] **Target Component:** `src/scene/Material.hpp`, `src/rt/WavefrontPipeline.cpp`, `shaders/compute/wavefront_shade_*.comp`, `shaders/compute/wavefront_sss.comp`
- [ ] **Objective:** Implement physical Subsurface Scattering (SSS) for translucent and organic materials (skin, marble, wax, jade, milk, foliage) beyond current thin-walled diffuse transmission.
- [ ] **Current State & Existing Foundations:**
  - **Thin-Walled Diffuse Transmission (`KHR_materials_diffuse_transmission`)**: Already supported in `wavefront_shade_complex.comp` and USD/glTF loaders. Evaluates light transmission across zero-thickness geometry ($\vec{L} \cdot \vec{N} < 0$), ideal for leaves, paper, and thin fabrics.
  - **Homogeneous Dielectric Volume Absorption (`KHR_materials_volume`)**: Already supported in `wavefront_shade_dielectric.comp` via Beer-Lambert exponential absorption ($\sigma_a = -\ln(C_\text{atten}) / d_\text{atten}$). Handles colored transparent media (glass, water), but internal scattering is zero ($\sigma_s = 0$).
  - **Gap**: Pathways lacks BSSRDF evaluation where light penetrates a boundary at $x_i$, multiple-scatters within a participating medium, and exits at a distinct position $x_o$.
- [ ] **Proposed Architectural Approaches:**
  - **Approach A: Path-Traced Random Walk SSS (Physical Ground Truth)**:
    - *Mechanism*: When a ray hits an SSS surface, enter the internal volume and trace a random walk using Woodcock delta tracking / null-collision techniques with Henyey-Greenstein phase function sampling until the ray exits the boundary mesh.
    - *Wavefront Integration*: Create a dedicated `wavefront_sss.comp` microkernel or extend the secondary bounce queues. Rays inside the medium stay in an internal scattering queue, decoupled from surface shading kernels to preserve the 83-VGPR / 100% occupancy target on RDNA 4 (`gfx1201`).
    - *Hardware Considerations*: Internal ray steps require fast BVH boundary testing (testing distance to mesh exit). Can utilize `rayQueryEXT` with `gl_RayFlagsCullFrontFacingTrianglesEXT` to locate boundary exits, or detached boundary test queues.
  - **Approach B: Screen-Space / Texture-Space Diffusion Approximation (Real-Time Realism)**:
    - *Mechanism*: Separable bilateral / Disney normalized diffusion filter applied to primary hit radiance in screen space, guided by depth, geometric normal, and per-pixel mean free path radius.
    - *Wavefront Integration*: Operates as an independent post-shading compute pass prior to temporal accumulation / upscaling.
    - *Hardware Considerations*: Minimal VGPR impact on core path tracing loop; predictable latency (~0.3–0.6 ms at 4K on dual R9700), but limited to camera-visible surfaces and subject to screen-edge silhouette artifacts.
  - **Approach C: Hybrid Burley Normalized Diffusion / Dipole Ray Guiding**:
    - *Mechanism*: At primary hit point $x_i$, sample an exit location $x_o$ on the surface according to the Burley BSSRDF profile disk, evaluate incoming direct illumination at $x_o$ via the detached shadow queue, and add to outgoing radiance at $x_i$.
- [ ] **Tasks & Implementation Steps:**
  - [ ] **Material Definition**:
    - Extend `MaterialGPU` and `ShadeMaterialGPU` with SSS parameters: `subsurfaceFactor`, `subsurfaceColor`, `subsurfaceRadius` (or mean free path $l_\text{mfp}$ / reduced scattering coefficient $\sigma_s'$), and phase anisotropy $g$.
    - Map parameters from glTF (`KHR_materials_subsurface` / `KHR_materials_translucency`) and OpenPBR / USD `subsurface` schemas in `UsdLoader.cpp`.
  - [ ] **Classifier & Queue Routing**:
    - Update `computeMaterialArchetype` in `Material.hpp` and `wavefront_classify.comp` to identify SSS materials.
    - Route SSS surfaces to the `COMPLEX` microkernel or an optional dedicated `ARCHETYPE_SSS` queue if multi-step random walk is used.
  - [ ] **Shader Implementation**:
    - Phase 1: Implement thin-walled and screen-space Burley normalized diffusion pass for real-time validation.
    - Phase 2: Implement full volumetric random walk kernel with decoupled queue dispatches for reference ground truth.
  - [ ] **Regression & Performance Validation**:
    - Create reference test scene with Stanford Lucy or subsurface sphere/wax model.
    - Validate energy conservation, multi-GPU split-frame consistency, and ensure zero regression on standard opaque/dielectric pipelines.
