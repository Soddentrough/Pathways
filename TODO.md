# Pathways - Engineering Roadmap & TODO

## Vulkan Specification Alignment & Next-Gen Pipeline Modernization (Spec 1.4.341 → 1.4.363+)

Following the review of the Vulkan 1.4 specification updates (through 1.4.363), the following architectural upgrades and optimizations remain open. The host SDK is now 1.4.357.0, so extensions ratified through 1.4.357 are available; items introduced in 1.4.361+ still require a further SDK upgrade.

---

### 1. Direct Buffer Device Address Commands (`VK_KHR_device_address_commands`)
- [ ] **Status:** Evaluated / Partially Redundant / SDK Available (Ratified in Vulkan 1.4.346; host SDK 1.4.357)
- [ ] **Target Component:** `src/rt/DGCManager.cpp`, `src/rt/WavefrontPipeline.cpp`
- [ ] **Objective:** Evaluate `vkCmdDispatchIndirect2KHR` across indirect compute dispatch paths.
- [ ] **Architectural Assessment & Notes (September 2026):**
  - **DGC Redundancy**: `DGCManager` already uses native 64-bit GPU device addresses (`genInfo.indirectAddress` and `genInfo.preprocessAddress`) via `VK_EXT_device_generated_commands`. DGC does not use `VkBuffer` handles for execution.
  - **Indirect Fallback**: Replacing `vkCmdDispatchIndirect` with `vkCmdDispatchIndirect2KHR` in the non-DGC fallback path generates byte-for-byte identical hardware PM4 packets on AMD Command Processors; CPU recording savings are negligible (< 0.001 ms).
  - **ReSTIR / NRC Clarification**: `ReSTIRManager` and `NRCManager` execute direct compute dispatches (`vkCmdDispatch`), not indirect dispatches. Their buffer bindings use descriptor sets (`VkDescriptorBufferInfo`). Eliminating buffer handles in ReSTIR/NRC requires Buffer Device Address (BDA) via Push Constants or `VK_EXT_descriptor_buffer`, not `VK_KHR_device_address_commands`.
- [ ] **Revised Tasks:**
  - [ ] Optional (API consistency only): swap `vkCmdDispatchIndirect` fallback calls in `DGCManager.cpp` and `WavefrontPipeline.cpp` to `vkCmdDispatchIndirect2KHR`.

---

### 2. Modular Material Pipeline Libraries (`VK_KHR_pipeline_library_group_handles`)
- [ ] **Status:** Proposed / Pending SDK Upgrade (Introduced in Vulkan 1.4.362)
- [ ] **Target Component:** `src/rt/RTPipeline.cpp`, `src/rt/WavefrontPipeline.cpp`
- [ ] **Objective:** Support dynamic composition of material evaluation microkernels using pipeline libraries without invalidating Shader Binding Table (SBT) entries.
- [ ] **Tasks:**
  - [ ] Enable `VK_KHR_pipeline_library_group_handles` on device creation.
  - [ ] Separate individual material BSDF lobes (`shade_diffuse`, `shade_dielectric`, `shade_conductor`, `shade_complex`) into modular pipeline libraries.
  - [ ] Query bitwise-identical group handles directly from pipeline libraries for dynamic SBT assembly.
  - [ ] Evaluate hot-reloading performance for material shaders during interactive sessions.

---

### 3. GPU-Side Assertions & Telemetry Diagnostics (`VK_KHR_shader_abort` & `VK_KHR_device_fault`)
- [ ] **Status:** Proposed / SDK Available (Introduced in Vulkan 1.4.347; host SDK 1.4.357)
- [ ] **Target Component:** `src/core/Logger.hpp`, `src/vulkan/VulkanContext.cpp`, `shaders/compute/*`
- [ ] **Objective:** Implement reliable GPU hang diagnosis and shader assertions during autonomous DGC execution.
- [ ] **Tasks:**
  - [ ] Enable `VK_KHR_device_fault` and configure `VkDeviceFaultAddressInfoEXT` dump handler in `VulkanContext`.
  - [ ] Enable `VK_KHR_shader_abort` in debug builds to trap out-of-bounds queue indices and invalid NaN/Inf radiance vectors directly on the GPU.
  - [ ] Integrate structured fault logs (faulting memory address, pipeline, shader stage) into Pathways telemetry.
  - [ ] Enable `VK_KHR_shader_constant_data` solely as the `VK_KHR_shader_abort` dependency for compile-time assertion messages (evaluated and rejected for sampling tables: divergent table indexing causes VGPR bloat vs. the zero-memory PCG RNG).

---

### 4. Command Buffer Lifecycle Optimization (Vulkan 1.4.361 / Issue 2782)
- [ ] **Status:** Proposed / Pending SDK Upgrade
- [ ] **Target Component:** `src/vulkan/VulkanContext.cpp`, `src/mgpu/MultiGpuManager.cpp`
- [ ] **Objective:** Streamline command buffer re-recording without explicit `vkResetCommandBuffer` calls.
- [ ] **Tasks:**
  - [ ] Update frame submission rings to call `vkBeginCommandBuffer` directly from the `RECORDING` state.
  - [ ] Verify compatibility across both dual AMD Radeon AI PRO R9700 devices under ROCm/RADV.

---

## Physical Material System & Light Transport Roadmap

### 5. Subsurface Scattering (BSSRDF & Volumetric Light Transport)
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

---

## Interactive Dynamics & Physics Simulation Roadmap

### 6. Rigid-Body Physics Engine Integration (Box3D) & Dynamic Bouncing Balls Showcase
- [ ] **Status:** Proposed / Architectural Design Phase
- [ ] **Target Component:** `src/physics/` (new), `src/core/Engine.cpp`, `src/scene/ProceduralScene.cpp`, CMake build system
- [ ] **Objective:** Integrate Erin Catto's **Box3D** (C17 3D rigid-body engine) to drive dynamic physical simulations in real-time, showcasing hardware ray-traced reflections, caustics, and shadows across hundreds of colliding rigid bodies.
- [ ] **Showcase Scene Concept ("Cornell Box Physics Sandbox"):**
  - An empty Cornell box enclosure featuring a transparent fourth wall (smooth dielectric glass panel, IOR ~1.5) facing the camera.
  - A dynamic bucket / hopper mechanism or initial cluster that dumps dozens/hundreds of reflective/dielectric bouncing spheres into the box.
  - Real-time simulation of rigid-body gravity, ball-to-ball and ball-to-boundary collisions, restitution, rolling friction, and resting contact.
- [ ] **Architectural Assessment & Integration Requirements:**
  - **Decoupled Fixed-Timestep Loop:**
    - Drive Box3D using a deterministic fixed sub-stepping loop (`FIXED_TIMESTEP = 1.0f / 60.0f` or `120.0f`) with a wall-clock accumulator (`timeAccumulator += frameDelta`) in `Engine::renderFrame()`, avoiding simulation instability across variable path-tracing frame rates.
    - Interpolate body transforms ($X_{\text{render}} = \text{lerp}(X_{\text{prev}}, X_{\text{curr}}, \alpha)$) for smooth rendering.
  - **TLAS Refit & Instancing Pipeline:**
    - Leverage Pathways' existing Tier-3 GPU-timeline TLAS refit infrastructure (`m_tlasInputInstancesBuffer`, `update_tlas_instances.comp`, and `recordBuildTLAS(..., updateMode=true)`).
    - Map rigid body positions/orientations directly into `ASInstanceGPUData::transform` each frame without reallocating or rebuilding BLAS geometry.
    - BLAS Prototype: Single shared unit-sphere BLAS (or procedural `SphereGPU` primitives) instanced $N$ times across the TLAS.
  - **Light Transport & Temporal Denoising with Dense Dynamic Objects:**
    - Real-time accumulation policy: Reset accumulation or evaluate 1-SPP real-time mode with Upways / FSR 3.1 neural reconstruction.
    - Caustics & Specular Reflections: Dynamic caustic photon tracing (`caustic_photon_trace.comp`) and high-specular bounces inside the transparent enclosure.
- [ ] **Tasks & Implementation Steps:**
  - [ ] Add `box3d` as a submodule or third-party C17 library in `third_party/box3d`.
  - [ ] Implement `PhysicsWorld` wrapper encapsulating Box3D world initialization, rigid bodies, shapes (sphere, box, plane), and simulation stepping.
  - [ ] Implement `createPhysicsCornellBoxScene()` in `ProceduralScene.cpp` with glass fourth wall and sphere prototype instances.
  - [ ] Wire fixed-timestep update in `Engine::renderFrame()` to sync Box3D body transforms into `Engine::updateInstanceTransform()`.
  - [ ] Add ImGui controls in `GuiManager` for physics reset, ball spawn rate, gravity, and restitution.

---

## Testing & Validation Roadmap

### 7. Upscaler Quality Validation (FSR 3.1 / FSR 4.1 ML / Upways SR)
- [ ] **Status:** Proposed
- [ ] **Target Component:** `tests/test_fidelityfx_features.cpp`
- [ ] **Background (November 2026):**
  - The former `edgeRetention` metric (renamed `gradientEnergyRatio`) is the ratio of summed Sobel gradient magnitude in the test image vs. the 256-SPP reference on reference edge pixels. It is not an edge-fidelity measure: it ignores edge position, orientation and shape, and residual Monte Carlo noise, ringing, over-sharpening and aliasing all *increase* it (raw unaccumulated 1-SPP scored highest at 1.214).
  - The `FSR 4.1 ML > FSR 3.1 + 0.03` assertion was removed because it was effectively ranking residual 1-SPP grain, not upscaling quality. The ratio is retained only as a crude blur/black-frame guard.
  - Current runs feed 1-SPP input into a static Cornell Box for 12 frames, so upscalers act mostly as temporal denoisers on content they are not tuned for.
- [ ] **Tasks:**
  - [ ] Isolate upscaling from denoising: feed well-converged (high-SPP or accumulated) input at the internal render resolution.
  - [ ] Compare against a converged native-resolution reference using full-reference perceptual/structural metrics (NVIDIA FLIP, SSIM/MS-SSIM) and per-pixel gradient error |∇img − ∇ref| instead of gradient-sum ratios.
  - [ ] Add camera motion and scenes with fine, diagonal and high-frequency detail (thin geometry, textures, specular highlights) to exercise reprojection, disocclusion and anti-aliasing.
  - [ ] Evaluate temporal stability (frame-to-frame flicker / temporal FLIP) separately from single-frame quality.
  - [ ] Report scores as informational with tolerance bounds vs. the reference; avoid hard "upscaler A must beat upscaler B" assertions.
