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

---

## Interactive Dynamics & Physics Simulation Roadmap

### 8. Rigid-Body Physics Engine Integration (Box3D) & Dynamic Bouncing Balls Showcase
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

## Material Authoring & Procedural Shader Architecture Roadmap

### 9. Native Procedural Shader Node Editor & Live Pipeline Engine
- [ ] **Status:** Proposed / Architectural Design Phase
- [ ] **Target Component:** `src/ui/GuiManager.cpp`, `src/ui/ShaderEditor/` (new), `src/rt/WavefrontPipeline.cpp`, `src/scene/Material.hpp`, `shaders/compute/procedural_noise.glsl`, `shaders/compute/wavefront_shade_*.comp`
- [ ] **Objective:** Design and implement a high-performance, node-based procedural shader authoring engine natively within Pathways (paired with an optional Blender live-link bridge), enabling interactive procedural material authoring with instant path-traced viewport feedback, automated GLSL/Slang code generation, and strict RDNA 4 (`gfx1201`) Wave32 VGPR occupancy budgeting.
- [ ] **Architectural Motivation & Strategic Comparison:**
  - **In-Engine Native Editor vs. DCC Transpilation (Strategy A)**:
    - *Native In-Engine Editor*: Eliminates the "preview disconnect" between Blender Cycles/EEVEE and Pathways' true path-traced multi-bounce lighting, ReSTIR GI, and Upways NRC denoising. Provides instant WYSIWYG iteration directly inside the Vulkan viewport.
    - *Performance-First Principle*: Rather than generic interpretation or unconstrained graph explosion, the node compiler generates clean, specialized GLSL/Slang microkernels with dead-code elimination, constant folding, and direct mapping into Pathways' DGC material archetypes (`DIFFUSE`, `CONDUCTOR`, `DIELECTRIC`, `COMPLEX`).
    - *Blender Live-Link Hybrid*: In addition to the native ImGui node canvas, expose a localhost IPC/socket bridge enabling artists to author graphs in Blender's mature Shader Editor while Pathways hot-reloads the transpiled GLSL in real-time (< 50ms).
- [ ] **Core Architectural Pillars & Engine Integration:**
  - **1. Node Graph Representation & DAG Compiler:**
    - Lightweight, cycle-free Directed Acyclic Graph (DAG) data model (`NodeGraph`, `Node`, `Pin`, `Link`).
    - Topological sort and dead-code elimination: only nodes connected to the terminal `MaterialOutput` (Base Color, Metallic, Roughness, Normal, Transmission, Emission) are emitted.
    - Compile-time constant propagation: static inputs (scale, tint, bias) are baked as GLSL constants rather than dynamic uniform registers.
  - **2. Procedural Math & Wavefront Microkernel Library:**
    - Build upon Pathways' existing `shaders/compute/procedural_noise.glsl`:
      - Analytical 3D Simplex noise with simultaneous gradients (`simplex3D_grad`) for zero-memory, tangentless normal perturbation without finite-difference texture taps.
      - Low-register 2D/3D Voronoi (`voronoi2D`, Chebyshev/Manhattan metrics, distance-to-edge).
      - Multi-scale Fractal Brownian Motion (`fbm3D_grad`) with domain rotation.
      - ColorRamp evaluation via piecewise `mix()` / smoothstep or compact 64-entry FP16 LUTs.
  - **3. RDNA 4 (`gfx1201`) Wave32 Occupancy & VGPR Guardrails:**
    - Strict register budgeting: target $\le 40$ VGPRs for 100% Wave32 occupancy on R9700.
    - Automated background RGA profiling: during compilation, invoke `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/rga` against `gfx1201` to display real-time ISA statistics, VGPR counts, and wave occupancy directly in the editor UI.
    - Automated Archetype Sorting: analyze graph outputs to automatically assign the material to the lightest possible DGC microkernel (`wavefront_shade_diffuse.comp` vs. `wavefront_shade_complex.comp`).
  - **4. Asynchronous Vulkan 1.4 Pipeline Hot-Reloading:**
    - Non-blocking background worker thread compiles generated GLSL to SPIR-V via `libshaderc` / `glslc`.
    - Creates new `VkPipeline` and swaps handles on a safe frame boundary without stalling GPU execution rings or dropping frames.
    - Resets progressive path-trace accumulation (`frameIndex = 0`) on material parameter mutation.
- [ ] **Tasks & Implementation Steps:**
  - [ ] **Phase 1: DAG Data Model & Code Generator**:
    - Implement `ShaderGraph` data structures in C++20 with serialization (JSON / USD UsdShade).
    - Implement topological code generator emitting GLSL snippets targeting `wavefront_common.glsl`.
    - Unit tests for arithmetic simplification, vector swizzling, and dead-branch pruning.
  - [ ] **Phase 2: Vulkan Runtime Hot-Reloading Infrastructure**:
    - Add asynchronous `PipelineCompiler` in `src/vulkan/` using `glslc` / `shaderc`.
    - Implement seamless `VkPipeline` handle swap in `WavefrontPipeline::recordShade()`.
    - Integrate RGA CLI invocation to parse `analysis.csv` and report VGPR / SGPR / LDS telemetry.
  - [ ] **Phase 3: Native ImGui Node Editor Canvas**:
    - Integrate `imgui-node-editor` (or lightweight canvas) into `src/ui/GuiManager.cpp`.
    - Implement core node catalog: Coordinates (`Generated`, `Object`, `UV`, `WorldNormal`), Math / VectorMath, Mapping, SimplexNoise, Voronoi, ColorRamp, PrincipledBSDF Output.
    - Add real-time RGA occupancy badge (Green $<40$ VGPRs, Yellow $41-64$, Red $>64$).
  - [ ] **Phase 4: Blender Live-Link Companion Addon**:
    - Write lightweight Blender Python export script (`addons/pathways_livelink.py`) watching `depsgraph_update_post`.
    - Stream generated shader snippets or JSON DAG over local IPC/TCP socket into Pathways for dual-monitor authoring.
