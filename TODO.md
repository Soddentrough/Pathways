# Pathways - Engineering Roadmap & TODO

## Vulkan Specification Alignment & Next-Gen Pipeline Modernization (Spec 1.4.341 → 1.4.363+)

Following the review of the Vulkan 1.4 specification updates (through 1.4.363), the following architectural upgrades and optimizations are planned once the host SDK is updated beyond 1.4.341.0.

---

### 1. Direct Buffer Device Address Commands (`VK_KHR_device_address_commands`)
- [ ] **Status:** Proposed / Pending SDK Upgrade (Ratified in Vulkan 1.4.346)
- [ ] **Target Component:** `src/rt/DGCManager.cpp`, `src/rt/ReSTIRManager.cpp`, `src/rt/NRCManager.cpp`
- [ ] **Objective:** Eliminate `VkBuffer` handle indirection across compute and ray tracing passes by switching directly to 64-bit GPU device addresses.
- [ ] **Tasks:**
  - [ ] Query and enable `VkPhysicalDeviceDeviceAddressCommandsFeaturesKHR::deviceAddressCommands` during logical device creation in `src/vulkan/VulkanContext.cpp`.
  - [ ] Replace `vkCmdDispatchIndirect` in `DGCManager::recordIndirectDispatch` fallback path with `vkCmdDispatchIndirect2KHR`, passing `argumentBuffer->getDeviceAddress()` directly.
  - [ ] Transition utility dispatches in `ReSTIRManager` and `NRCManager` to device address commands.
  - [ ] Benchmark CPU command recording overhead and dispatch latency.

---

### 2. Compute Latency Hiding via Split Barriers (`VK_EXT_shader_split_barrier`)
- [ ] **Status:** Proposed / Pending SDK Upgrade (Introduced in Vulkan 1.4.351)
- [ ] **Target Component:** `shaders/compute/wavefront_classify.comp`, `shaders/compute/wavefront_shadow.comp`
- [ ] **Objective:** Overlap global memory writes into SoA ray queues with ALU arithmetic to hide memory latency on AMD RDNA 4 (`gfx1201`).
- [ ] **Tasks:**
  - [ ] Enable `VK_EXT_shader_split_barrier` device extension and SPIR-V capability.
  - [ ] Refactor `wavefront_classify.comp` to split barrier arrival (`subgroupMemoryBarrierArrival` / `OpControlBarrierWaitINTEL`) from barrier wait.
  - [ ] Allow 3D Morton quantization and directional octant ballot calculations to execute while ray queue writes are in flight.
  - [ ] Measure Wave32 occupancy and memory stall reduction using RGP (Radeon GPU Profiler).

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
- [ ] **Status:** Proposed / Pending SDK Upgrade (Introduced in Vulkan 1.4.347)
- [ ] **Target Component:** `shaders/compute/restir_common.glsl`, sampling tables
- [ ] **Objective:** Free up VGPR registers and descriptor binding slots by embedding static tables directly into shader binaries.
- [ ] **Tasks:**
  - [ ] Embed Sobol sampling matrices, Halton sequences, and precomputed GGX distribution tables using `VK_KHR_shader_constant_data`.
  - [ ] Remove descriptor sets and buffer fetches currently dedicated to static sampling tables.
  - [ ] Verify VGPR count reductions across wavefront shading kernels.

---

### 6. Command Buffer Lifecycle Optimization (Vulkan 1.4.361 / Issue 2782)
- [ ] **Status:** Proposed / Pending SDK Upgrade
- [ ] **Target Component:** `src/vulkan/VulkanContext.cpp`, `src/mgpu/MultiGpuManager.cpp`
- [ ] **Objective:** Streamline command buffer re-recording without explicit `vkResetCommandBuffer` calls.
- [ ] **Tasks:**
  - [ ] Update frame submission rings to call `vkBeginCommandBuffer` directly from the `RECORDING` state.
  - [ ] Verify compatibility across both dual AMD Radeon AI PRO R9700 devices under ROCm/RADV.
