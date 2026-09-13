# Pathways - Engineering TODO & Feature Roadmap

## 1. Dynamic Quality Governor & Target Frame Rate Limiter (Adaptive SPP)

- **Status:** Complete (Implemented in `src/core/QualityGovernor.hpp/cpp`)
- **Target Hardware:** Dual AMD Radeon AI PRO R9700 (gfx1201 / RDNA 4), single-GPU fallback
- **Priority:** High (Fully operational in production)

---

### 1.1 Motivation & Problem Statement

On modern high-end hardware like the Dual AMD Radeon AI PRO R9700, Pathways achieves frame rates far in excess of standard display refresh rates when running at 1 primary + 4 bounce rays (e.g. 140–240+ FPS, or 1.1 ms – 6.0 ms per frame at 4K Native). 

Running uncapped at 1 SPP produces several drawbacks:
1. **Perceptual Noise during Motion:** When the camera moves, progressive accumulation resets (`m_frameIndex = 0`), leaving the user with noisy 1-SPP output despite the GPU running at full throttle.
2. **Unused Headroom:** Tracing 240 frames/second at 1 SPP wastes GPU compute that could instead be invested into multiple samples per pixel (2–4+ SPP) within a locked 60 FPS (16.67 ms) or 120 FPS (8.33 ms) frame budget.
3. **Power & Thermal Inefficiency:** Uncapped rendering draws peak power while producing redundant intermediate frames that exceed display refresh capability.

By establishing a **Target Frame Rate (e.g., 60 FPS / 16.67 ms)** and pairing it with an online **Dynamic Quality Governor**, the engine can dynamically adjust the per-frame sample count (SPP) and bounce depth to maximize fidelity and eliminate noise within the target frametime.

---

### 1.2 Ray Budget Breakdown at 60 FPS (16.67 ms)

At a target frame rate of $60\text{ FPS}$ ($\Delta t_{\text{frame}} = 16.667\text{ ms}$):
- **Fixed Pipeline Overhead:** $\sim 0.5\text{ ms} - 1.0\text{ ms}$ (ACES tonemapping compute pass, swapchain blit/copy, Dear ImGui overlay, queue submit latency).
- **Available Ray Tracing Budget:** $\Delta t_{\text{RT\_budget}} \approx 15.0\text{ ms} - 15.5\text{ ms}$.

Empirical headroom on Dual RDNA 4 GPUs at 4K Native:
### 1.3 Architectural Design & Components

```
┌────────────────────────────────────────────────────────┐
│   GPU Query Pool Timestamps (vkGetQueryPoolResults)    │
│     Measures exact hardware ray tracing execution      │
└───────────────────────────┬────────────────────────────┘
                            │ gpuRtMs (EMA filtered)
                            ▼
┌────────────────────────────────────────────────────────┐
│            Dynamic Quality Governor (CPU)              │
│  - Damped Exponential Moving Average (alpha = 0.1)     │
│  - Conservative Upgrade: projected < 0.85 * Budget     │
│  - Fast Downgrade: single frame > 0.95 * Budget        │
│  - Debounce timer (20-30 frame cooldown on upgrade)    │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────────────┐ ┌───────────────────────┐
│     Coarse: Dynamic SPP      │ │ Fine: Dynamic Bounces │
│    (1 .. 16 SPP per frame)   │ │  (2 .. 8 max bounces) │
└──────────────┬───────────────┘ └───────────┬───────────┘
               │                             │
               └──────────────┬──────────────┘
                              │ CameraUniform UBO update (64 bytes)
                              ▼
┌────────────────────────────────────────────────────────┐
│     Vulkan Hardware Ray Tracing (raytrace.rgen)        │
│          Zero pipeline stalls or rebuilds              │
└─────────────────────────────┬──────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────┐
│        High-Precision Frame Pacer / Limiter            │
│   Locks swapchain presentation to exact 60.0 FPS       │
└────────────────────────────────────────────────────────┘
```

#### A. Controller Logic & Oscillation Prevention
Because SPP is an integer ($1 \to 2$ is $+100\%$ rays), naive adjustment causes hunting/oscillation. The governor will implement:
- **Exponential Moving Average (EMA):** $\bar{t}_{n} = \alpha t_n + (1-\alpha)\bar{t}_{n-1}$ ($\alpha = 0.1$).
- **Upgrade Guard Band:** Only increment $\text{SPP} \to \text{SPP} + 1$ if:
  $$\bar{t}_{\text{pred}} = \bar{t} \times \frac{\text{SPP} + 1}{\text{SPP}} < 0.85 \times \Delta t_{\text{RT\_budget}}$$
- **Fast Emergency Downgrade:** Immediately drop SPP if any single frame breaches $0.95 \times \Delta t_{\text{RT\_budget}}$.
- **Bounce Modulation for Fractional Granularity:** When moving from $N \to N+1$ SPP would overshoot the budget, fine-tune `max_bounces` ($4 \to 6 \to 8$) to absorb intermediate headroom.

#### B. Asymmetric Multi-GPU Distribution
In `MultiGpuMode::SampleParallel`, the governor can distribute samples asymmetrically between Dual GPUs:
- Example: GPU 0 executes 2 SPP, GPU 1 executes 1 SPP $\implies$ Total 3 SPP composite.
- Allows single-SPP granularity across the dual-GPU pipeline without pipeline stalls.

#### C. Camera Motion vs. Stationary Convergence
- **Camera Moving:** Lock to 60 FPS with maximum feasible dynamic SPP to keep interactive navigation clean.
- **Camera Stationary:** Progressive accumulation continues (`m_frameIndex++`), converging to ground truth $2\times\text{--}4\times$ faster per second.

---

### 1.4 Implementation Checklist

- [x] **1. Configuration & CLI Parameters (`Config.hpp`, `Config.cpp`):**
  - Add `uint32_t target_fps = 0` (0 = uncapped, default: 0 or 60).
  - Add `bool adaptive_spp = false`.
  - Add `uint32_t min_spp = 1`, `uint32_t max_spp = 16`.
  - CLI flags: `--target-fps <int>`, `--adaptive-spp`, `--min-spp <int>`, `--max-spp <int>`.
- [x] **2. Dynamic Quality Governor Class (`core/QualityGovernor.hpp` / `Engine.cpp`):**
  - Maintain timestamp history and EMA calculation.
  - Implement hysteresis, upgrade/downgrade guard bands, and cooldown timers.
  - Expose current target SPP, current bounce limit, and predicted headroom percentage.
- [x] **3. Frame Pacing Engine (`Engine.cpp`):**
  - Implement high-resolution frame sleep / pacing timer before command submit or swapchain present to guarantee rock-solid frame times when the GPU finishes early.
- [x] **4. Multi-GPU Sample Parallel Balancing (`MultiGpuManager.cpp`):**
  - Handle asymmetric SPP splits (`(spp + 1) / 2` and `spp / 2`) dynamically.
- [x] **5. UI Controls & Telemetry (`GuiManager.cpp`):**
  - Add "Dynamic Quality Governor" section in Dear ImGui:
    - Target FPS slider (30, 60, 90, 120, 144, Uncapped).
    - Adaptive SPP toggle.
    - Live readout: `Target: 60 FPS | Dynamic SPP: 3 | Bounces: 6 | RT Time: 13.2 ms | Headroom: 15%`.
- [x] **6. Validation & Regression Testing:**
  - Verify zero memory leaks or descriptor churn.
  - Verify stability across camera movements and scene switches (`scripts/run_headless_tests.sh`).

---

## 2. Dynamic Lights & Many-Light Importance Sampling Architecture

- **Status:** Proposed / Architectural Blueprint
- **Target Hardware:** Dual AMD Radeon AI PRO R9700 (gfx1201 / RDNA 4), Vulkan 1.4
- **Priority:** High (Directly amplifies stochastic Next-Event Estimation effectiveness in production scenes)

### 2.1 Motivation & Problem Statement

Currently, Pathways represents analytical lights (point, spot, area lights) in a flat, uniform `LightsBuffer` (binding 5). While highly efficient for scenes with a few dozen lights, this model exhibits three distinct limitations:

1. **Static Light Bindings:** Light transformations and intensities are loaded statically upon scene initialization. Real-time scenes require animated lights (flickering flames, moving vehicle headlights, swinging pendants, oscillating spotlights, orbiting celestial sources).
2. **The Many-Light Sampling Problem ($N > 100$ to $10,000+$):**
   - In standard Next-Event Estimation, uniform random light picking selects each light with probability $p = 1/N$.
   - In a scene with 5,000 lights, drawing 1 candidate light uniformly has a negligible probability of proposing lights that are unoccluded and physically close to the surface point.
   - Consequently, uniform light picking suffers high variance and slow Monte Carlo convergence in many-light scenes.
3. **Emissive Geometry as First-Class Lights:** Emissive mesh triangles (e.g. neon signs, digital screens, architectural light panels) are currently evaluated solely upon accidental ray intersection rather than through explicit Next-Event Estimation.

---

### 2.2 Architectural Design

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Dynamic Light Kinematics System (CPU)                │
│   - Evaluates keyframe animations, spline paths, physical dynamics     │
│   - Computes world-space bounds, orientations, and flux updates        │
│   - Tracks previous-frame transformations for temporal reprojection    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Pipelined staging copy
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                 GPU Dynamic Light Storage (SSBO, Binding 5)            │
│   - Double-buffered host-visible memory / staging ring buffer          │
│   - Synchronized across Dual GPUs via PCIe external host memory        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               GPU Many-Light Hierarchy (Light BVH / Light Tree)        │
│   - Parallel BVH construction (Morton code sorting / LBVH)             │
│   - Node Data: Bounding Box (AABB) + Normal Cone + Total Flux (Phi)    │
│   - Traversed in O(log N) to sample lights proportional to irradiance  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               Importance Sampled Next-Event Estimation (NEE)           │
│   - Stage 1: Traverse Light Tree in O(log N) to sample light ~ E(x)   │
│   - Stage 2: Evaluate BSDF and calculate MIS sampling weights          │
│   - Stage 3: Single shadow ray query (100% throughput)                 │
└────────────────────────────────────────────────────────────────────────┘
```

#### A. Kinematic Light Controller & Motion Tracking
Each light in `LightsBuffer` is augmented with kinematic tracking:
- `vec4 prevPosition`: Position in frame $t-1$ (for moving point/spot/area sources).
- `vec4 prevNormal`: Orientation in frame $t-1$.
- `vec4 velocity`: Real-time motion vector $(\text{m/s})$.

#### B. GPU Light Tree / Hierarchical Light BVH
For scenes with large light counts ($N \in [10^2, 10^5]$), Pathways will construct a GPU-resident **Light Tree** (hierarchical cluster BVH):
- **Node Structure (32 Bytes per node):**
  - `vec3 bboxMin, bboxMax`: Bounding box of all contained light geometries.
  - `vec3 coneAxis, float coneAngle`: Normal cone bounding light emission directions.
  - `float totalFlux`: Radiant flux $\Phi = \sum_{i} \Phi_i$ within the subtree.
  - `uint childLeft, childRight`: Node hierarchy indices.
- **Importance Metric during Ray Traversal:**
  At surface hit point $x$ with normal $\mathbf{n}_x$:
  $$I(x, \text{node}) \approx \frac{\Phi_{\text{node}} \cdot \max(0, \mathbf{n}_x \cdot \mathbf{d}_{\text{node}}) \cdot \cos(\theta_{\text{cone}})}{\max(d_{\text{min}}^2, \epsilon)}$$
- Traversing the tree in $O(\log N)$ stochastically guides candidate proposals toward high-irradiance sources, replacing uniform $1/N$ sampling with physically driven importance sampling.

#### C. Emissive Triangle Acceleration Buffer
- Meshes marked with non-zero emissive materials will have their triangles indexed into a dedicated `EmissiveTrianglesBuffer`.
- Small emissive meshes are clustered into proxy area lights; high-polygon emissive fixtures are sampled hierarchically via surface-area CDFs with power weighting.

---

### 2.3 Implementation Roadmap

- [ ] **1. Kinematic Dynamic Light Core (`scene/Light.hpp`, `scene/SceneData.hpp`):**
  - Add `prevPosition`, `prevNormal`, `angularVelocity`, and `linearVelocity` to `LightGPU`.
  - Implement dynamic light update pass in `Engine::updateScene(float deltaTime)`.
  - Add staging buffer / host-visible transfer ring buffer for per-frame light state synchronization.
- [ ] **2. Animated Light Sequences & Scene Loader (`scene/GltfLoader.cpp`):**
  - Parse KHR_lights_punctual glTF nodes attached to animated scene hierarchy nodes.
  - Support user-controllable interactive lights (orbiting sun, point light gizmo in Dear ImGui).
- [ ] **3. GPU Light Tree Builder (`shaders/compute/light_tree_build.comp`):**
  - Compute 30-bit Morton codes for light bounds; parallel radix sort in compute.
  - Build Linear BVH (LBVH) nodes with bounding box, normal cone, and radiant flux hierarchy.
- [ ] **4. Light Tree Traversal in Stochastic NEE (`shaders/rt/raytrace.rchit` & Wavefront Shade):**
  - Implement stackless or short-stack stochastic traversal of Light Tree for importance-sampled NEE.
  - Integrate target PDF evaluation with tree selection probability $q(\text{light} \mid x)$.
- [ ] **5. Dual-GPU Synchronization (`mgpu/MultiGpuManager.cpp`):**
  - Broadcast dynamic light updates and Light Tree buffer to secondary GPU node via zero-copy host memory.
- [ ] **6. Performance & Regression Testing:**
  - Benchmark on multi-light stress test scenes (100, 1,000, and 10,000 dynamic lights).
  - Verify sub-8ms frame budget and zero validation layer warnings.

---

## 3. Cluster DAG & Virtualized Geometry Ray Traversal (Pure Path Traced Micro-BVH)

- **Status:** Proposed / Long-Term Technical Architecture
- **Target Hardware:** Dual AMD Radeon AI PRO R9700 (RDNA 4 / gfx1201), Vulkan 1.4 (`VK_KHR_acceleration_structure`, `VK_KHR_ray_query`)
- **Priority:** High-Yield Architectural Evolution (Enables billion-triangle cinematic CAD/film assets in a pure path tracer)

### 3.1 Motivation & Problem Statement

Modern offline film assets, digital twins, and photogrammetry models easily exceed tens of millions of triangles. Traditional hardware ray tracing with static monolithic BLAS structures faces fundamental scalability walls:

1. **VRAM Footprint & Build Times:** Monolithic BLAS structures require $\sim 64\text{--}128\text{ bytes}$ per triangle in acceleration structure memory. A 100-million triangle scene consumes 6–12 GB solely for BVH storage, creating extreme memory pressure and stalling GPU scene builds.
2. **Sub-Pixel Ray Tracing Divergence:** Tracing rays against dense sub-pixel geometry causes extreme warp/wavefront divergence on RDNA 4 (Wave32), as adjacent rays traverse deep monolithic BVH hierarchies with inconsistent leaf sizes.
3. **Absence of Continuous Geometric LOD:** Traditional level-of-detail relies on discrete mesh switches, which produce noticeable silhouette popping, shadow discontinuities, and costly full-BLAS rebuilds.

By adopting a **Pure Path-Traced Cluster DAG Architecture** with **Dynamic Micro-BVH Allocation**, Pathways preserves its 100% path-traced foundation across all camera and indirect bounces while achieving continuous LOD, sub-pixel fidelity, and bounded BVH memory overhead without rasterization or screen-space compromises.

---

### 3.2 Architectural Design

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Offline / Pre-Pass Meshlet Clustering                │
│   - Mesh partitioned into bounded clusters: 64-128 vertices, 64-126 tris│
│   - Cluster Directed Acyclic Graph (DAG) for continuous LOD tree       │
│   - Each cluster carries: Bounding Sphere, Normal Cone, Geometric Error│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│             GPU Cluster Continuous LOD Selection & Compaction          │
│   - Compute Shader evaluates active DAG frontier per frame:            │
│       * Ray footprint & projected geometric error metric               │
│       * Continuous LOD transition blending without crack artifacts     │
│       * Hierarchical cluster frustum & backface bounding-cone culling  │
│   - Surviving clusters compacted via Vulkan 1.4 DGC token stream       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────┴────────────────────────────────────┐
│              Pure Path Traced Acceleration Hierarchy                   │
├───────────────────────────────────────┬────────────────────────────────┤
│    PRIMARY & SPECULAR RAYS:           │   DIFFUSE & SHADOW RAYS:       │
│    Active Cluster Micro-BVH (Fast)    │   Coarse Proxy Cluster BLAS    │
├───────────────────────────────────────┼────────────────────────────────┤
│ - Dynamic Micro-BLAS built for active │ - Compact fixed-LOD proxy BLAS │
│   fine-LOD clusters (linear scratch)  │   for incoherent secondary GI  │
│ - Hardware ray queries / RTP traversal│ - Drastically cuts traversal   │
│ - 100% physically correct primary hits│   divergence on secondary rays │
│ - Direct input to Wavefront queues    │ - Evaluates MIS NEE & shadows  │
└───────────────────────────────────────┴────────────────────────────────┘
```

#### A. Meshlet Cluster DAG Generation
- **Cluster Properties:** Meshes are decomposed into localized clusters of $\le 128$ vertices and $\le 126$ triangles, optimized for RDNA 4 Wave32 execution:
  - 1 cluster $\implies$ 2–4 Wave32 wavefronts for parallel primitive processing.
- **DAG Construction:**
  1. Leaf clusters (LOD 0) represent raw high-resolution geometry.
  2. Adjacent clusters are grouped into pairs/quads, simplified (quadric error metric decimation), and partitioned into parent clusters (LOD 1).
  3. Recursion continues until the root cluster is formed.
  4. Each cluster stores an error sphere $(c, r)$ and bounding cone $(\mathbf{a}, \theta)$ ensuring crack-free boundary matching between adjacent LODs.

#### B. GPU Continuous LOD Selection Pass
To eliminate unneeded geometry and preserve uniform ray traversal depth:
- Compute shaders evaluate the cluster DAG frontier against camera view parameters and ray footprints.
- Active LOD clusters are compacted into dynamic linear GPU buffers via device-generated commands, ensuring only visible, perceptually necessary detail is built into the acceleration hierarchy.

#### C. Pure Path Traced Cluster BVH Traversal
Preserving Pathways' pure path tracer principles:
1. **No Rasterization or Screen-Space Buffers:** All primary camera visibility, secondary bounces, and shadows are resolved exclusively via pure ray tracing (`VK_KHR_ray_query` and hardware ray tracing pipelines).
2. **Dynamic Cluster BLAS:** Active cluster subsets are packed into localized micro-BLAS instances built per frame with `VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR` using linear GPU scratch memory.
3. **Coarse Proxy BLAS:** Incoherent diffuse and distant GI rays traverse a compact, simplified cluster proxy BLAS, reducing BVH memory footprint by over $80\%$ without visual divergence.
4. **Wavefront Integration:** Direct lighting and multi-bounce indirect scattering operate directly on compacted wavefront ray queues with zero scratch spilling.

---

### 3.3 Implementation Roadmap

- [ ] **1. Meshlet Pre-Processing Pipeline (`tools/meshlet_converter` / `third_party/meshoptimizer`):**
  - Integrate cluster decomposition: vertex cache optimization, cluster splitting, cone normal computation.
  - Implement Quadric Error Metric (QEM) simplification for multi-level cluster DAG generation.
- [ ] **2. GPU Cluster Selection Compute Shader (`shaders/compute/cluster_select.comp`):**
  - Implement continuous LOD selection, projected error evaluation, and cluster compaction.
  - Generate DGC indirect dispatch tokens for BVH building and ray traversal.
- [ ] **3. Dynamic Micro-BLAS Acceleration Structure Builder:**
  - Allocate linear GPU scratch pool for fast per-frame micro-BLAS instantiation with Vulkan 1.4 acceleration structures.
  - Streamline TLAS updates referencing active cluster BLAS instances.
- [ ] **4. Dual-Scale BVH Traversal in Megakernel & Wavefront Pipelines:**
  - Integrate cluster micro-BVH ray queries for primary and coherent secondary rays.
  - Integrate coarse proxy traversal for incoherent multi-bounce diffuse GI.
- [ ] **5. Benchmarking & Validation:**
  - Benchmark on multi-million polygon datasets (e.g. Stanford Lucy, high-detail CAD scans).
  - Verify sustained 60+ FPS performance on Dual R9700 hardware under 100% path tracing.

---

## 4. Next-Generation Scene Ingestion: glTF 2.1 (64-Bit GLB Container) & OpenUSD Binary Crate (`.usdc`)

- **Status:** Proposed / Architectural Specification & Backlog
- **Target Hardware:** Dual AMD Radeon AI PRO R9700 (gfx1201 / RDNA 4), 64 GB System RAM, 64 GB VRAM
- **Priority:** High (Unblocks massive photogrammetry scans, digital twins, and cinematic film assets exceeding the 4 GiB 32-bit barrier)

---

### 4.1 Motivation & Problem Statement

Pathways currently relies on **glTF 2.0 (`.gltf`, `.glb`)** via `cgltf v1.15` as its sole runtime file-based ingestion pipeline. While effective for lightweight real-time assets, this architecture exhibits critical limitations when handling industrial-grade, dense datasets:

1. **The 4 GiB GLB File Ceiling:**
   - The glTF 2.0 binary format (`GlbVersion = 2`) stores chunk lengths and buffer offsets as 32-bit unsigned integers (`uint32_t`).
   - Photogrammetry scans, large architectural models, CAD datasets, and complex VFX scenes easily exceed 4 GiB, causing `cgltf` to reject the file or truncate offsets.
   - The **glTF 2.1** specification (released June 2026) addresses this by introducing Binary Format Version 3 with **64-bit chunk length fields and offsets**, lifting the 4 GiB ceiling.
2. **Absence of 64-Bit Accessor Types (`DOUBLE`, `INT64`):**
   - glTF 2.0 only supports component types up to 32-bit (`FLOAT`, `UNSIGNED_INT`).
   - Large-world coordinate spaces (planetary GIS, aerospace simulations, expansive cityscapes) require double-precision floating-point coordinates (`DOUBLE`) to prevent floating-point precision jitter near distant geometry.
   - glTF 2.1 officially adds `DOUBLE` (64-bit float) and `INT64` / `UINT64` accessors.
3. **Monolithic Asset Delivery vs. Production Binary Crate (`.usdc`):**
   - glTF files require parsing JSON hierarchies and copying buffers, creating substantial CPU ingestion latency on multi-gigabyte models.
   - Pixar's **OpenUSD Binary Crate (`.usdc`)** format provides a high-performance, memory-mapped, zero-copy binary serialization format specifically designed for rapid streaming of multi-million polygon geometries and material graphs without deserialization bottlenecks.

---

### 4.2 Architectural Design & Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                          Unified High-Capacity Scene Ingestion                         │
└────────────────────────────────────────────────────────────────────────────────────────┘

    glTF 2.1 (.gltf / .glb v3 64-bit)                OpenUSD Binary Crate (.usdc / .usdz)
    - 64-bit Chunk Headers (>4 GiB)                  - Zero-Copy Memory-Mapped Tables
    - DOUBLE / INT64 Accessors                       - VtArray<GfVec3d> Double Precision
                   │                                                  │
                   ▼                                                  ▼
    ┌──────────────────────────────┐                 ┌────────────────────────────────┐
    │     Modern glTF 2.1 Parser   │                 │     OpenUSD Core / Crate API   │
    │     (fastgltf / cgltf-ng)    │                 │   (pxr::UsdStage, pxr::UsdGeom)│
    └──────────────┬───────────────┘                 └────────────────┬───────────────┘
                   │                                                  │
                   └────────────────────────┬─────────────────────────┘
                                            │
                                            ▼
    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │                      64-Bit Geometry Adapter & Normalization                       │
    │  - Camera-Relative Translation: x_rel = float(x_world - camera_world)              │
    │  - Float64 -> Float32 demotion for Vulkan Hardware Ray Tracing BLAS                │
    │  - Multi-BLAS Spatial Chunking (partitions meshes exceeding 2^32-1 vertex limit)   │
    └───────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │                              Pathways Core Scene Data                              │
    │  - TriangleGPU Buffer (std430 SSBO: 160 B/tri, 32-bit floats, 32-bit material IDs) │
    │  - MaterialGPU Buffer (PBR, Transmission, Clearcoat, Emissive)                     │
    │  - AccelerationStructureManager (VK_FORMAT_R32G32B32_SFLOAT, VK_INDEX_TYPE_UINT32)│
    └───────────────────────────────────────┬────────────────────────────────────────────┘
                                            │
                                            ▼
    ┌────────────────────────────────────────────────────────────────────────────────────┐
    │                    Dual AMD Radeon AI PRO R9700 Hardware RT                        │
    │  - Native 32-bit RDNA 4 Ray Accelerators (Wave32 Traversal & Intersection)         │
    │  - Multi-GPU Zero-Copy Shared Host Memory Pipelining                               │
    └────────────────────────────────────────────────────────────────────────────────────┘
```

#### A. glTF 2.1 Specification Support
- **64-Bit GLB Container (Version 3):**
  - Parse binary header: Magic `0x46546C67`, Version `3`, followed by 64-bit unsigned integer chunk lengths.
  - Remove memory mapping limits, enabling seamless loading of 10–50+ GB single-file `.glb` captures.
- **Extended Accessors:**
  - Support `componentType = 5130` (`DOUBLE` / 64-bit IEEE 754 float) and `5128` (`INT64` / `UINT64`).
  - Convert double-precision vertex attributes on ingest into camera-relative single-precision floats (`float32`) for compatibility with Vulkan hardware RT.
- **Implicit Shapes & BVH Extensions:**
  - Parse glTF 2.1 native implicit primitives (analytical boxes, spheres, capsules, cylinders) directly into Pathways analytical intersection pipelines (`SphereGPU` / procedural primitives).

#### B. OpenUSD Binary Crate (`.usdc`) Ingestion
- **Memory-Mapped Direct Ingestion:**
  - Implement `UsdLoader::loadSceneData(filepath)` utilizing the OpenUSD C++ API (`pxr::UsdStage::Open`).
  - Stream `UsdGeomMesh` points (`VtArray<GfVec3f>` and `VtArray<GfVec3d>`), face vertex counts, and face vertex indices directly from the binary Crate file into Vulkan staging buffers without temporary allocations.
- **MaterialX & UsdPreviewSurface Translation:**
  - Map `UsdPreviewSurface` inputs (`diffuseColor`, `metallic`, `roughness`, `clearcoat`, `ior`, `opacity`) into `MaterialGPU` parameters.
  - Resolve asset paths relative to the USD root layer with support for packaged `.usdz` ZIP archives.

#### C. Handling Ultra-Dense 64-Bit Geometry in Vulkan Ray Tracing
- **The Vulkan 32-Bit RT Barrier:**
  - Hardware Ray Tracing units (`VK_KHR_acceleration_structure`) and SPIR-V ray tracing builtins (`gl_PrimitiveID`) operate strictly on 32-bit indices and 32-bit floating-point bounding boxes/vertices.
  - Neither Vulkan nor GPU RT hardware supports 64-bit index buffers (`VK_INDEX_TYPE_UINT64` does not exist in the specification).
- **Multi-BLAS Partitioning Scheme:**
  - If a dense CAD or photogrammetry mesh contains more than $2^{32}-1$ indices (~1.43 billion triangles), Pathways will automatically partition the geometry across spatial clusters into multiple individual BLAS structures under a unified top-level acceleration structure (TLAS).
- **Camera-Relative Jitter Elimination:**
  - High-precision world coordinates ($\mathbf{x}_{\text{world}} \in \mathbb{R}^3$, 64-bit float) are translated to camera-relative space before uploading to `VkBuffer`:
    $$\mathbf{x}_{\text{rel}} = \text{float32}(\mathbf{x}_{\text{world}} - \mathbf{x}_{\text{cam}})$$
  - This eliminates numerical precision degradation at large coordinates while maintaining full hardware RT throughput on RDNA 4.

---

### 4.3 Implementation Roadmap & Checklist

- [ ] **1. Parser Modernization for glTF 2.1 (`third_party/fastgltf` or `cgltf` update):**
  - Integrate `fastgltf` (modern C++20 glTF parser) supporting glTF 2.0 & 2.1, GLB version 3 (64-bit chunks), SIMD JSON decoding, and memory-mapped file buffers.
  - Update `GltfLoader.cpp` to parse 64-bit buffer views and chunk lengths without truncation.
- [ ] **2. glTF 2.1 Extended Accessor Decoding (`src/scene/GltfLoader.cpp`):**
  - Implement decoders for `DOUBLE` (64-bit float) vertex positions and `INT64` / `UINT64` indices.
  - Implement automatic CPU demotion to 32-bit floats with camera-relative centering.
- [ ] **3. OpenUSD Binary Crate Loader (`src/scene/UsdLoader.hpp` / `.cpp`):**
  - Create native USD stage reader (`UsdLoader`) linked against OpenUSD Core C++ libraries.
  - Ingest `UsdGeomMesh` and `UsdGeomSubset` data directly into Pathways `SceneData`.
  - Extract UVs, normals, tangents, and material bindings from `UsdShadeMaterial`.
- [ ] **4. Ultra-Dense Mesh Partitioning (`src/scene/MeshPartition.hpp`):**
  - Implement spatial k-d tree or AABB clustering to split geometries exceeding $2^{31}$ vertices into sub-mesh primitives.
  - Register partitioned primitives as separate BLAS geometries or multi-BLAS TLAS instances.
- [ ] **5. Dynamic Scene Registry & GUI Extension (`src/scene/SceneRegistry.cpp`, `src/ui/GuiManager.cpp`):**
  - Extend scene discovery scanner to recognize `.usdc`, `.usd`, and `.usdz` extensions alongside `.gltf` and `.glb`.
  - Display container version (`glTF 2.0` vs `glTF 2.1 (64-bit)` vs `OpenUSD Crate`) and precision mode in the Dear ImGui Active Scene card.
- [ ] **6. Performance & Scale Validation:**
  - Ingest and benchmark a multi-gigabyte (>4 GiB) glTF 2.1 model and a production `.usdc` scene.
  - Verify zero GPU memory leaks, sub-8ms frame budget on Dual R9700 GPUs, and clean Vulkan validation pass.

---

## 5. Multi-GPU Load Balancing & Ray Distribution Optimization

- **Status:** Identified in 4K DGC Benchmark Battery / Follow-Up Backlog
- **Target Hardware:** Dual AMD Radeon AI PRO R9700 (gfx1201 / RDNA 4), Vulkan 1.4
- **Priority:** Medium-High (Directly unlocks linear 1.9x–2.0x scaling across all complex scene topologies)

---

### 5.1 Empirical Scaling Discrepancy & Problem Statement

In the comprehensive 4K native 80-run benchmark battery conducted on Dual AMD Radeon AI PRO R9700 hardware, multi-GPU scaling exhibited sharp disparities depending on scene topology, material concentration, and ray propagation depth:

| Scene | Single-GPU (Technique D) | Multi-GPU (Technique D) | Speedup | Scaling Efficiency | Bottleneck Profile |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Cornell Box** | 7.53 ms | 3.89 ms | **1.94x** | **97.0%** | Uniform diffuse geometry; symmetric workload |
| **Cornell Caustic** | 7.28 ms | 4.00 ms | **1.82x** | **91.0%** | Concentrated dielectric; minor tile variance |
| **Coffee Maker** | 8.44 ms | 3.81 ms | **2.21x** | **110.5%** | Cache-bound single GPU; split working set fits L2/L3 |
| **Living Room** | 9.33 ms | 5.73 ms | **1.63x** | **81.5%** | Moderate spatial variance in bounce depth |
| **Classroom** | 12.68 ms | 7.87 ms | **1.61x** | **80.5%** | Moderate occlusion variance across view |
| **Kitchen Extended** | 13.54 ms | 8.75 ms | **1.55x** | **77.5%** | Severe structural imbalance (dense cabinets vs open counters) |
| **Dragon Attenuation** | 5.72 ms | 3.83 ms | **1.49x** | **74.5%** | Centered high-bounce glass; sky backdrop terminates early |
| **Bistro Interior** | 3.16 ms | 2.11 ms | **1.50x** | **75.0%** | Fast frame time; PCIe sync/compositing latency wall |

Three scenes in particular demonstrated sub-80% scaling efficiency: **`Kitchen Extended` (77.5%)**, **`Dragon Attenuation` (74.5%)**, and **`Bistro Interior` (75.0%)**. Analysis reveals three distinct root causes:

1. **Spatial Ray Divergence & Workload Asymmetry (*Kitchen Extended*, *Dragon Attenuation*):**
   - In static screen-space tile or split-frame distribution (e.g. half-screen or uniform checkerboard), geometric complexity and bounce propagation are rarely uniform across the image.
   - In *Dragon Attenuation*, the central region contains dense glass transmission with up to 4 refraction/reflection bounces, whereas peripheral tiles immediately hit the environment/skybox and terminate after 1 bounce. If one GPU processes the center and the other the borders, one GPU idles while the other grinds through complex paths.
   - In *Kitchen Extended*, one half of the viewport features multi-surface metallic reflections and dense cabinet occlusion, while the other features broad diffuse walls and countertops. The busy GPU determines total frame time, creating large synchronization stalls at the inter-GPU composite barrier.

2. **Amdahl's Law & Fixed PCIe Latency Floor (*Bistro Interior*):**
   - At 4K native, single-GPU execution of *Bistro Interior* is exceptionally fast (3.16 ms, or ~316 FPS).
   - Multi-GPU compositing incurs fixed overhead: fence signaling, inter-GPU DMA/PCIe transfer of the secondary half-frame or tile buffers (~0.4–0.6 ms), and composite compute pass submission (~0.1 ms).
   - When total GPU compute drops below 2.5 ms, a 0.5 ms fixed transfer floor accounts for >20% of the entire frame time, mathematically capping maximum scaling at $\approx 1.5\times$.

3. **Wavefront Compaction & Queue Disparity Across GPUs:**
   - In the wavefront pipeline, rays are sorted into specialized archetype queues (`diffuse`, `dielectric`, `conductor`, `complex`).
   - If GPU 0 processes 90% dielectric rays and GPU 1 processes 90% diffuse rays, GPU 0 executes 4 indirect bounce dispatches with heavy register usage, while GPU 1 completes early and waits at the Vulkan queue synchronization barrier.

---

### 5.2 Proposed Architecture & Solutions

```
┌────────────────────────────────────────────────────────────────────────┐
│               Dynamic Multi-GPU Workload Optimization                  │
└────────────────────────────────────────────────────────────────────────┘

    Option A: Fine-Grained Dynamic Work Stealing (Tile Queue)
    ┌────────────────────────────────────────────────────────────────────┐
    │ Atomic Tile Queue in Host-Visible / PCIe P2P Memory                │
    │ [ Tile 0 ] [ Tile 1 ] [ Tile 2 ] ... [ Tile N-1 ] (e.g. 64x64 px)  │
    └──────────────────┬───────────────────────────────┬─────────────────┘
                       │ atomicAdd                     │ atomicAdd
                       ▼                               ▼
               ┌───────────────┐               ┌───────────────┐
               │     GPU 0     │               │     GPU 1     │
               │  Pulls tiles  │               │  Pulls tiles  │
               │ continuously  │               │ continuously  │
               └───────────────┘               └───────────────┘
                       ▲                               ▲
                       └───────────────┬───────────────┘
                                       │ Automatically load balances
                                       │ spatial ray variance!

    Option B: Online Dynamic Split-Boundary Feedback Governor
    ┌────────────────────────────────────────────────────────────────────┐
    │ GPU Timestamp Feedback: t(GPU0) vs t(GPU1)                         │
    │ If t(GPU0) > t(GPU1) + delta: Shift split line X_split -= step     │
    │ If t(GPU1) > t(GPU0) + delta: Shift split line X_split += step     │
    └────────────────────────────────────────────────────────────────────┘

    Option C: Sample-Parallel Domain Decomposition (SPP >= 2)
    ┌────────────────────────────────────────────────────────────────────┐
    │ Both GPUs render full 4K frame at N/2 SPP (Identical workload!)    │
    │ Zero spatial imbalance; resolved via fast FP16 accumulator blend   │
    └────────────────────────────────────────────────────────────────────┘
```

#### A. Fine-Grained Dynamic Work-Stealing Tile Queue
- Replace static viewport splits with a pool of $64 \times 64$ or $32 \times 32$ pixel workgroup tiles.
- Workgroups atomically claim tile indices via an atomic counter located in PCIe peer-to-peer or host-pinned memory (`VK_MEMORY_PROPERTY_HOST_COHERENT_BIT`).
- The GPU handling simpler regions naturally processes more tiles, while the GPU encountering heavy dielectric ray bounces processes fewer tiles, automatically balancing active compute time to within 2–3% variance.

#### B. Online Adaptive Split-Boundary Feedback
- For split-frame rendering without dynamic work stealing:
  - Query hardware execution timestamps (`vkGetQueryPoolResults`) on GPU 0 and GPU 1.
  - Dynamically slide the split-screen dividing line per frame:
    $$\Delta X_{\text{split}} = k \cdot (t_{\text{GPU1}} - t_{\text{GPU0}})$$
  - Shifts screen area away from the overburdened GPU until both GPUs finish ray tracing simultaneously.

#### C. Asymmetric Sample-Parallel Rendering (`MultiGpuMode::SampleParallel`)
- For interactive rendering when SPP $\ge 2$ (or when paired with the Dynamic Quality Governor):
  - Each GPU renders the full screen using half the total sample budget (e.g. 1 SPP on GPU 0, 1 SPP on GPU 1).
  - Because both GPUs traverse identical spatial rays across the entire scene, load imbalance drops to 0.0%.
  - High-bandwidth P2P blending merges the two FP16 HDR buffers with near 2.0x linear scaling.

#### D. Overlapped Async Transfer & P2P Compositing
- Double-buffer the inter-GPU transfer staging buffers.
- Overlap the PCIe transfer of bounce $k-1$ results with the primary ray generation of bounce $k$, hiding the 0.4–0.6 ms transfer latency beneath compute passes.

---

### 5.3 Implementation Roadmap & Action Items

- [x] **1. Per-GPU Execution Telemetry in ImGui (`src/ui/GuiManager.cpp`, `src/mgpu/MultiGpuManager.cpp`):**
  - Expose individual GPU 0 and GPU 1 ray tracing timestamps side-by-side in Dear ImGui.
  - Display real-time workload imbalance metric: $\Delta t_{\text{imbalance}} = |t_{\text{GPU0}} - t_{\text{GPU1}}|$.
  - Completed: Live dual-GPU profiling window displays per-GPU execution times, transfer times, link mode, and imbalance deltas.

- [x] **2. P2P Direct BAR Transfer Optimization (`src/vulkan/VulkanContext.cpp`, `src/mgpu/MultiGpuManager.cpp`):**
  - Utilized `VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf` for zero-copy VRAM-to-VRAM peer transfer over PCIe BAR.
  - Completed & Empirically Validated:
    - Primary bottleneck identified: fixed host memory round-trip latency (`parallelMemcpy` over PCIe host-visible staging).
    - Tonemap & Merge pass latency reduced from ~0.50 ms down to **0.090 – 0.135 ms** across all native 4K scenes (3840x2160).
    - Host memory double-hops completely eliminated.
    - Verified via automated unit test [`tests/test_p2p_direct_bar.cpp`](file:///home/naoki/Development/Pathways/tests/test_p2p_direct_bar.cpp).

- [x] **3. Architectural Evaluation of Spatial Partitioning vs. Checkerboard Tiling (Post-Mortem):**
  - **Hypothesis**: Scaling limitations in *Kitchen Extended*, *Dragon Attenuation*, and *Bistro Interior* were initially hypothesized to stem from spatial ray variance in screen space.
  - **Findings**:
    1. Pathways was already employing fine-grained **$64 \times 64$ Checkerboard Tiling** (2,040 alternating tiles across $3840 \times 2160$). Both GPUs already received an almost identical sample of dense and light geometric regions.
    2. An Online Adaptive Split-Boundary Governor was implemented and evaluated against Checkerboard Tiling across all three scenes. The delta was marginal:
       - *Kitchen Extended*: 9.45 ms (Checkerboard) vs 9.09 ms (Split) $\to$ 3.8% delta.
       - *Dragon Attenuation*: 4.22 ms (Checkerboard) vs 4.17 ms (Split) $\to$ 1.1% delta.
       - *Bistro Interior*: 2.21 ms (Checkerboard) vs 2.22 ms (Split) $\to$ 0.4% delta.
    3. **Conclusion**: Spatial ray divergence was not the primary bottleneck. The scaling ceiling on short frames is governed by Amdahl's Law on fixed pipeline overhead (queue synchronization, presentation, display master tasks on GPU 0).
    4. **Decision**: Retained the clean, battle-tested $64 \times 64$ Checkerboard Tiling as the default 1-SPP mode; discarded the complex split-governor logic to keep the codebase lean and maintainable.

- [x] **4. Sample-Parallel Domain Decomposition (`MultiGpuMode::SampleParallel`):**
  - Retained for multi-sample rendering ($\ge 2$ SPP): each GPU renders full resolution at $N/2$ SPP and merges in a 0.09 ms FP16 accumulator pass.
  - Yields up to 1.57x scaling (78.5% efficiency) on *Bistro Interior* with zero spatial artifacts.

- [x] **5. Dynamic Work-Stealing Tile Queue Analysis:**
  - Evaluated cross-GPU atomic queue architecture via imported DMA-BUF storage buffers (`scratch/test_p2p_atomic.cpp`).
  - Finding: Discrete PCIe 4.0/5.0 interfaces without coherent xGMI/CXL fabric lack hardware-snooped atomic caches across separate physical GPUs; remote atomic contention induces substantial memory bus serialization. Discarded in favor of static checkerboard + DMA-BUF P2P.

---

## 6. Full Multi-Instance Object-Space Graph & Per-Asset BLAS Instancing

- **Status:** Proposed / Architectural Roadmap
- **Target Hardware:** Dual AMD Radeon AI PRO R9700 (RDNA 4 / `gfx1201`), Vulkan 1.4
- **Priority:** High (Next-generation scene scalability and dynamic asset support)

### 6.1 Motivation & Architectural Evolution

Pathways currently employs a pre-transformed world-space BLAS design: during scene loading, all mesh primitives are baked into global world coordinates and merged into a monolithic vertex buffer. While this enables direct 1:1 hardware primitive indexing (`triangles[primId]`), it introduces two fundamental architectural limitations at scale:
1. **No Geometry Deduplication (VRAM Inflation):** In complex architectural and game environments containing repeated assets (e.g. 50 identical dining chairs, 20 light fixtures, 10,000 screws or fence links), vertices and acceleration structures are fully duplicated in VRAM, increasing memory footprint by $5\times$ to $20\times$.
2. **Static Geometry Lock-In:** Because vertices are baked in world space, moving or animating any object requires rebuilding or refitting the global BLAS ($O(N \log N)$), which is prohibitive for real-time framerates.

### 6.2 Target Architecture: Multi-BLAS Asset Cache & TLAS Instance Graph

```
┌─────────────────────────────────────────────────────────────┐
│             Asset Cache (Unique Local-Space BLASes)         │
│  - BLAS 0: Table Asset (Local Object Space)                 │
│  - BLAS 1: Chair Asset (Local Object Space)                 │
│  - BLAS 2: Glass Cup (Non-Opaque / Transmission)            │
└──────────────────────────────┬──────────────────────────────┘
                               │ Referenced by device address
                               ▼
┌─────────────────────────────────────────────────────────────┐
│          Top-Level Acceleration Structure (TLAS)            │
│  - Instance 0: Chair #1 -> Transform M0, customIndex = 0    │
│  - Instance 1: Chair #2 -> Transform M1, customIndex = 0    │
│  - Instance 2: Table    -> Transform M2, customIndex = 1    │
│  - Instance 3: Cup      -> Transform M3, customIndex = 2    │
└─────────────────────────────────────────────────────────────┘
```

#### Key Technical Requirements:
1. **Local Object-Space BLAS Cache:**
   - Unique mesh primitives are built into dedicated local-space BLASes once upon load.
   - Opaque assets set `VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR`; transparent/alpha assets are flagged accordingly.
2. **TLAS Instance Transform Matrix:**
   - Instances carry dynamic $3 \times 4$ affine transforms (`VkTransformMatrixKHR`), enabling real-time rigid body movement, physics, and camera-independent motion by simply writing 64 bytes to the instance buffer.
3. **Shader World-Space Reconstruction:**
   - Shaders utilize `rayQueryGetIntersectionObjectToWorldEXT` or `gl_ObjectToWorld3x4EXT` to dynamically transform local surface normals and tangents into world space:
     $$\mathbf{N}_{\text{world}} = \text{normalize}\left(\mathbf{M}^{-T} \mathbf{N}_{\text{local}}\right)$$
   - Primitive indexing is resolved via `customIndex` (asset descriptor index) + `gl_PrimitiveID`.
4. **Hardware Ray Masking:**
   - Utilize 8-bit TLAS instance masks (`mask = 0x01` for opaque, `0x02` for non-opaque) to cull non-opaque objects completely from primary or shadow ray traversal when appropriate.

# ANIMATIONS

1. We should support animated geometry:
https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/AnimatedColorsCube

2. Animated UVs:
https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/AnimationPointerUVs

# COMPRESSION

1. KHR_mesh_quantization
https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/MeshoptCubeTest
