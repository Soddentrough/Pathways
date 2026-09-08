# Pathways - Engineering TODO & Feature Roadmap

## 1. Dynamic Quality Governor & Target Frame Rate Limiter (Adaptive SPP)

- **Status:** Proposed / Backlog
- **Target Hardware:** Dual AMD Radeon AI PRO R9700 (gfx1201 / RDNA 4), single-GPU fallback
- **Priority:** Medium-High (High visual impact during interactive navigation)

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
- `DamagedHelmet` (baseline ~1.13 ms): **12–14 SPP** feasible within 15.5 ms (~3.6x noise reduction).
- `DragonAttenuation` (baseline ~4.10 ms): **3–4 SPP** feasible within 15.5 ms (~2.0x noise reduction).
- `living-room` (baseline ~6.01 ms): **2–3 SPP** feasible within 15.5 ms (~1.5x–1.7x noise reduction).

---

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

- [ ] **1. Configuration & CLI Parameters (`Config.hpp`, `Config.cpp`):**
  - Add `uint32_t target_fps = 0` (0 = uncapped, default: 0 or 60).
  - Add `bool adaptive_spp = false`.
  - Add `uint32_t min_spp = 1`, `uint32_t max_spp = 16`.
  - CLI flags: `--target-fps <int>`, `--adaptive-spp`, `--min-spp <int>`, `--max-spp <int>`.
- [ ] **2. Dynamic Quality Governor Class (`core/QualityGovernor.hpp` / `Engine.cpp`):**
  - Maintain timestamp history and EMA calculation.
  - Implement hysteresis, upgrade/downgrade guard bands, and cooldown timers.
  - Expose current target SPP, current bounce limit, and predicted headroom percentage.
- [ ] **3. Frame Pacing Engine (`Engine.cpp`):**
  - Implement high-resolution frame sleep / pacing timer before command submit or swapchain present to guarantee rock-solid frame times when the GPU finishes early.
- [ ] **4. Multi-GPU Sample Parallel Balancing (`MultiGpuManager.cpp`):**
  - Handle asymmetric SPP splits (`(spp + 1) / 2` and `spp / 2`) dynamically.
- [ ] **5. UI Controls & Telemetry (`GuiManager.cpp`):**
  - Add "Dynamic Quality Governor" section in Dear ImGui:
    - Target FPS slider (30, 60, 90, 120, 144, Uncapped).
    - Adaptive SPP toggle.
    - Live readout: `Target: 60 FPS | Dynamic SPP: 3 | Bounces: 6 | RT Time: 13.2 ms | Headroom: 15%`.
- [ ] **6. Validation & Regression Testing:**
  - Verify zero memory leaks or descriptor churn.
  - Verify stability across camera movements and scene switches (`scripts/run_headless_tests.sh`).

---

## 2. Dynamic Lights & Many-Light Importance Sampling Architecture

- **Status:** Proposed / Architectural Blueprint
- **Target Hardware:** Dual AMD Radeon AI PRO R9700 (gfx1201 / RDNA 4), Vulkan 1.4
- **Priority:** High (Directly amplifies ReSTIR DI effectiveness in production scenes)

### 2.1 Motivation & Problem Statement

Currently, Pathways represents analytical lights (point, spot, area lights) in a flat, uniform `LightsBuffer` (binding 5). While highly efficient for scenes with a few dozen lights, this model exhibits three distinct limitations:

1. **Static Light Bindings:** Light transformations and intensities are loaded statically upon scene initialization. Real-time scenes require animated lights (flickering flames, moving vehicle headlights, swinging pendants, oscillating spotlights, orbiting celestial sources).
2. **The Many-Light Sampling Problem ($N > 100$ to $10,000+$):**
   - In ReSTIR DI initial candidate generation, uniform random light picking selects each light with probability $p = 1/N$.
   - In a scene with 5,000 lights, drawing $M_{\text{init}} = 4$ candidates has a negligible probability of proposing lights that are unoccluded and physically close to the surface point.
   - Consequently, reservoirs are populated with zero-weight candidates, leading to high initial variance, slow temporal convergence, and disocclusion noise.
3. **Emissive Geometry as First-Class Lights:** Emissive mesh triangles (e.g. neon signs, digital screens, architectural light panels) are currently evaluated solely upon accidental ray intersection rather than through explicit Next-Event Estimation or ReSTIR candidate generation.

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
│               Enhanced ReSTIR DI Candidate Generation                  │
│   - Stage 1: Traverse Light Tree to draw candidate lights ~ E(x)       │
│   - Stage 2: Evaluate unshadowed target p_hat via BSDF and geometry    │
│   - Stage 3: Chao's WRS streaming accumulation + spatio-temporal merge │
│   - Stage 4: Single deferred shadow ray query (100% throughput)        │
└────────────────────────────────────────────────────────────────────────┘
```

#### A. Kinematic Light Controller & Motion Tracking
Each light in `LightsBuffer` is augmented with kinematic tracking:
- `vec4 prevPosition`: Position in frame $t-1$ (for moving point/spot/area sources).
- `vec4 prevNormal`: Orientation in frame $t-1$.
- `vec4 velocity`: Real-time motion vector $(\text{m/s})$.
- **Temporal Reprojection with Dynamic Lights:** During ReSTIR temporal reuse, candidate lights that underwent rapid translation/rotation have their historical radiance re-evaluated at the current surface point using their historical and current spatial relationship, preventing catastrophic disocclusion streaks.

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
- [ ] **4. Light Tree Traversal in ReSTIR DI (`shaders/rt/raytrace.rchit`):**
  - Implement stackless or short-stack stochastic traversal of Light Tree for candidate generation.
  - Integrate target PDF evaluation with tree selection probability $q(\text{light} \mid x)$.
- [ ] **5. Dual-GPU Synchronization (`mgpu/MultiGpuManager.cpp`):**
  - Broadcast dynamic light updates and Light Tree buffer to secondary GPU node via zero-copy host memory.
- [ ] **6. Performance & Regression Testing:**
  - Benchmark on multi-light stress test scenes (100, 1,000, and 10,000 dynamic lights).
  - Verify sub-8ms frame budget and zero validation layer warnings.

---

## 3. Meshlets & Virtualized Geometry Pipeline (Cluster DAG & Hybrid RT)

- **Status:** Proposed / Long-Term Technical Architecture
- **Target Hardware:** Dual AMD Radeon AI PRO R9700 (RDNA 4 / gfx1201), Vulkan 1.4 (`VK_EXT_mesh_shader`)
- **Priority:** High-Yield Architectural Evolution (Enables billion-triangle cinematic CAD/film assets)

### 3.1 Motivation & Problem Statement

Modern offline film assets, digital twins, and photogrammetry models easily exceed tens of millions of triangles. Traditional hardware ray tracing with static monolithic BLAS structures faces fundamental scalability walls:

1. **VRAM Footprint & Build Times:** Monolithic BLAS structures require $\sim 64\text{--}128\text{ bytes}$ per triangle in acceleration structure memory. A 100-million triangle scene consumes 6–12 GB solely for BVH storage, creating extreme memory pressure and stalling GPU scene builds.
2. **Sub-Pixel Ray Tracing Divergence:** Tracing primary camera rays against dense sub-pixel geometry causes extreme warp/wavefront divergence on RDNA 4 (Wave32), as adjacent rays hit different micro-triangles within the same pixel footprint.
3. **Absence of Continuous Geometric LOD:** Traditional level-of-detail relies on discrete mesh switches, which produce noticeable silhouette popping, shadow discontinuities, and costly BLAS rebuilds.

By adopting **Meshlets (Cluster DAGs)** and pairing **Virtualized Geometry** with **Hybrid Hardware Ray Tracing**, Pathways can achieve seamless continuous LOD with sub-pixel fidelity and bounded BVH memory overhead.

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
│                   GPU Cluster Culling & LOD Selection Pass             │
│   - Compute / Task Shader evaluates active DAG frontier per frame:     │
│       * Frustum culling (6 camera planes)                              │
│       * Backface cone culling (dot(viewDir, coneAxis) > coneAngle)     │
│       * Screen-space error metric: project(error_world) < 1.0 pixel    │
│       * Two-Phase Occlusion Culling via previous frame's Hi-Z Pyramid  │
│   - Surviving clusters compacted via Vulkan 1.4 DGC token stream       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────┴────────────────────────────────────┐
│                    Hybrid Dual-Path Geometry Pipeline                  │
├───────────────────────────────────────┬────────────────────────────────┤
│    PRIMARY RAYS: Raster / Mesh Shader │   SECONDARY / INDIRECT RAYS:   │
│         (VK_EXT_mesh_shader)          │    Hardware RT (VK_KHR_rt)     │
├───────────────────────────────────────┼────────────────────────────────┤
│ - Sub-pixel rasterization of active   │ - Dynamic Micro-BLAS built for │
│   LOD meshlet clusters                │   active cluster LODs (fast)   │
│ - Zero ray traversal divergence       │ - Coarse proxy BLAS for        │
│ - Outputs high-precision G-Buffer:    │   distant GI / reflections     │
│   HitPoint, Normal, Material, Depth   │ - Evaluates indirect diffuse,  │
│ - Direct feeding into ReSTIR DI       │   specular, and shadow rays    │
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

#### B. Two-Phase GPU Culling with Hi-Z
To eliminate invisible geometry prior to rendering:
- **Phase 1 (Conservative Visibility):** Cull clusters against previous frame's Hierarchical-Z (Hi-Z) pyramid. Visible clusters are dispatched immediately.
- **Phase 2 (Depth Verification):** Once Phase 1 renders new depth, clusters previously marked occluded are re-tested against the new Hi-Z. Newly disoccluded clusters are rendered in a secondary pass.

#### C. Hybrid Virtualized Geometry + Hardware Ray Tracing
To bridge meshlets with hardware ray tracing:
1. **Primary Ray Optimization:** Primary camera visibility is evaluated via **Mesh Shaders (`VK_EXT_mesh_shader`)** directly writing into the primary G-buffer (position, normal, depth, material). This circumvents primary ray BVH traversal overhead and resolves sub-pixel geometry without aliasing.
2. **ReSTIR DI Integration:** The ReSTIR DI temporal and spatial resampling passes consume the high-precision G-buffer directly, utilizing the packed geometry format in `ReservoirGPU.pad`.
3. **Secondary Ray Acceleration:**
   - **Dynamic Cluster BLAS:** For reflections and GI, active cluster subsets are packed into localized micro-BLAS instances built with `VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR`.
   - **Coarse Proxy BLAS:** Distant geometry uses a compact fixed-LOD proxy BLAS for ray tracing queries, reducing total BVH memory by over $80\%$.

---

### 3.3 Implementation Roadmap

- [ ] **1. Meshlet Pre-Processing Pipeline (`tools/meshlet_converter` / `third_party/meshoptimizer`):**
  - Integrate cluster decomposition: vertex cache optimization, cluster splitting, cone normal computation.
  - Implement Quadric Error Metric (QEM) simplification for multi-level cluster DAG generation.
- [ ] **2. Vulkan Mesh Shader Pipeline (`shaders/mesh/` & `vulkan/MeshletPipeline.cpp`):**
  - Enable `VK_EXT_mesh_shader` extension on AMD Radeon AI PRO R9700.
  - Implement Task Shader (`.task`) for cluster-level frustum, backface cone, and projected error culling.
  - Implement Mesh Shader (`.mesh`) for Wave32 vertex/primitive emission into framebuffer attachments.
- [ ] **3. Hierarchical-Z Pyramid Generator (`shaders/compute/hiz_generate.comp`):**
  - Parallel depth downsampling compute shader creating min-depth mipmaps for occlusion culling.
- [ ] **4. Virtualized Geometry G-Buffer Integration with Pathways Core:**
  - Connect mesh shader G-buffer output directly to ReSTIR DI reservoir allocation.
- [ ] **5. Dynamic Cluster BLAS Builder for Hardware Ray Tracing:**
  - Allocate linear GPU scratch pool for fast per-frame micro-BLAS instantiation.
  - Support hybrid ray tracing with `VK_KHR_ray_query` in compute shaders for indirect bounce calculations.
- [ ] **6. Benchmarking & Validation:**
  - Benchmark on multi-million polygon datasets (e.g. Stanford Lucy, high-detail CAD scans).
  - Verify sustained sub-8ms 120 FPS performance on Dual R9700 hardware.

