# Universal Scene Description (OpenUSD) Architecture & Integration Blueprint

## 1. Executive Summary & Strategic Rationale

Pathways currently utilizes **glTF 2.0 (`.gltf`, `.glb`)** via `cgltf` as its primary scene ingestion pipeline. While glTF is an excellent interchange standard for web and real-time assets, it is fundamentally a static delivery format:
- It lacks non-destructive composition, multi-layer asset referencing, variant switching, and hierarchical overrides.
- Its material and lighting specifications are constrained to simplified transmission/PBR models without native support for cinematic lighting hierarchies or complex shading graphs.

**Universal Scene Description (OpenUSD)**, originated by Pixar and standardized by the Alliance for OpenUSD (AOUSD: Pixar, Apple, NVIDIA, Autodesk, Adobe), is the undisputed standard for visual effects, feature animation, architectural visualization, digital twins, and industrial robotics.

By integrating OpenUSD support, Pathways will achieve two primary capabilities:
1. **Direct High-Fidelity Asset Ingestion (`UsdLoader`):** Native loading of `.usd`, `.usda` (ASCII), `.usdc` (binary Crate), and `.usdz` (packaged archives) into Pathways' pure Vulkan 1.4 ray tracing engine.
2. **Hydra Render Delegate (`hdPathways`):** Transforming Pathways into an interactive, real-time hardware ray tracing viewport for industry-standard DCC applications (**Houdini Solaris, Maya, Blender USD viewport, Omniverse, and `usdview`**) powered by Dual AMD Radeon AI PRO R9700 GPUs.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                       Pathways OpenUSD Ingestion Architecture                     │
└───────────────────────────────────────────────────────────────────────────────────┘

         USD Asset Files (.usd, .usda, .usdc, .usdz) / DCC Live Stage
                                     │
                                     ▼
                    ┌─────────────────────────────────┐
                    │    Pixar OpenUSD C++ Core API   │
                    │   (Stage, Prims, Composition)   │
                    └────────────────┬────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 │                                       │
                 ▼                                       ▼
    ┌───────────────────────────┐           ┌───────────────────────────┐
    │   Direct Scene Loader     │           │   Hydra Render Delegate   │
    │      (`UsdLoader`)        │           │       (`hdPathways`)      │
    │  Standalone Pathways GUI  │           │   Houdini / usdview / DCC │
    └────────────┬──────────────┘           └────────────┬──────────────┘
                 │                                       │
                 └───────────────────┬───────────────────┘
                                     │
                                     ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │                     Pathways `SceneData` Core                     │
    │  - TriangleGPU Buffer (Vertices, Normals, Tangents, Material IDs) │
    │  - MaterialGPU Buffer (PBR, Dielectrics, Clearcoat, Emissive)     │
    │  - LightsBuffer (Analytical Area, Spot, Directional, Dome Lights) │
    │  - Texture Pools (Base Color, Normal, Metallic-Roughness, ORM)    │
    └────────────────────────────────┬──────────────────────────────────┘
                                     │
                                     ▼
    ┌───────────────────────────────────────────────────────────────────┐
    │     Vulkan 1.4 Hardware Ray Tracing & Native Wavefront Path Tracer│
    │  - VK_KHR_acceleration_structure (BLAS / TLAS Multi-Instance)     │
    │  - High-Throughput Wavefront Path Tracing (Stochastic MIS NEE)    │
    │  - Dual GPU Pipelined Execution (RADV GFX1201 / RDNA 4 Wave32)    │
    └───────────────────────────────────────────────────────────────────┘
```

---

## 2. Technical Comparison: glTF 2.0 vs. OpenUSD

| Capability | glTF 2.0 (`cgltf`) | Universal Scene Description (`pxr`) | Pathways Engineering Impact |
| :--- | :--- | :--- | :--- |
| **Data Scope** | Single asset / flat scene delivery | Composable multi-department scene graphs | Ingest entire film-scale environments |
| **Composition Arcs** | None (monolithic file) | SubLayers, References, Payloads, Variants | Non-destructive edits, multi-LOD variant toggles |
| **Storage Formats** | JSON (`.gltf`), Binary pack (`.glb`) | Text (`.usda`), Zero-Copy Binary (`.usdc`), ZIP (`.usdz`) | Memory-mapped zero-copy parsing of massive datasets |
| **Instancing Model** | Flat node transform hierarchy | Native Point Instancing (`UsdGeomPointInstancer`) | Direct 1:1 mapping to Vulkan TLAS instances |
| **Lighting Standard** | KHR_lights_punctual (basic) | `UsdLux` (Rect, Disk, Sphere, Cylinder, Dome, Distant) | Full physical area lights for stochastic MIS NEE |
| **Material Standard** | Metallic-Roughness PBR | `UsdPreviewSurface` + MaterialX (`usdMtlx`) | Native clearcoat, IOR, transmission, and sheen |
| **DCC Ecosystem** | Export-only interchange | Live two-way interchange via Hydra | Interactive live viewport in Houdini/Maya |

---

## 3. Data Model Mapping to Pathways Engine

Pathways models scenes internally via `SceneData` (`src/scene/ProceduralScene.hpp`). Every USD primitive resolves cleanly into Pathways' GPU memory buffers:

### 3.1 Geometry (`UsdGeomMesh` $\to$ `TriangleGPU`)
- **Topological Traversal:**
  USD meshes store topology via `faceVertexCounts` (integers indicating polygon face degree) and `faceVertexIndices`.
  - USD polygons can be arbitrary n-gons or quads. Pathways requires triangulated primitives (`TriangleGPU`: 160 bytes).
  - Triangulation pipeline:
    - Convex faces: Fast fan triangulation $(v_0, v_i, v_{i+1})$.
    - Concave n-gons: Ear-clipping triangulation.
- **Attributes & Primvars:**
  - `points` $\to$ `Vertex.position.xyz`.
  - `normals` (face-varying or vertex-varying) $\to$ `Vertex.normal.xyz`.
  - `primvars:st` or `primvars:uv` $\to$ `Vertex.position.w` ($u$) and `Vertex.normal.w` ($v$).
  - Tangents $\to$ Generated via MikkTSpace algorithm; stored in `Vertex.tangent` ($xyz$ + handness sign $w$).
- **Coordinates & Spatial Scale:**
  - Up-Axis: Checked via `UsdGeomGetStageUpAxis(stage)`. If `UsdGeomTokens->z`, apply a $90^\circ$ basis transformation $\begin{pmatrix} 1 & 0 & 0 \\ 0 & 0 & 1 \\ 0 & -1 & 0 \end{pmatrix}$.
  - Real-World Units: Checked via `UsdGeomGetStageMetersPerUnit(stage)`. Pathways renders in SI meters; all vertices and bounds are scaled by `metersPerUnit`.

### 3.2 Native Instancing (`UsdGeomPointInstancer` $\to$ Vulkan TLAS)
Large USD scenes (such as forests, cities, or particle debris) rely heavily on `UsdGeomPointInstancer`:
- Instead of duplicating geometry in the BLAS, `UsdGeomPointInstancer` provides per-instance `positions`, `orientations` (quaternions), and `scales`.
- Pathways maps each prototype mesh to a single `VkAccelerationStructureKHR` (BLAS), and instantiates each point as a `VkAccelerationStructureInstanceKHR` in the TLAS:
  ```cpp
  VkAccelerationStructureInstanceKHR instance{};
  instance.transform = glmMatrixToVkTransformMatrixKHR(worldTransform);
  instance.accelerationStructureReference = prototypeBLAS.deviceAddress;
  instance.mask = 0xFF;
  instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
  ```
  This reduces VRAM footprint by multiple gigabytes and enables rendering billions of instanced triangles at 120+ FPS.

### 3.3 Materials (`UsdShade` & `UsdPreviewSurface` $\to$ `MaterialGPU`)
The `UsdPreviewSurface` standard maps 1:1 to Pathways' physical GGX shader model:

| `UsdPreviewSurface` Attribute | Type | Pathways `MaterialGPU` Field | Pathways Shader Mapping |
| :--- | :--- | :--- | :--- |
| `diffuseColor` | `color3f` / texture | `baseColor` (`vec4`) | Diffuse Lambertian albedo |
| `metallic` | `float` / texture | `metallic` (`float`) | Conductor Fresnel blend ($F_0$) |
| `roughness` | `float` / texture | `roughness` (`float`) | $\alpha = \text{roughness}^2$ for GGX |
| `ior` | `float` | `ior` (`float`) | Dielectric Fresnel Snell's Law |
| `opacity` / `opacityThreshold` | `float` | `transmission` / cutout | Alpha blend or dielectric refraction |
| `clearcoat` | `float` / texture | `clearcoat` (`float`) | Secondary specular lobe |
| `clearcoatRoughness` | `float` | `clearcoatRoughness` (`float`) | Clearcoat GGX distribution |
| `emissiveColor` | `color3f` / texture | `emission` (`vec4`) | Unshadowed analytical emission |
| `normal` | `normal3f` / texture | Normal map sampler | Tangent-space perturbed shading normal |

### 3.4 Lights (`UsdLux` $\to$ `LightGPU` & MIS NEE)
USD features first-class physical lighting primitives in the `UsdLux` schema:
- **`UsdLuxRectLight`:** Mapped to rectangular Area Light (`LightGPU` with `u` and `v` edge vectors).
- **`UsdLuxDiskLight`:** Mapped to elliptical/disk Area Light.
- **`UsdLuxSphereLight`:** Mapped to spherical Area Light.
- **`UsdLuxDistantLight`:** Mapped to directional sun light.
- **`UsdLuxDomeLight`:** Mapped to Pathways' 32-bit floating-point environment map sky dome.
- **Synergy with Next-Event Estimation:**
  Production USD scenes contain dozens or hundreds of `UsdLux` lights. Pathways' Wavefront path tracer ingests these lights into `LightsBuffer`, executing stochastic Next-Event Estimation with Multiple Importance Sampling (MIS).

---

## 4. Architectural Implementation Options

### Option A: Official Pixar OpenUSD C++ Library (`pxr`) — *Recommended*
- **Description:** Link directly against Pixar's official OpenUSD monolithic shared library (`/usr/lib64/libusd_ms.so`), already installed on the host system under Fedora Linux 44.
- **System Package Availability:**
  - `usd-libs-26.03-4.fc44.x86_64` (Already installed).
  - `usd-devel-26.03-4.fc44.x86_64` (Available via Fedora repositories; provides `/usr/include/pxr/` and `/usr/lib64/cmake/pxr/pxrConfig.cmake`).
- **Strengths:**
  - 100% complete specification compliance: full composition engine (PCP), binary Crate parsing, USDA, USDZ archive extraction, variant sets, and animation time-samples.
  - Native foundation for the Hydra Render Delegate (`hdPathways`).
  - Native integration with MaterialX (`usdMtlx`).
- **CMake Integration:**
  ```cmake
  find_package(pxr REQUIRED)
  target_link_libraries(pathways PRIVATE pxr::usd pxr::usdGeom pxr::usdShade pxr::usdLux)
  ```

### Option B: Standalone Header-Only Ingestion (`tinyusdz`)
- **Description:** Embed `tinyusdz` (dependency-free C++14/17 library by Syoyo Fujita) as a third-party submodule.
- **Strengths:** Zero system dependencies, lightweight binary footprint (~3 MB), builds on any platform without package managers.
- **Limitations:** Limited support for complex composition arcs, no Hydra delegate support, experimental MaterialX.
- **Verdict:** Excellent fallback for minimal portable builds, but insufficient for studio pipeline integration.

---

## 5. Hydra Render Delegate (`hdPathways`) Architecture

To integrate Pathways directly into DCC applications (Houdini Solaris, Maya, Blender, `usdview`), Pathways can implement a **Hydra Render Delegate**:

```
┌───────────────────────────────────────────────────────────────────┐
│                    DCC Application (Houdini Solaris / Maya)       │
├───────────────────────────────────────────────────────────────────┤
│                     Hydra Scene Index / Render Index              │
└─────────────────────────────────┬─────────────────────────────────┘
                                  │ Calls delegate primitives
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│                        hdPathways Delegate                        │
├────────────────────────┬──────────────────────────┬───────────────┤
│    HdPathwaysMesh      │    HdPathwaysMaterial    │ HdPathwaysLight│
│  Ingests points, tris  │  Compiles PreviewSurface │ Ingests UsdLux│
│  Builds Vulkan BLAS    │  Updates MaterialGPU     │ Updates Lights│
└────────────────────────┴──────────────────────────┴───────────────┘
                                  │
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│                     HdPathwaysRenderPass                          │
│   - Triggers Vulkan 1.4 hardware ray tracing (vkCmdTraceRaysKHR)  │
│   - Executes Wavefront path tracing with stochastic MIS NEE       │
│   - Blits result into OpenGL/Vulkan external memory viewport      │
└───────────────────────────────────────────────────────────────────┘
```

### Core Hydra Classes to Implement:
1. **`HdPathwaysRenderDelegate`:** Factory class reporting supported prim types, token formats, and AOV descriptor capabilities.
2. **`HdPathwaysRenderPass`:** Executes the actual ray tracing render loop per viewport refresh, reading camera transforms and submitting command buffers.
3. **`HdPathwaysMesh` (Rprim):** Receives mesh topology and vertex buffers, managing dynamic BLAS rebuilds / refits on deformation.
4. **`HdPathwaysMaterial` (Sprim):** Parses material networks, texture coordinates, and compiles parameters into Pathways' UBO/SSBO slots.
5. **`HdPathwaysLight` (Sprim):** Tracks interactive light adjustments (intensity, exposure, position, radius).

---

## 6. Phased Implementation Roadmap

### Phase 1: Native USD Scene Reader (`UsdLoader`)
- **Target:** Load static `.usd`, `.usda`, `.usdc`, and `.usdz` assets directly via CLI (`--scene assets/scene.usda`) and ImGui scene picker.
- **Deliverables:**
  1. `CMakeLists.txt`: Add optional `PATHWAYS_ENABLE_USD` flag with `find_package(pxr)`.
  2. `src/scene/UsdLoader.hpp` & `UsdLoader.cpp`:
     - Stage initialization and boundary resolution.
     - Mesh extraction, ear-clipping triangulation, and MikkTSpace tangent calculation.
     - `UsdPreviewSurface` material translation and texture loading (including memory-based `.usdz` decompression).
     - `UsdLux` area light extraction into `LightGPU`.
     - `UsdGeomCamera` extraction.
  3. `src/scene/SceneRegistry.cpp`:
     - Extend directory scanner to index `.usd`, `.usda`, `.usdc`, `.usdz` files alongside `.gltf`/`.glb`.
     - Read USD stage metadata (stage bounds, prim count) for GUI previews.

### Phase 2: Animation & Instancing
- **Target:** Support USD animated time-samples and large-scale point instancing.
- **Deliverables:**
  1. Interactive timeline scrubber in Dear ImGui (`UsdTimeCode`).
  2. Point instancer translation to Vulkan 1.4 TLAS instances (`VkAccelerationStructureInstanceKHR`).
  3. Motion vectors for dynamic geometry in temporal reprojection and denoising.

### Phase 3: Hydra Render Delegate (`hdPathways`)
- **Target:** Standalone shared library `libhdPathways.so` discoverable via `PXR_PLUGINPATH_NAME`.
- **Deliverables:**
  1. Full implementation of `HdRenderDelegate`, `HdRenderPass`, and prim proxies.
  2. Interop with host application viewports via `VK_KHR_external_memory`.
  3. Interactive validation in Pixar's official `usdview`.

---

## 7. Conclusion

Adding OpenUSD support aligns Pathways with contemporary professional graphics pipelines. By leveraging the existing `usd-libs` infrastructure on Fedora Linux 44 and mapping USD prims directly into Pathways' pure Vulkan 1.4 `SceneData` model, Pathways will provide real-time hardware ray tracing, stochastic MIS NEE, and dual-GPU acceleration for cinematic production assets.
