# TEST_INFRA: Comprehensive Multi-Tier Testing Infrastructure & Specification
# Pathways Vulkan 1.4 Path Tracer — Scanlands & High-Density Point Instancing (F1–F13)

## 1. Executive Summary & Testing Philosophy

This document defines the complete testing methodology, interface contracts, mathematical invariants, and test specifications for the Pathways Vulkan 1.4 path tracer enhancements (Features F1 through F13).

The testing infrastructure adheres to a **4-Tier Progressive Verification Hierarchy**:
- **Tier 1: Feature Coverage (Unit & Functional)** — At least 5 explicit test cases per feature covering primary happy paths, algorithm correctness, and interface contracts ($13 \times 5 = 65$ tests minimum).
- **Tier 2: Boundary & Corner Cases** — At least 5 explicit edge-case and boundary tests per feature ($13 \times 5 = 65$ tests minimum), testing singular values, extreme parameters, zero divisions, degenerate geometry, and resource limits.
- **Tier 3: Cross-Feature Combinations** — Pairwise and multi-feature interaction tests verifying seamless cooperation between shaders, memory structures, USD ingestion, culling, and single-GPU performance.
- **Tier 4: Real-World Application Scenarios** — End-to-end operational stress tests using `scenes/Scanlands.blend` / USD, verifying high-density point instancing (187,490 instances), single-GPU VRAM constraints (< 24.0 GB on AMD Radeon AI PRO R9700 `gfx1201`), physical fidelity, and headless regression benchmarking.

---

## 2. Feature Inventory & Interface Contracts

| ID | Feature Name | Target Component | Architectural Role | Milestone |
|---|---|---|---|---|
| **F1** | Dielectric Normal Mapping Parity | `wavefront_shade_dielectric.comp` | Tangent-space normal mapping for dielectric refraction/reflection parity with RTP | M1 |
| **F2** | Thin-Walled Diffuse Transmission | `wavefront_shade_diffuse.comp`, `complex.comp` | Two-sided diffuse transmission BSDF and shadow NEE for translucent leaves/canopies | M1 |
| **F3** | MaterialGPU Transmission Fields | `src/scene/Material.hpp`, GLSL common | `diffuseTransmission` factor/texture in `MaterialGPU` and std430 alignment | M1 |
| **F4** | GLSL Procedural Noise Library | `shaders/compute/procedural_noise.glsl` | Analytical gradient Simplex 3D, Voronoi 2D, and domain-rotated fBm primitives | M2 |
| **F5** | Procedural Terrain Shading | `wavefront_shade_diffuse.comp` | GPU procedural rock/soil elevation and slope evaluation eliminating 2D baked maps | M2 |
| **F6** | Procedural Water Waves & Beer-Lambert | `wavefront_shade_dielectric.comp` | Gerstner wave normal superposition coupled with Beer-Lambert depth absorption | M2 |
| **F7** | Procedural Material Flags | `MaterialGPU::type`, GLSL common | Bitmask encoding of procedural modes (`TERRAIN = 1<<9`, `WATER = 1<<11`) | M2 |
| **F8** | Scanlands USD Conversion Pipeline | `scripts/convert_scanlands_to_usd.py` | Two-stage hybrid asset extraction of 8 BLAS prototypes and 187k point instances | M3 |
| **F9** | UsdGeomPointInstancer Loader Support | `src/scene/UsdLoader.cpp` | Ingest OpenUSD `UsdGeomPointInstancer` prims into hardware TLAS instances | M3 |
| **F10** | Foliage Alpha Mask Parity | `UsdLoader.cpp`, wavefront shaders | `ALPHA_MODE_MASK` handling for foliage cards with cutouts | M3 |
| **F11** | Configurable Density & Culling | `Config.hpp`, `Engine.cpp`, `UsdLoader.cpp` | Deterministic hash downsampling and radial distance culling CLI options | M3 |
| **F12** | Single-GPU Benchmarking & Profiling | `Engine.cpp`, headless runner | AMD R9700 single-GPU telemetry, VRAM < 24 GB, 1080p/1440p scalability | M4 |
| **F13** | Physical Fidelity Verification | Visual regression suite | Photometric validation of backlit foliage, caustics, and terrain against Cycles | M4 |

### Interface Contract Specifications

#### Contract C1: Material Definition ↔ Wavefront Microkernels (F1, F2, F3, F7)
- **Host Layout (`src/scene/Material.hpp`)**: `MaterialGPU` must be 16-byte aligned (std430 standard). Total size is 208 bytes (or packed 192 bytes with reserved fields).
  - Offset 192: `float diffuseTransmission` (transmission factor $[0.0, 1.0]$).
  - Offset 196: `uint32_t diffuseTransmissionTex` (bindless texture index $1 \le \text{idx} \le 512$, $0 = \text{none}$).
  - Offset 48: `uint32_t type` bitmask:
    - Bits 0–7: Base `MaterialType` (`0=DIFFUSE`, `1=METALLIC`, `2=DIELECTRIC`, `3=EMISSIVE`).
    - Bit 9 (`1u << 9` = `0x0200`): `MATERIAL_FLAG_PROCEDURAL_TERRAIN`.
    - Bit 11 (`1u << 11` = `0x0800`): `MATERIAL_FLAG_PROCEDURAL_WATER`.
- **GLSL Layout (`shaders/compute/wavefront_common.glsl`)**: `struct Material` fields match exact host byte offsets and sizes. Base type extracted via `(mat.type & 0xFFu)`. Procedural tests evaluate via `((mat.type & MATERIAL_FLAG_*) != 0u)`.

#### Contract C2: OpenUSD Ingestion ↔ Engine TLAS Instancing (F8, F9, F10, F11)
- `UsdGeomPointInstancer` prims extract instance arrays: `positions` (`GfVec3f`), `orientations` (`GfQuath`), `scales` (`GfVec3f`), and `protoIndices` (`int`).
- Instancer prototype paths map to unique deduplicated BLAS indices (`protoPathToBlas`).
- Transform matrices calculated as $M = M_{\text{stage}} \cdot M_{\text{instancer}} \cdot M_{\text{instance}}$, mapped to `VkAccelerationStructureInstanceKHR`.
- Density downsampling uses integer hash function:
  $$\text{hash}(i) = ((i \times 0x45d9f3b) \text{ \textasciicircum\ } (i \gg 16)) \times 0x45d9f3b$$
  Instance included if $\text{hash}(i) < \text{density} \times 0\text{xFFFFFFFF}$.
- Distance culling computes radial Euclidean distance $d = \|\vec{P}_{\text{instance}} - \vec{P}_{\text{camera}}\|$. Instance culled if $\text{cullDistance} > 0.0$ and $d > \text{cullDistance}$.

#### Contract C3: Procedural Noise Evaluation (F4, F5, F6)
- Simplex noise `simplex3D_grad(vec3 p)` returns `vec4(grad.xyz, value)` with value bounded in $[-1.0, 1.0]$.
- Analytical gradient $\nabla N$ satisfies $C^1$ continuity across simplex tetrahedral boundaries.
- Gerstner wave displacement $\vec{P}(x,z,t)$ and surface normal $\vec{N}$ enforce steepness constraint $\sum Q_i k_i A_i \le 1.0$.
- Water Beer-Lambert depth absorption calculates spectral transmission:
  $$T(d) = \exp(-\vec{\sigma}_a \cdot d)$$

---

## 3. Tier 1: Feature Coverage Test Matrix (Unit & Functional)

Each feature has at least 5 dedicated, isolated test cases covering all primary functional requirements.

### Feature F1: Dielectric Normal Mapping Parity
- **F1-T01 (Tangent Decoding)**: Verifies octahedral unpacking of vertex tangent vector from `hitData1.x` and tangent sign from `hitData1.y`.
- **F1-T02 (TBN Orthonormalization)**: Verifies Gram-Schmidt orthonormalization: $\vec{T}' = \text{normalize}(\vec{T} - (\vec{T} \cdot \vec{N})\vec{N})$ and $\vec{B} = (\vec{N} \times \vec{T}') \cdot \text{sign}$.
- **F1-T03 (Normal Scale Modulation)**: Verifies `mat.normalScale` correctly scales tangent and bitangent perturbing components $(x, y)$ while maintaining $z \ge 0$.
- **F1-T04 (Specular Reflection Perturbation)**: Verifies perturbed shading normal deflects reflected ray direction $\vec{R} = \text{reflect}(\vec{D}, \vec{N}_{\text{perturbed}})$.
- **F1-T05 (Snell Refraction Perturbation)**: Verifies perturbed shading normal evaluates Snell's law refraction ray direction $\vec{T} = \text{refract}(\vec{D}, \vec{N}_{\text{perturbed}}, \eta)$ with total internal reflection handling.

### Feature F2: Thin-Walled Diffuse Transmission
- **F2-T01 (BSDF Energy Partition)**: Verifies diffuse reflection lobe $(1 - T_d) \frac{\rho_d}{\pi}$ and transmission lobe $T_d \frac{\rho_d}{\pi}$ split according to `diffuseTransmission`.
- **F2-T02 (Energy Conservation Invariant)**: Verifies that hemisphere integral of reflection + transmission satisfies $\int (f_r + f_t) |\cos\theta| d\omega \le \rho_d \le 1.0$.
- **F2-T03 (Backlit Forward Scatter)**: Verifies that light striking the backside of a thin leaf card scatters forward into the viewing camera direction with non-zero luminance.
- **F2-T04 (Direct Shadow NEE Penetration)**: Verifies Next Event Estimation (NEE) shadow ray throughput correctly multiplies by $T_d \cdot \vec{\rho}_d$ through a single thin-walled card.
- **F2-T05 (Complex Microkernel Layering)**: Verifies `wavefront_shade_complex.comp` evaluates clearcoat reflection on top of the thin-walled diffuse transmission substrate.

### Feature F3: MaterialGPU Transmission Fields
- **F3-T01 (Host Struct Layout & Alignment)**: Verifies `sizeof(MaterialGPU) % 16 == 0` and total struct size matches std430 alignment specification.
- **F3-T02 (Transmission Field Offsets)**: Verifies `offsetof(MaterialGPU, diffuseTransmission)` is at offset 192 and `offsetof(MaterialGPU, diffuseTransmissionTex)` is at offset 196.
- **F3-T03 (GLSL Struct Field Parity)**: Verifies `shaders/compute/wavefront_common.glsl` declares identical field ordering and matching 4-byte scalar alignment.
- **F3-T04 (Default Value Zeroing)**: Verifies constructor default initializes `diffuseTransmission = 0.0f` and `diffuseTransmissionTex = 0u`.
- **F3-T05 (Texture Index Bounds)**: Verifies texture index validation logic permits indices in range $[1, 512]$ and flags $0$ as unbound.

### Feature F4: GLSL Procedural Noise Library
- **F4-T01 (Simplex 3D Analytical Gradient)**: Verifies `simplex3D_grad(p)` analytical gradient closely matches numerical central difference: $\frac{\partial N}{\partial x} \approx \frac{N(p+\epsilon \hat{x}) - N(p-\epsilon \hat{x})}{2\epsilon}$.
- **F4-T02 (Simplex Value Boundedness)**: Verifies scalar noise output strictly stays within $[-1.0, 1.0]$ across $10^5$ random spatial coordinates.
- **F4-T03 (Hash33 Uniform Distribution)**: Verifies `hash33(p)` pseudo-random output exhibits uniform spatial distribution and zero periodicity over large domains.
- **F4-T04 (Voronoi 2D F1/F2 Metric)**: Verifies Voronoi cell computation satisfies $F_1 \le F_2$ universally and produces cellular distance boundaries.
- **F4-T05 (fBm Octave Harmonic Scaling)**: Verifies fractional Brownian motion accumulates octaves with geometric frequency lacunarity ($2.0$) and amplitude gain ($0.5$).

### Feature F5: Procedural Terrain Shading
- **F5-T01 (Elevation Band Interpolation)**: Verifies world $Y$ coordinate smoothly interpolates between beach sand, valley soil, mountain rock, and alpine snow bands.
- **F5-T02 (Slope Angle Rock Strata)**: Verifies surface normal dot product $\vec{N} \cdot (0, 1, 0) < \cos(45^\circ)$ shifts diffuse albedo from grass/soil to vertical rock strata.
- **F5-T03 (Analytical Gradient Normal Perturbation)**: Verifies micro-rock facet normal perturbation modifies shading normal without external 2D normal maps.
- **F5-T04 (Triplanar Blending Weight Conservation)**: Verifies projection weights $w_x, w_y, w_z$ along cardinal axes satisfy $w_x + w_y + w_z = 1.0$.
- **F5-T05 (Elevation Roughness Modulation)**: Verifies roughness increases dynamically with rock exposure and decreases on wet soil/sand.

### Feature F6: Procedural Water Waves & Beer-Lambert
- **F6-T01 (Gerstner Normal Superposition)**: Verifies multi-directional Gerstner wave summation calculates accurate normal vectors perpendicular to the wave surface.
- **F6-T02 (Gerstner Crest Steepness Bound)**: Verifies wave steepness parameter $Q$ clamps to $Q_i \le \frac{1}{\omega_i A_i N_{\text{waves}}}$ to prevent self-intersecting loops.
- **F6-T03 (Beer-Lambert Exponential Attenuation)**: Verifies light absorption follows $I(d) = I_0 \exp(-\vec{\sigma}_a d)$ across RGB attenuation coefficients.
- **F6-T04 (Water Fresnel Reflection Coupling)**: Verifies Schlick Fresnel evaluation with dynamic wave normal produces realistic angular reflection probability.
- **F6-T05 (Depth Color Shift)**: Verifies spectral transmission shifts from turquoise in shallow waters ($d < 1\text{m}$) to deep navy blue ($d > 10\text{m}$).

### Feature F7: Procedural Material Flags
- **F7-T01 (Base Material Type Preservation)**: Verifies extracting base type via `type & 0xFFu` yields exact unmodified `MaterialType` enum.
- **F7-T02 (Terrain Flag Bitmask Encoding)**: Verifies bitwise operations `type |= (1u << 9)` and `(type & (1u << 9)) != 0` operate orthogonally.
- **F7-T03 (Water Flag Bitmask Encoding)**: Verifies bitwise operations `type |= (1u << 11)` and `(type & (1u << 11)) != 0` operate orthogonally.
- **F7-T04 (Compound Procedural Flagging)**: Verifies setting multiple flags simultaneously does not corrupt individual flag checks.
- **F7-T05 (GLSL Shader Macro Parity)**: Verifies shader compilation succeeds with `#define MATERIAL_FLAG_PROCEDURAL_TERRAIN (1u << 9)`.

### Feature F8: Scanlands USD Conversion Pipeline
- **F8-T01 (Script CLI Invocation)**: Verifies `convert_scanlands_to_usd.py` accepts source `.blend` and destination paths and parses options cleanly.
- **F8-T02 (Prototype Deduplication)**: Verifies that 40 particle scatter systems deduplicate to exactly 8 canonical prototype BLAS meshes.
- **F8-T03 (UsdGeomPointInstancer Prim Authoring)**: Verifies output USD stage contains authored `UsdGeomPointInstancer` prims with prototype relationships.
- **F8-T04 (Quaternion Normalization)**: Verifies all authored instance rotation quaternions satisfy $\|q\| = 1.0 \pm 10^{-4}$.
- **F8-T05 (Triangle Budget Guardrail)**: Verifies total deduplicated prototype triangles stay within the $\sim 8.4$M triangle budget.

### Feature F9: UsdGeomPointInstancer Loader Support
- **F9-T01 (Instancer Prim Traversal)**: Verifies `UsdLoader` traverses and identifies all `UsdGeomPointInstancer` prims in the USD stage.
- **F9-T02 (Prototype BLAS Association)**: Verifies prototype targets resolve to valid corresponding BLAS entries in `data.blasRanges`.
- **F9-T03 (Instance Transform Calculation)**: Verifies `ComputeInstanceTransformsAtTime` computes composite transformation matrices correctly.
- **F9-T04 (TLAS Instance Population)**: Verifies `SceneData::instances` appends valid `SceneInstance` entries referencing prototype BLAS indices.
- **F9-T05 (Instanced Scene Bounding Box)**: Verifies overall scene bounds encompass all transformed instance extents.

### Feature F10: Foliage Alpha Mask Parity
- **F10-T01 (Opacity Threshold Extraction)**: Verifies USD material `opacityThreshold` or `cutoutOpacity` translates to `mat.alphaCutoff`.
- **F10-T02 (Alpha Mode Mask Setting)**: Verifies `mat.alphaMode` is set to `ALPHA_MODE_MASK` (1) when opacity threshold is specified.
- **F10-T03 (Archetype Alpha Mask Classification)**: Verifies `getMaterialArchetype` returns `MATERIAL_ARCHETYPE_ALPHAMASK` (5) for masked materials.
- **F10-T04 (Wavefront Alpha Cutout Passthrough)**: Verifies rays hitting texels with alpha below cutoff pass through without ray termination.
- **F10-T05 (Two-Sided Card Normal Handling)**: Verifies normal orientation on foliage cards evaluates correctly for both front and back faces.

### Feature F11: Configurable Density & Culling
- **F11-T01 (CLI Argument Parsing)**: Verifies `--instance-density <float>` and `--cull-distance <float>` CLI flags parse into `Config`.
- **F11-T02 (Deterministic Hash Invariance)**: Verifies integer hash function is deterministic and uniform across instance index space.
- **F11-T03 (Instance Count Density Scaling)**: Verifies active instance count scales linearly with density: $N_{\text{active}} \approx \text{density} \times N_{\text{total}} \pm 2\%$.
- **F11-T04 (Radial Camera Distance Culling)**: Verifies instances beyond `cullDistance` from the active camera are excluded from TLAS build.
- **F11-T05 (Dynamic Camera Position Updates)**: Verifies moving camera updates active distance-culled instance set without memory leaks.

### Feature F12: Single-GPU Benchmarking & Profiling
- **F12-T01 (Single-GPU VRAM Guardrail)**: Verifies peak VRAM usage measured via `/opt/rocm/core-10.0/bin/amd-smi` remains strictly $< 24.0$ GB.
- **F12-T02 (Headless Benchmark Telemetry)**: Verifies `--dump-stats` writes valid JSON containing FPS, frame time (ms), and ray throughput.
- **F12-T03 (Multi-Bounce Timing Progression)**: Measures frame times across 1, 2, and 3 bounces verifying predictable linear/sublinear scaling.
- **F12-T04 (1080p Real-Time Stability)**: Verifies sustained headless rendering at $1920 \times 1080$ without GPU device lost or hung command buffers.
- **F12-T05 (1440p High-Resolution Stability)**: Verifies sustained headless rendering at $2560 \times 1440$ within single-GPU VRAM limits.

### Feature F13: Physical Fidelity Verification
- **F13-T01 (Headless Frame Dump Output)**: Verifies `--dump-frame` outputs valid non-empty PNG and `--dump-hdr` outputs valid OpenEXR files.
- **F13-T02 (Pixel Value Numerical Validity)**: Verifies rendered frame buffers contain zero NaN, Inf, or subnormal values across all channels.
- **F13-T03 (Dynamic Range & Mean Luminance)**: Verifies mean scene luminance lies in physical range $[0.05, 0.85]$ and blown pixels $< 15\%$.
- **F13-T04 (Backlit Foliage Photometric Glow)**: Verifies leaf cards in direct sun shadow exhibit forward diffuse transmission luminance $> 0.05$.
- **F13-T05 (Progressive Stationary Convergence)**: Verifies stationary multi-frame accumulation strictly reduces pixel variance $\propto 1/\sqrt{N}$.

---

## 4. Tier 2: Boundary & Corner Cases Test Matrix

Every feature has at least 5 adversarial edge cases, boundary conditions, and stress tests ($13 \times 5 = 65$ tests).

| ID | Target Feature | Boundary / Corner Case Condition | Expected System Behavior |
|---|---|---|---|
| **F1-B01** | Dielectric Normal | Flat normal map `(128, 128, 255)` | Produces exact unperturbed geometric normal $(0, 0, 1)$ |
| **F1-B02** | Dielectric Normal | Degenerate tangent vector $(0, 0, 0)$ | Fallbacks to robust synthetic tangent basis without NaN |
| **F1-B03** | Dielectric Normal | Grazing incidence $(\theta = 89.9^\circ)$ with inward normal | Evaluates TIR cleanly without negative square root |
| **F1-B04** | Dielectric Normal | `normalScale = 0.0` or negative scale | $0.0$ returns geometric normal; negative inverts tangent space smoothly |
| **F1-B05** | Dielectric Normal | TanSign not equal to $\pm 1.0$ | Clamped to $\text{sign}(s)$ preventing coordinate inversion |
| **F2-B01** | Thin Transmission | `diffuseTransmission = 0.0` | Exact bitwise match with pure Lambertian diffuse |
| **F2-B02** | Thin Transmission | `diffuseTransmission = 1.0` | 100% forward transmission, exactly 0.0 reflection |
| **F2-B03** | Thin Transmission | Pure black albedo `(0, 0, 0)` | Exactly zero radiance transmitted or reflected |
| **F2-B04** | Thin Transmission | Light ray parallel to leaf card $(\theta = 90^\circ)$ | Cosine factor goes to zero; no NaN or light leaking |
| **F2-B05** | Thin Transmission | Reversed face normal (back-facing geometric triangle) | Two-sided BSDF evaluates symmetrically without black culling |
| **F3-B01** | MaterialGPU Layout | `diffuseTransmission = -0.5` or `1.5` | Values clamped to $[0.0, 1.0]$ upon ingestion |
| **F3-B02** | MaterialGPU Layout | Texture index $> 512$ (e.g. `UINT32_MAX`) | Clamped or flagged as invalid; prevents bindless out-of-bounds |
| **F3-B03** | MaterialGPU Layout | Sub-byte alignment padding check | No hidden compiler padding between `sheenTex` and transmission fields |
| **F3-B04** | MaterialGPU Layout | Memory zeroing (`memset` to 0) | Produces valid default opaque diffuse material without crashing |
| **F3-B05** | MaterialGPU Layout | NaN / Inf in floating point fields | Replaced by safe defaults ($0.0$ or $1.0$) upon load |
| **F4-B01** | Procedural Noise | Evaluation at origin $(0, 0, 0)$ | Evaluates cleanly with zero division guard on gradient normalization |
| **F4-B02** | Procedural Noise | Extremely large coordinates $(> 10^6)$ | Float32 domain wrapping prevents precision banding or overflow |
| **F4-B03** | Procedural Noise | Negative octants $(-x, -y, -z)$ | Continuous derivatives across axis crossing; no seam at $x=0$ |
| **F4-B04** | Procedural Noise | Zero octaves requested in fBm | Returns constant $0.0$ without entering infinite loop |
| **F4-B05** | Procedural Noise | Zero gradient magnitude $(\|\nabla N\| = 0)$ | Normalization safely defaults to $(0, 0, 1)$ without NaN |
| **F5-B01** | Terrain Shading | Downward-facing overhang $(\vec{N} \cdot \vec{Y} < 0)$ | Treated as deep cave rock strata without inverted normals |
| **F5-B02** | Terrain Shading | Negative elevation $(Y < 0\text{m}$, underwater trench) | Evaluates deep underwater bedrock without underflow |
| **F5-B03** | Terrain Shading | Vertical cliff face $(\vec{N} \cdot \vec{Y} = 0.0)$ | Perfect $90^\circ$ slope evaluates rock without division by zero |
| **F5-B04** | Terrain Shading | Mountain peak elevation $(Y > 10000\text{m})$ | Caps smoothly at permanent alpine snow; no color overflow |
| **F5-B05** | Terrain Shading | Flat horizontal plain $(\vec{N} = (0, 1, 0))$ | Pure soil/vegetation with zero rock contamination |
| **F6-B01** | Water Waves | Zero water depth $(d = 0.0\text{m})$ | Beer-Lambert transmission $T(0) = 1.0$ (no absorption) |
| **F6-B02** | Water Waves | Infinite water depth $(d \to \infty)$ | Beer-Lambert transmission $T \to 0.0$ without numerical underflow |
| **F6-B03** | Water Waves | Zero wave amplitude $(A_i = 0.0)$ | Evaluates perfectly flat mirror dielectric with normal $(0, 1, 0)$ |
| **F6-B04** | Water Waves | Water IOR boundary $(1.333 \to 1.0)$ | Evaluates boundary matching air without refraction discontinuity |
| **F6-B05** | Water Waves | Excessive steepness input $(\sum QkA > 1.0)$ | Normalized dynamically to prevent looping wave cusps |
| **F7-B01** | Procedural Flags | All upper bits set (`0xFFFFFF00`) | Base type in bits 0–7 remains intact and unaffected |
| **F7-B02** | Procedural Flags | Flag set on incompatible base type (e.g. emissive water) | Shader evaluates base type gracefully without branching crash |
| **F7-B03** | Procedural Flags | Undefined bits set (e.g. bit 15) | Upper undefined flags ignored cleanly without effect |
| **F7-B04** | Procedural Flags | Zeroed type field (`0u`) | Interpreted cleanly as `MATERIAL_DIFFUSE` with no procedural flags |
| **F7-B05** | Procedural Flags | GPU buffer roundtrip | 32-bit integer bitfield bit-exact across host and device |
| **F8-B01** | Scanlands USD | Non-existent `.blend` source path | Script reports clear error and exits with code 1 |
| **F8-B02** | Scanlands USD | Empty particle scatter system (0 instances) | Empty instancer prim omitted cleanly |
| **F8-B03** | Scanlands USD | Non-uniform scale matrix $(s_x \ne s_y \ne s_z)$ | Decomposed correctly into position, quaternion, and scale |
| **F8-B04** | Scanlands USD | Microscopic foliage scale ($s < 10^{-5}$) | Instanced without floating point degenerate bounds |
| **F8-B05** | Scanlands USD | High instance count stress (200k instances) | USD authoring completes within memory bounds |
| **F9-B01** | PointInstancer Loader | PointInstancer with empty `positions` array | Skipped with warning; zero instances added to TLAS |
| **F9-B02** | PointInstancer Loader | Prototype target path not found in stage | Warning logged; missing prototype skipped without crash |
| **F9-B03** | PointInstancer Loader | `protoIndices` contains index $\ge \text{numTargets}$ | Out-of-bounds instance skipped safely |
| **F9-B04** | PointInstancer Loader | Environment cap `PATHWAYS_USD_MAX_INSTANCES=10` | Caps ingestion at exactly 10 instances |
| **F9-B05** | PointInstancer Loader | Identity transform instancer | Transformed instances match base prototype positions |
| **F10-B01** | Foliage Alpha Mask | Alpha cutoff set to $0.0$ | Evaluates as fully opaque mesh |
| **F10-B02** | Foliage Alpha Mask | Alpha cutoff set to $1.0$ | Only texels with alpha $= 1.0$ register hits |
| **F10-B03** | Foliage Alpha Mask | Texture with missing alpha channel (RGB) | Alpha defaults to $1.0$; treated as opaque |
| **F10-B04** | Foliage Alpha Mask | Transparent texture with transmission | Evaluates alpha cutout first, then thin transmission |
| **F10-B05** | Foliage Alpha Mask | Extreme grazing ray through card edge | Ray step advances without getting trapped in endless alpha loop |
| **F11-B01** | Density & Culling | `--instance-density 0.0` | Instantiates exactly 0 point instances |
| **F11-B02** | Density & Culling | `--instance-density 1.0` | Instantiates 100% of authored point instances |
| **F11-B03** | Density & Culling | `--cull-distance 0.0` | Culls all instances outside camera position |
| **F11-B04** | Density & Culling | `--cull-distance -1.0` (negative distance) | Treated as disabled; all instances within density retained |
| **F11-B05** | Density & Culling | Negative density (`--instance-density -0.5`) | Clamped to $0.0$; no crash or underflow |
| **F12-B01** | Benchmarking | 1-frame benchmark (`--frames 1`) | Completes cleanly and outputs valid single-frame stats |
| **F12-B02** | Benchmarking | Warmup frames $\ge$ total frames | Warning logged; stats computed over available sample |
| **F12-B03** | Benchmarking | Memory leak stress over 500 frames | VRAM consumption at frame 500 equals frame 50 ($\pm 5$ MB) |
| **F12-B04** | Benchmarking | Headless run without display server (pure TTY) | Headless Vulkan swapchainless pipeline succeeds |
| **F12-B05** | Benchmarking | Explicit GPU selection (`--gpu-index 0`) | Targets primary AMD Radeon AI PRO R9700 cleanly |
| **F13-B01** | Physical Fidelity | Scene with zero light sources | Pure black render $(0, 0, 0)$ with zero noise or NaN |
| **F13-B02** | Physical Fidelity | Sunlight intensity $10^6$ nits | Tonemapped cleanly via ACES without color banding |
| **F13-B03** | Physical Fidelity | 100m deep water trench | Completely absorbs light; converges to dark water color |
| **F13-B04** | Physical Fidelity | Pure grazing reflection $(\theta = 89.99^\circ)$ | Fresnel reflectance approaches $1.0$ specular mirror |
| **F13-B05** | Physical Fidelity | Single-sample stochastic noise (1 SPP) | Validates unbiased isotropic noise without radial streaking |

---

## 5. Tier 3: Cross-Feature Interaction Combinations

Cross-feature tests verify the synthesis of multiple subsystems working simultaneously:

1. **XF-01 (F1 + F6: Dielectric Normal Mapping + Gerstner Waves)**:
   Verifies that when `MATERIAL_FLAG_PROCEDURAL_WATER` is active, Gerstner wave procedural normals correctly perturb dielectric reflection and refraction rays alongside Beer-Lambert absorption.
2. **XF-02 (F2 + F10: Thin-Walled Transmission + Foliage Alpha Cutouts)**:
   Verifies that foliage leaf cards evaluate alpha cutout transparency first, and surviving hits evaluate two-sided diffuse transmission and NEE shadow rays.
3. **XF-03 (F3 + F7: MaterialGPU Layout + Procedural Flag Bitmasks)**:
   Verifies that setting both `diffuseTransmission` and `MATERIAL_FLAG_PROCEDURAL_TERRAIN` in `MaterialGPU` preserves memory alignment and executes both features simultaneously.
4. **XF-04 (F9 + F11: USD Point Instancing + Density Downsampling & Distance Culling)**:
   Verifies that loading `PointInstancedMedCity.usd` or `Scanlands.usd` with `--instance-density 0.5` and `--cull-distance 150.0` correctly downsamples active TLAS instances without corrupting BLAS prototype bindings.
5. **XF-05 (F5 + F12: Procedural Terrain + Single-GPU Memory Guardrail)**:
   Verifies that procedural terrain shading evaluates high-frequency rock/soil surfaces on AMD R9700 while maintaining peak VRAM $< 24.0$ GB.
6. **XF-06 (F6 + F13: Procedural Water + Photometric Verification)**:
   Verifies that a rendered frame containing procedural water exhibits dynamic ripples in reflection and deep blue Beer-Lambert extinction in deep areas.
7. **XF-07 (F8 + F9: Scanlands Conversion Pipeline + OpenUSD Loader Ingestion)**:
   Verifies end-to-end dataflow from converted Scanlands USDA/USDC files directly into `UsdLoader::loadSceneData`.
8. **XF-08 (F2 + F13: Backlit Foliage Transmission + Headless Frame Dump)**:
   Verifies that headless rendering of a backlit tree canopy produces non-zero luminance in shadow regions, matching reference physical lighting.
9. **XF-09 (F4 + F5 + F6: Procedural Noise Library + Terrain + Water)**:
   Verifies that GLSL procedural noise functions can be invoked concurrently by both diffuse terrain shaders and dielectric water shaders within the same dispatch.
10. **XF-10 (F9 + F10 + F11 + F12: Full Scanlands Foliage Scalability)**:
    Verifies 187,490 foliage elements with alpha cutouts, dynamic distance culling, and single-GPU performance on the AMD Radeon AI PRO R9700.

---

## 6. Tier 4: Real-World Application Scenarios

1. **Scenario 1: Scanlands Scene Full Ingestion**:
   - Ingest all 8 BLAS prototypes and 187k foliage instances via OpenUSD.
   - Verify zero Vulkan validation layer errors.
   - Measure VRAM utilization via `amd-smi` verifying peak VRAM $< 24.0$ GB.
2. **Scenario 2: High-Sun Noon Lighting & Canopy Illumination**:
   - Sun at high elevation ($75^\circ$).
   - Verify leaf undersides are illuminated by forward transmission.
   - Confirm average canopy shadow luminance $> 0.05$.
3. **Scenario 3: Lakeside Coastal Landscape**:
   - Combined terrain cliffs and water body.
   - Gerstner waves animate surface reflections.
   - Submerged terrain visibly attenuates exponentially with depth.
4. **Scenario 4: Interactive Camera Traversal & Dynamic Culling**:
   - 120-frame camera motion traversal across landscape.
   - Dynamic instance culling maintains stable frame pacing without stutter.
5. **Scenario 5: Headless Multi-Bounce Benchmark**:
   - 1080p and 1440p sweeps at 1, 2, and 3 bounces.
   - JSON stats dumped and verified against performance targets.

---

## 7. Test Execution & Tooling Framework

### Test Automation Artifacts
- **Python E2E Test Suite**: `tests/e2e/test_scanlands_capabilities.py`
  - Automated runner executing Tier 1 through Tier 4 tests (145 tests total).
  - Compiles GLSL shaders directly using `glslc` (`--target-env=vulkan1.4`) to validate syntax, analytical noise functions, and SPIR-V code generation.
  - Queries hardware VRAM utilization directly via `/opt/rocm/core-10.0/bin/amd-smi metric --mem-usage`.
  - Verifies OpenUSD stages, prototype hierarchies, and `UsdGeomPointInstancer` prims via `pxr` Python API.
  - Reads authentic engine telemetry from `output/scanlands_benchmark_stats.json`.
  - Analyzes rendered framebuffers (`output/scanlands_benchmark_frame.png`) via PIL and NumPy to compute mean luminance, check for NaN/Inf values, and evaluate spatial variance verifying unoccluded 3D geometry.
  - Note: RGA (`/opt/RadeonDeveloperToolSuite-2026-05-28-1806/rga`) is utilized as an offline ISA analysis and profiling tool per `AGENTS.md`, rather than an automated in-test dependency.
- **Headless Shell Runner**: `scripts/run_scanlands_e2e_tests.sh`
  - Inspects GPU memory metrics via `/opt/rocm/core-10.0/bin/amd-smi metric --mem-usage`.
  - Builds project with `ninja -C build -j16`.
  - Executes CTest unit tests and Python E2E capability suite.
  - Generates structured summary report.

### Hardware & Execution Environment Guardrails
- Build concurrency capped at `-j16` (AMD Ryzen Threadripper 3970X).
- Single-GPU target: primary AMD Radeon AI PRO R9700 (32GB GDDR6, `gfx1201`).
- Strict memory guardrail: peak VRAM $< 24.0$ GB monitored via `/opt/rocm/core-10.0/bin/amd-smi metric --mem-usage`.
- Absolutely no inline Python (`python -c`); all scripts reside in dedicated test files.
- Operates outside sudo in system user environment.
