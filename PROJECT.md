# Project: Pathways Scanlands Architectural Capabilities & High-Density Point Instancing

## Architecture
Pathways is a pure Vulkan 1.4 path tracer optimized for AMD RDNA 4 (`gfx1201`) dual-GPU and single-GPU execution.
This project enhances the engine with:
1. Wavefront microkernel material enhancements:
   - Thin-walled two-sided diffuse transmission in `wavefront_shade_diffuse.comp`, `wavefront_shade_complex.comp`, and `MaterialGPU`.
   - Tangent-space normal mapping parity in `wavefront_shade_dielectric.comp` matching `raytrace.rchit`.
2. GPU Procedural Shading Architecture:
   - GLSL continuous procedural noise library (`shaders/compute/procedural_noise.glsl`) featuring analytical gradient Simplex 3D, Voronoi 2D, and domain-rotated fBM.
   - Direct procedural terrain and rock surface evaluation in `wavefront_shade_diffuse.comp`.
   - Directional wave normal evaluation in `wavefront_shade_dielectric.comp` coupled to native Beer-Lambert depth absorption.
3. High-Density Asset Ingestion & OpenUSD Point Instancing:
   - Conversion pipeline (`scripts/convert_scanlands_to_usd.py`) for `scenes/Scanlands.blend` authoring `UsdGeomPointInstancer` with 8 deduplicated BLAS prototypes (187,490 foliage elements).
   - Ingestion support in `src/scene/UsdLoader.cpp` translating point instancers directly into `VkAccelerationStructureInstanceKHR` TLAS instances.
   - Scalability parameters: `--instance-density` and `--cull-distance`.
   - Alpha mask mode support for foliage cards.
4. Single-GPU Performance & Physical Fidelity Benchmarking:
   - Profile on AMD Radeon AI PRO R9700 (32GB GDDR6, `gfx1201`).
   - Strict VRAM limit < 24.0 GB via `/opt/rocm/core-10.0/bin/amd-smi metric --mem-usage`.
   - 1080p / 1440p / 4K benchmarks, 1–3 bounces, density scaling.
   - Visual verification against Blender Cycles reference.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | Dielectric Normal Mapping Parity | Unpack tangents and evaluate normal maps in `wavefront_shade_dielectric.comp` | M1 | Survey 1 |
| F2 | Thin-Walled Diffuse Transmission | Two-sided diffuse transmission BSDF and NEE in diffuse/complex microkernels | M1 | Survey 1 |
| F3 | MaterialGPU Transmission Fields | Add diffuse transmission factor/texture to `MaterialGPU` and `Material` struct | M1 | Survey 1 |
| F4 | GLSL Procedural Noise Library | Implement `simplex3D_grad`, `hash33`, `voronoi2D`, `fbm3D_grad` in GLSL | M2 | Survey 2 |
| F5 | Procedural Terrain Shading | GPU procedural terrain/rock evaluation in `wavefront_shade_diffuse.comp` | M2 | Survey 2 |
| F6 | Procedural Water Waves & Beer-Lambert | Directional Gerstner wave normals coupled with Beer-Lambert depth absorption | M2 | Survey 2 |
| F7 | Procedural Material Flags | Encode procedural material tags in `MaterialGPU::type` bits 8–31 | M2 | Survey 2 |
| F8 | Scanlands USD Conversion Pipeline | Two-stage hybrid script extracting 8 BLAS prototypes and 187k point instances | M3 | Survey 3 |
| F9 | UsdGeomPointInstancer Loader Support | Ingest USD point instancers as native TLAS instances in `UsdLoader.cpp` | M3 | Survey 3 |
| F10 | Foliage Alpha Mask Parity | Support `ALPHA_MODE_MASK` for foliage cards in `UsdLoader.cpp` | M3 | Survey 3 |
| F11 | Configurable Density & Culling | CLI options `--instance-density` and `--cull-distance` | M3 | Survey 3 |
| F12 | Single-GPU Benchmarking & Profiling | Profile on AMD R9700 (VRAM < 24 GB, 1080p/1440p, 1–3 bounces, density tiers) | M4 | Survey 3 |
| F13 | Physical Fidelity Verification | Validate backlit foliage, water caustics, and terrain against Cycles via frame dump | M4 | Survey 1/2 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Thin-Walled Transmission & Dielectric Normal Parity | F1, F2, F3 | none | DONE |
| M2 | GPU Procedural Shading Architecture | F4, F5, F6, F7 | M1 | DONE |
| M3 | Asset Ingestion & Point Instancing Pipeline | F8, F9, F10, F11 | none | DONE |
| M4 | Single-GPU Performance & Fidelity Benchmarking | F12, F13 | M1, M2, M3 | DONE |
| E2E | E2E Testing Suite Track | Comprehensive multi-tier test suite across F1–F13 | none | DONE |

## Interface Contracts
### Material Definition ↔ Shaders
- `MaterialGPU`: std430 208-byte layout.
- Diffuse transmission: `float diffuseTransmission` at offset 192, `uint diffuseTransmissionTex` at offset 196, `vec2 diffuseTransPad` at offset 200.
- Procedural flags: `uint type` bitmask:
  - bits 0–7: base type (`0=DIFFUSE`, `1=METALLIC`, `2=DIELECTRIC`, `3=EMISSIVE`)
  - bit 9: `MATERIAL_FLAG_PROCEDURAL_TERRAIN (1u << 9)`
  - bit 11: `MATERIAL_FLAG_PROCEDURAL_WATER (1u << 11)`

### OpenUSD Ingestion ↔ Engine TLAS
- Point instancer positions, quaternions, and prototype indices map directly to `VkAccelerationStructureInstanceKHR`.
- Prototypes loaded once as BLAS; instancer populates instance transforms referencing prototype BLAS acceleration structures.
- Density downsampling uses deterministic hash: `hash(instanceId) < density * 0xFFFFFFFF`.
- Distance culling: `length(instancePos - cameraPos) <= cullDistance`.

## Code Layout
- `src/scene/Material.hpp`: `MaterialGPU` struct definition (208 bytes) and flag constants.
- `shaders/compute/wavefront_common.glsl`: `struct Material` and archetype classification.
- `shaders/compute/raytrace_comp.comp`, `shaders/compute/wavefront_persistent.comp`: 208-byte std430 `struct Material`.
- `shaders/compute/wavefront_shade_dielectric.comp`: Tangent unpacking, normal map perturbation, colinear tangent guard, procedural waves.
- `shaders/compute/wavefront_shade_diffuse.comp`: Two-sided diffuse transmission, procedural terrain shading.
- `shaders/compute/wavefront_shade_complex.comp`: Two-sided diffuse transmission for complex materials.
- `shaders/compute/procedural_noise.glsl`: GLSL noise primitives (`simplex3D_grad`, `voronoi2D`, `fbm3D_grad`, `evaluateWaterWaves`).
- `scripts/convert_scanlands_to_usd.py`: Python/bpy/pxr hybrid asset conversion script with atmospheric occlusion mesh pruning.
- `src/scene/UsdLoader.cpp`: OpenUSD `UsdGeomPointInstancer` parsing, alpha mask setup, foliage diffuse transmission wiring, dome light HDRI ingestion.
- `src/core/Config.hpp`, `src/core/Engine.cpp`: CLI arguments, culling/density parameters, environment map binding.
- `tests/e2e/test_scanlands_capabilities.py`: 145 genuine automated multi-tier tests.
