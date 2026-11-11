# Neural Radiance Caching (NRC) Architecture

## Overview

Pathways includes an optional online **Neural Radiance Cache (`--nrc`)** managed by [`NRCManager`](../src/rt/NRCManager.hpp) to terminate long diffuse indirect ray paths early (`--nrc-bounce`, default `2`) and query an on-device neural network trained continuously per frame.

---

## Active Compute Pipeline (`NrcBackend::NativeWave32`)

When `--nrc` is enabled on hardware supporting `VK_KHR_cooperative_matrix` (`16x16x16` FP16 Wave32 subgroup matrices, e.g., AMD RDNA 3/4 `gfx11`/`gfx1201`), [`NRCManager`](../src/rt/NRCManager.cpp) executes three Vulkan 1.4 compute passes:

1. **Encoding & Cooperative-Matrix Inference (`shaders/compute/nrc_encode_infer.comp`):**
   - Encodes world-space hit position (normalized to scene AABB `[boundsMin, boundsMax]`), shading normal, view direction, roughness, and diffuse albedo using frequency (`sin`/`cos` octave) + one-blob/spherical directional encoding into a 64-channel FP16 input vector.
   - Evaluates a 4-layer `64 -> 64 -> 64 -> 64 -> 3` MLP in shared memory + subgroup registers using `coopMatMulAdd` (`VK_KHR_cooperative_matrix`).
2. **Online Self-Training (`shaders/compute/nrc_train.comp`):**
   - A configurable fraction of rays (`--nrc-train-ratio`, default `0.03` = 3%) record `NRCTrainingRecord` entries into `m_trainQueue` (`m_maxTrainRecords = max(16384, pixels / 16)`).
   - Updates FP32 master weights + Adam moments (`m_masterWeights`, `m_adamMoments`) and converts updated weights to FP16 (`m_weights`) for next-frame inference.
3. **Final Radiance Splatting (`shaders/compute/nrc_resolve.comp`):**
   - Multiplies predicted cache irradiance by path throughput and accumulates into the active radiance targets (`uAccumImage` / `uMlDiffuseImage` / `uMlSpecularImage`).

---

## Backend Availability & Honest Provider Status

| Backend Enum | CLI Flag | Current Status |
| :--- | :--- | :--- |
| `NrcBackend::NativeWave32` | `--nrc-backend native` | **Active** on GPUs exposing `VK_KHR_cooperative_matrix` (allocated lazily when `--nrc` is enabled). |
| `NrcBackend::FfxRadianceCache` | `--nrc-backend ffx` | **Stub Slot Only** — AMD FidelityFX SDK v2.4.0 ships Radiance Caching `0.9.0` exclusively for DirectX 12 on Windows; no Vulkan provider binary or source exists upstream yet. Pathways retains the dynamic `dlopen`/`LoadLibrary` module slot (`amd_fidelityfx_radiancecache_vk`) and reports stub unavailability honestly until a native Vulkan provider ships. |
| `NrcBackend::Auto` | `--nrc` (default) | Uses a native Vulkan SDK provider if dynamically loaded; otherwise selects `NativeWave32`. |

---

## Current Limitations & Future Roadmap

- **Positional Encoding vs. Multi-Resolution Hash Grid:** The current encoder uses multi-octave `sin`/`cos` positional encoding rather than a spatial multi-resolution hash table; high-frequency sharp indirect shadow boundaries are smoother than Müller et al. (2021) hash-encoded NRC.
- **Multi-GPU Checkerboard Mode:** Secondary GPUs do not synchronize NRC weight updates across PCIe per frame; in `--mgpu tile` mode, NRC is disabled so even/odd tiles remain consistent, and the UI/telemetry explicitly reflects this inactive state.
- **Planned Upgrade (NIRC):** Self-training multi-bounce bootstrap suffixes combined with Neural Two-Level Monte Carlo (NIRC) residual error estimation to eliminate termination bias.
