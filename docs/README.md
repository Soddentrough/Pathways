# Pathways Documentation Hub

Welcome to the official technical documentation directory for the **Pathways** real-time Vulkan 1.4 path tracing engine.

---

## Documentation Index

```
docs/
├── README.md                           # Documentation Hub & Navigation Portal (this file)
├── ARCHITECTURE.md                     # Core Engine Architecture & Technical Specification
├── BUILD_LINUX.md                      # Linux Build, Toolchain & Execution Guide
├── BUILD_WINDOWS.md                    # Windows 11 Build, MSYS2 & Packaging Guide
├── VULKAN_API_AUDIT.md                 # Complete Vulkan 1.4 API Call Audit & Platform Coverage
├── reports/                            # Empirical Benchmarking & Deep Technical Audits
│   ├── material_shader_review.md       # Comprehensive Material Shader & BSDF Mathematical Review
│   └── scanlands_benchmark_report.md   # Scanlands 359M Triangle Single-GPU Performance Benchmark
├── images/                             # Real-time 4K reference renders & architectural diagrams
└── [idea_* / scratch_*]                # Internal research scratchpads & feature proposals (gitignored)
```

---

## 1. Architecture & Engineering Specifications

- **[ARCHITECTURE.md](ARCHITECTURE.md)**:
  Comprehensive technical specification detailing Pathways' pure wavefront path tracing architecture, GPU-autonomous Device-Generated Commands (DGC), 3D Morton + Material dual-binning, zero-copy multi-GPU scaling (`VK_EXT_external_memory_host`), AMD FSR 3.1 spatiotemporal super-resolution, ReSTIR DI reservoir sampling, and OpenUSD stage ingestion.

---

## 2. Platform Build & Toolchain Guides

- **[BUILD_LINUX.md](BUILD_LINUX.md)**:
  Detailed build and runtime instructions for Linux workstations (Fedora, Ubuntu/Debian, Arch Linux). Covers C++23 compilers (GCC 14+, Clang 20+), Ninja, CMake Presets (`linux-release`, `linux-debug`, `linux-test`), Mesa RADV / AMD ROCm 10 driver configuration, multi-GPU streaming, and automated test execution.
- **[BUILD_WINDOWS.md](BUILD_WINDOWS.md)**:
  Complete Windows 11 setup guide using MSYS2 UCRT64 (Clang 20 + LLD or GCC 15), Vulkan SDK 1.4+, PowerShell orchestrator (`build.ps1`), batch launchers, and CPack installer generation.

---

## 3. API Compliance & Platform Analysis

- **[VULKAN_API_AUDIT.md](VULKAN_API_AUDIT.md)**:
  Rigorous audit of every Vulkan API call invoked across the Pathways engine and third-party libraries. Includes multi-platform coverage analysis across 2,300+ devices from the Vulkan Hardware Database, comparing Desktop PC, Android/Mobile, and Global hardware adoption. Can be updated live via `python3 scripts/audit_vulkan_api.py`.

---

## 4. Empirical Benchmarking & Technical Reports

The [`reports/`](reports/) directory contains formal research reports and architectural audits:

- **[reports/material_shader_review.md](reports/material_shader_review.md)**:
  Exhaustive 1,400+ line mathematical and microarchitectural audit of Pathways BSDF formulations (microfacet normal distributions, Smith correlated masking-shadowing, dielectric transmission, Airy thin-film iridescence, and RDNA 4 VGPR occupancy optimization).
- **[reports/scanlands_benchmark_report.md](reports/scanlands_benchmark_report.md)**:
  Empirical single-GPU performance benchmark running the `Scanlands` OpenUSD production landscape asset (187,491 foliage instances, 358.9M instanced triangles) on an AMD Radeon AI PRO R9700. Covers VRAM safety, dynamic TLAS generation, and unoccluded frame pacing.

---

## 5. Visual Reference Showcase

The [`images/`](images/) directory contains uncompressed native 4K reference renders demonstrating hardware throughput, physical light transport, and scene fidelity:
- **[images/cornell_box.png](images/cornell_box.png)**: Classic Cornell Box diffuse inter-reflection and dielectric glass transmission.
- **[images/breakfast_table.png](images/breakfast_table.png)**: 270K-triangle architectural interior with Venetian blind shadow penumbras.
- **[images/dragon.png](images/dragon.png)**: High-curvature Stanford Dragon with volumetric Beer-Lambert absorption and chromatic dispersion.
- **[images/point_instance_city.png](images/point_instance_city.png)**: Massive OpenUSD 40,000-instance urban environment (~49M triangles).

---

## 6. Developer Notes, Ideas & Scratchpads Taxonomy

To maintain a clean repository history, internal development notes and forward-looking research proposals are organized under prefixed naming conventions:
- **`docs/idea_*`**: Exploratory research proposals, theoretical math derivations, and prospective feature designs (e.g. `idea_dgc_material_sorting.md`, `idea_upscaling_architecture.md`, `idea_nrc.md`, `idea_restir.md`).
- **`docs/scratch_*`**: Transient developer scratchpads, internal milestone checklists, and historical telemetry logs (e.g. `scratch_profiling.md`, `scratch_todo.md`, `scratch_project_review_2026.md`).

These files are retained locally for developer reference and debugging but are excluded from git tracking via `.gitignore`.
