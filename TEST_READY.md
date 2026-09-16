# TEST_READY: Pathways Scanlands E2E Automated Test Suite

## 1. Test Runner Invocation Commands

The comprehensive test suite can be invoked via either the dedicated shell runner or the direct Python test runner:

### Primary Invocation (Shell Runner)
```bash
./scripts/run_scanlands_e2e_tests.sh
```
*Features executed:*
- Queries AMD GPU hardware metrics and VRAM utilization via `/opt/rocm/core-10.0/bin/amd-smi`.
- Incremental compilation check via `ninja -C build -j16`.
- Execution of CTest unit test suites (`ctest --test-dir build --output-on-failure`).
- Execution of the complete multi-tier E2E test suite (145 tests across Tiers 1–4).

### Standalone E2E Capability Runner (Python)
```bash
python3 tests/e2e/test_scanlands_capabilities.py
```

---

## 2. Expected Results & Pass Criteria

- **Total Tests**: 145 tests
- **Pass Rate**: 100% (145 Passed, 0 Failed, 0 Errors)
- **Execution Time**: ~5 seconds
- **Exit Code**: `0` on success, `1` on failure
- **Single-GPU VRAM Threshold**: Strict peak VRAM $< 24.0$ GB on AMD Radeon AI PRO R9700 (`gfx1201`).
- **Build Constraints**: Build jobs strictly capped at `-j16` (no unbounded parallel jobs).
- **Environment**: System-wide Linux Wayland environment, zero virtualenvs/containers, runs outside sudo.

---

## 3. Test Coverage Summary by Tier

| Test Tier | Scope & Focus | Test Count | Pass / Fail | Pass Rate |
|---|---|---|---|---|
| **Tier 1** | Feature Coverage (Unit & Functional, F1–F13) | 65 | 65 / 0 | **100%** |
| **Tier 2** | Boundary & Corner Cases (F1–F13) | 65 | 65 / 0 | **100%** |
| **Tier 3** | Cross-Feature Interactions (Pairwise & Multi) | 10 | 10 / 0 | **100%** |
| **Tier 4** | Real-World Application Scenarios (Scanlands E2E) | 5 | 5 / 0 | **100%** |
| **Total** | **Comprehensive Multi-Tier Coverage** | **145** | **145 / 0** | **100.00%** |

---

## 4. Feature Coverage Checklist (F1 through F13)

| ID | Feature Name | Tier 1 (Unit) | Tier 2 (Boundary) | Tier 3 (Cross) | Status |
|---|---|---|---|---|---|
| **F1** | Dielectric Normal Mapping Parity | 5 tests (F1-T01..05) | 5 tests (F1-B01..05) | XF-01 | **COVERED** |
| **F2** | Thin-Walled Diffuse Transmission | 5 tests (F2-T01..05) | 5 tests (F2-B01..05) | XF-02, XF-08 | **COVERED** |
| **F3** | MaterialGPU Transmission Fields | 5 tests (F3-T01..05) | 5 tests (F3-B01..05) | XF-03 | **COVERED** |
| **F4** | GLSL Procedural Noise Library | 5 tests (F4-T01..05) | 5 tests (F4-B01..05) | XF-09 | **COVERED** |
| **F5** | Procedural Terrain Shading | 5 tests (F5-T01..05) | 5 tests (F5-B01..05) | XF-05, XF-09 | **COVERED** |
| **F6** | Procedural Water Waves & Beer-Lambert | 5 tests (F6-T01..05) | 5 tests (F6-B01..05) | XF-01, XF-06, XF-09 | **COVERED** |
| **F7** | Procedural Material Flags | 5 tests (F7-T01..05) | 5 tests (F7-B01..05) | XF-03 | **COVERED** |
| **F8** | Scanlands USD Conversion Pipeline | 5 tests (F8-T01..05) | 5 tests (F8-B01..05) | XF-07 | **COVERED** |
| **F9** | UsdGeomPointInstancer Loader Support | 5 tests (F9-T01..05) | 5 tests (F9-B01..05) | XF-04, XF-07, XF-10 | **COVERED** |
| **F10** | Foliage Alpha Mask Parity | 5 tests (F10-T01..05) | 5 tests (F10-B01..05) | XF-02, XF-10 | **COVERED** |
| **F11** | Configurable Density & Culling | 5 tests (F11-T01..05) | 5 tests (F11-B01..05) | XF-04, XF-10 | **COVERED** |
| **F12** | Single-GPU Benchmarking & Profiling | 5 tests (F12-T01..05) | 5 tests (F12-B01..05) | XF-05, XF-10 | **COVERED** |
| **F13** | Physical Fidelity Verification | 5 tests (F13-T01..05) | 5 tests (F13-B01..05) | XF-06, XF-08 | **COVERED** |

---

## 5. Test Artifacts Summary

1. **`TEST_INFRA.md`** — Specification of test architecture, mathematical models, contracts C1–C3, and exhaustive test matrix for Tiers 1–4.
2. **`TEST_READY.md`** — Runner invocation documentation, expected results, and verification checklist (this file).
3. **`tests/e2e/test_scanlands_capabilities.py`** — Automated multi-tier executable test suite containing 145 independent test cases.
4. **`scripts/run_scanlands_e2e_tests.sh`** — Headless regression execution wrapper adhering to Threadripper `-j16` limits, AMD SMI monitoring, CTest execution, and E2E verification.
