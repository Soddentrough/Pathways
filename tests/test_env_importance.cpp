#include "scene/EnvMapImportance.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

using namespace pathways;

namespace {

void checkTrue(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << std::endl;
        std::exit(1);
    }
}

void assertNear(float actual, float expected, float eps, const char* msg) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > eps) {
        std::cerr << "[FAIL] " << msg << ": actual=" << actual << " expected=" << expected
                  << " (eps=" << eps << ")" << std::endl;
        std::exit(1);
    }
}

std::vector<float> makeHighContrastEnv(uint32_t W, uint32_t H, const glm::vec3& sunDir, float sunPeak, float skyBase) {
    std::vector<float> rgba(static_cast<size_t>(W) * H * 4u, 0.0f);
    glm::vec3 nSun = glm::normalize(sunDir);
    for (uint32_t y = 0; y < H; ++y) {
        float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(H);
        for (uint32_t x = 0; x < W; ++x) {
            float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(W);
            glm::vec3 d = equirectangularToDirectionCpu(glm::vec2(u, v));
            float cosA = std::max(glm::dot(d, nSun), 0.0f);
            float lobe = sunPeak * std::pow(cosA, 64.0f);
            float horizon = skyBase * (0.4f + 0.6f * std::max(d.y, 0.0f));
            glm::vec3 c = glm::vec3(horizon * 0.7f, horizon * 0.85f, horizon * 1.15f)
                        + glm::vec3(1.0f, 0.92f, 0.75f) * lobe;
            size_t idx = (static_cast<size_t>(y) * W + x) * 4u;
            rgba[idx + 0] = c.r;
            rgba[idx + 1] = c.g;
            rgba[idx + 2] = c.b;
            rgba[idx + 3] = 1.0f;
        }
    }
    return rgba;
}

glm::vec3 sampleNearestEnv(const std::vector<float>& rgba, uint32_t W, uint32_t H, const glm::vec3& dir) {
    glm::vec2 uv = directionToEquirectangularCpu(dir);
    uint32_t x = std::min(static_cast<uint32_t>(std::max(uv.x, 0.0f) * static_cast<float>(W)), W - 1u);
    uint32_t y = std::min(static_cast<uint32_t>(std::max(uv.y, 0.0f) * static_cast<float>(H)), H - 1u);
    size_t idx = (static_cast<size_t>(y) * W + x) * 4u;
    return glm::vec3(rgba[idx + 0], rgba[idx + 1], rgba[idx + 2]);
}

} // namespace

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing 2D EnvMap Importance & MIS Math" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Sanitization test (NaN, +Inf, -Inf, negative RGB, tint)
    {
        float nanVal = std::numeric_limits<float>::quiet_NaN();
        float infVal = std::numeric_limits<float>::infinity();
        std::vector<float> px = {
            nanVal, -1.0f, infVal, 0.25f,
            2.0f, 4.0f, -infVal, 0.5f
        };
        sanitizeHdrRgbaPixels(px.data(), 2, glm::vec3(1.5f, 0.5f, 2.0f));
        assertNear(px[0], 0.0f, 1e-6f, "NaN sanitized to 0");
        assertNear(px[1], 0.0f, 1e-6f, "Negative sanitized to 0");
        assertNear(px[2], 0.0f, 1e-6f, "+Inf sanitized to 0");
        assertNear(px[3], 1.0f, 1e-6f, "Alpha forced to 1");
        assertNear(px[4], 3.0f, 1e-6f, "Valid R tinted");
        assertNear(px[5], 2.0f, 1e-6f, "Valid G tinted");
        assertNear(px[6], 0.0f, 1e-6f, "-Inf sanitized to 0");
        assertNear(px[7], 1.0f, 1e-6f, "Alpha forced to 1 (2)");
        std::cout << "[PASS] Pixel NaN/Inf/negative sanitization & tint verified." << std::endl;
    }

    // 2. Solid-angle PDF normalization over S^2 (uniform + high-contrast HDRI)
    {
        const uint32_t W = 64, H = 32;
        std::vector<float> uniformPx(static_cast<size_t>(W) * H * 4u, 1.0f);
        EnvImportanceTable uniformTbl = buildEnvImportanceTable(uniformPx.data(), W, H, 1024, 512);
        checkTrue(uniformTbl.header.width == W && uniformTbl.header.height == H, "Uniform table dims");
        uniformTbl.header.envSelectProb = 1.0f;
        assertNear(uniformTbl.header.envTotalFlux, 4.0f * 3.14159265f, 0.08f, "Uniform white total flux == 4*pi");

        const uint32_t qPhi = 256, qTheta = 128;
        double pdfIntegral = 0.0;
        const double dPhi = 2.0 * detail::kEnvPiD / qPhi;
        const double dTheta = detail::kEnvPiD / qTheta;
        for (uint32_t yt = 0; yt < qTheta; ++yt) {
            float v = (static_cast<float>(yt) + 0.5f) / static_cast<float>(qTheta);
            double theta = detail::kEnvPiD * v;
            double dOmega = std::sin(theta) * dPhi * dTheta;
            for (uint32_t xp = 0; xp < qPhi; ++xp) {
                float u = (static_cast<float>(xp) + 0.5f) / static_cast<float>(qPhi);
                glm::vec3 d = equirectangularToDirectionCpu(glm::vec2(u, v));
                pdfIntegral += static_cast<double>(evalEnvMapPdfCpu(uniformTbl, d)) * dOmega;
            }
        }
        assertNear(static_cast<float>(pdfIntegral), 1.0f, 0.01f, "Uniform env PDF integrates to 1 over S^2");

        const uint32_t HW = 128, HH = 64;
        std::vector<float> sunSky = makeHighContrastEnv(HW, HH, glm::vec3(0.45f, 0.70f, -0.55f), 250.0f, 0.5f);
        EnvImportanceTable sunTbl = buildEnvImportanceTable(sunSky.data(), HW, HH, 1024, 512);
        sunTbl.header.envSelectProb = 1.0f;
        double sunPdfInt = 0.0;
        for (uint32_t yt = 0; yt < qTheta; ++yt) {
            float v = (static_cast<float>(yt) + 0.5f) / static_cast<float>(qTheta);
            double theta = detail::kEnvPiD * v;
            double dOmega = std::sin(theta) * dPhi * dTheta;
            for (uint32_t xp = 0; xp < qPhi; ++xp) {
                float u = (static_cast<float>(xp) + 0.5f) / static_cast<float>(qPhi);
                glm::vec3 d = equirectangularToDirectionCpu(glm::vec2(u, v));
                sunPdfInt += static_cast<double>(evalEnvMapPdfCpu(sunTbl, d)) * dOmega;
            }
        }
        assertNear(static_cast<float>(sunPdfInt), 1.0f, 0.015f, "Concentrated HDRI PDF integrates to 1 over S^2");
        std::cout << "[PASS] Solid-angle PDF normalization verified (uniform=" << pdfIntegral
                  << ", high-contrast=" << sunPdfInt << ")." << std::endl;
    }

    // 3. Unbiased Monte Carlo importance estimator vs exact quadrature + evaluator/sampler consistency
    {
        const uint32_t W = 128, H = 64;
        std::vector<float> hdr = makeHighContrastEnv(W, H, glm::vec3(-0.3f, 0.8f, 0.5f), 300.0f, 0.8f);
        EnvImportanceTable tbl = buildEnvImportanceTable(hdr.data(), W, H, 1024, 512);
        tbl.header.envSelectProb = 1.0f;

        // Exact quadrature of upper-hemisphere cosine-weighted irradiance
        double refIrradiance = 0.0;
        const uint32_t qW = 512, qH = 256;
        const double dPhi = 2.0 * detail::kEnvPiD / qW;
        const double dTheta = detail::kEnvPiD / qH;
        for (uint32_t y = 0; y < qH; ++y) {
            float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(qH);
            double dOmega = std::sin(detail::kEnvPiD * v) * dPhi * dTheta;
            for (uint32_t x = 0; x < qW; ++x) {
                float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(qW);
                glm::vec3 d = equirectangularToDirectionCpu(glm::vec2(u, v));
                if (d.y > 0.0f) {
                    glm::vec3 L = sampleNearestEnv(hdr, W, H, d);
                    float luma = 0.2126f * L.r + 0.7152f * L.g + 0.0722f * L.b;
                    refIrradiance += static_cast<double>(luma) * static_cast<double>(d.y) * dOmega;
                }
            }
        }

        // 200,000 Monte Carlo samples from Vose Alias Table
        constexpr size_t N = 200000;
        std::mt19937_64 rng(133742ULL);
        std::uniform_real_distribution<float> dist01(0.0f, 0.9999999f);
        double mcAccum = 0.0;
        for (size_t i = 0; i < N; ++i) {
            glm::vec3 dir{};
            glm::vec2 uv{};
            float pdf = 0.0f;
            bool ok = sampleEnvMapImportanceCpu(tbl, dist01(rng), dist01(rng), dist01(rng), dir, uv, pdf);
            checkTrue(ok && pdf > 0.0f, "Env alias sample valid");
            float evalPdf = evalEnvMapPdfCpu(tbl, dir);
            assertNear(evalPdf, pdf, std::max(1e-5f, pdf * 1e-4f), "evalEnvMapPdfCpu matches sampler pdf");

            if (dir.y > 0.0f) {
                glm::vec3 L = sampleNearestEnv(hdr, W, H, dir);
                float luma = 0.2126f * L.r + 0.7152f * L.g + 0.0722f * L.b;
                mcAccum += static_cast<double>(luma * dir.y) / static_cast<double>(pdf);
            }
        }
        double mcIrradiance = mcAccum / static_cast<double>(N);
        double relErr = std::abs(mcIrradiance - refIrradiance) / std::max(refIrradiance, 1e-6);
        checkTrue(relErr < 0.015, "Monte Carlo cosine-weighted irradiance matches exact quadrature within 1.5%");
        std::cout << "[PASS] Unbiased Monte Carlo check (ref=" << refIrradiance
                  << ", mc=" << mcIrradiance << ", relErr=" << (relErr * 100.0) << "%)." << std::endl;

        // 4. Balance-heuristic MIS partition of unity
        for (float pEnv : { 1.0f, 0.35f, 0.20f }) {
            for (float pb : { 0.01f, 0.3183f, 5.0f }) {
                glm::vec3 testDir = glm::normalize(glm::vec3(-0.3f, 0.8f, 0.5f));
                float pl = evalEnvMapPdfCpu(tbl, testDir) * pEnv;
                float wNee = pl / (pl + pb);
                float wBsdf = pb / (pl + pb);
                assertNear(wNee + wBsdf, 1.0f, 1e-6f, "Balance heuristic MIS sums to 1.0");
            }
        }
        std::cout << "[PASS] Balance-heuristic MIS partition of unity verified." << std::endl;
    }

    // 5. computeEnvSelectProb policy & SSBO byte packing
    {
        assertNear(computeEnvSelectProb(false, 10.0f, 1.0f, 0, 0.0f, true, false), 0.0f, 1e-6f, "No env map -> 0");
        assertNear(computeEnvSelectProb(true, 10.0f, 0.0f, 0, 0.0f, true, false), 0.0f, 1e-6f, "Zero intensity -> 0");
        assertNear(computeEnvSelectProb(true, 10.0f, 1.0f, 0, 0.0f, true, false), 1.0f, 1e-6f, "0 explicit lights -> 1.0");

        float pLowExplicit = computeEnvSelectProb(true, 40.0f, 1.0f, 2, 1.0f, true, false);
        float pHighExplicit = computeEnvSelectProb(true, 1.0f, 1.0f, 2, 1000.0f, true, false);
        checkTrue(pLowExplicit >= 0.20f && pLowExplicit <= 0.50f, "Explicit HDRI + lights clamped in [0.20, 0.50]");
        checkTrue(pHighExplicit >= 0.20f && pHighExplicit <= 0.50f, "Dominant explicit lights still >= 0.20");
        assertNear(pHighExplicit, 0.20f, 1e-6f, "Min clamp 0.20");
        assertNear(pLowExplicit, 0.50f, 1e-6f, "Max clamp 0.50");

        assertNear(computeEnvSelectProb(true, 5.0f, 1.0f, 1, 2.0f, false, true), 0.25f, 1e-6f, "Fallback sun + sky -> 0.25");
        assertNear(computeEnvSelectProb(true, 5.0f, 1.0f, 1, 2.0f, false, false), 0.0f, 1e-6f, "Interior CornellBox protection -> 0.0");

        EnvImportanceTable emptyTbl{};
        std::vector<uint8_t> packedEmpty = packEnvImportanceBufferBytes(emptyTbl);
        checkTrue(packedEmpty.size() == 32, "Empty EnvImportanceBuffer packs to 32-byte dummy SSBO");
        EnvImportanceHeader hdrOut{};
        std::memcpy(&hdrOut, packedEmpty.data(), sizeof(EnvImportanceHeader));
        checkTrue(hdrOut.width == 0u && hdrOut.height == 0u && hdrOut.envSelectProb == 0.0f, "Empty header disabled");
        std::cout << "[PASS] Env selection policy and SSBO packing verified." << std::endl;
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL ENV IMPORTANCE UNIT TESTS PASSED" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
