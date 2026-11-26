#pragma once

#include "scene/Light.hpp"

#include <glm/glm.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

namespace pathways {

// =============================================================================
// 2D Equirectangular Environment Importance Table (Binding 42 SSBO)
// =============================================================================
// Builds a luminance * sin(theta) weighted Vose Alias Table on the CPU for O(1)
// GPU environment-map Next Event Estimation (NEE) and exact balance-heuristic MIS.
//
// Layout matches shaders/compute/env_importance.glsl (std430):
//   Header (16 bytes): width, height, envSelectProb, envTotalFlux
//   Entries (16 bytes each): q (threshold), alias (uint32 index), pdf (P * W * H), pad
// =============================================================================

struct EnvImportanceHeader {
    uint32_t width = 0;         // Importance grid width (0 => env NEE disabled)
    uint32_t height = 0;        // Importance grid height
    float envSelectProb = 0.0f; // Mixture probability p_env in [0, 1]
    float envTotalFlux = 0.0f;  // Integral_{S^2} L_luma(w) dw
};
static_assert(sizeof(EnvImportanceHeader) == 16, "EnvImportanceHeader must be 16 bytes (std430)");

struct EnvAliasEntry {
    float q = 1.0f;             // Vose alias threshold in [0, 1]
    uint32_t alias = 0;         // Alias linear cell index in [0, width * height - 1]
    float pdf = 0.0f;           // Continuous UV density p_uv(cell) = P(cell) * width * height
    float pad = 0.0f;           // 16-byte aligned (one 128-bit RDNA4 vector load)
};
static_assert(sizeof(EnvAliasEntry) == 16, "EnvAliasEntry must be 16 bytes (std430)");

struct EnvImportanceTable {
    EnvImportanceHeader header{};
    std::vector<EnvAliasEntry> entries{ EnvAliasEntry{} };
};

namespace detail {
constexpr float kEnvPi = 3.14159265358979323846f;
constexpr double kEnvPiD = 3.14159265358979323846;

inline float envLuminance(float r, float g, float b) {
    return 0.2126f * r + 0.7152f * g + 0.0722f * b;
}
} // namespace detail

/// Sanitizes float RGBA pixels in-place: replaces non-finite or negative RGB values
/// with 0.0, applies optional tint, and forces alpha to 1.0.
inline void sanitizeHdrRgbaPixels(float* rgba, size_t pixelCount, const glm::vec3& tint = glm::vec3(1.0f)) {
    if (!rgba || pixelCount == 0) return;
    const glm::vec3 safeTint(
        (std::isfinite(tint.x) && tint.x > 0.0f) ? tint.x : 0.0f,
        (std::isfinite(tint.y) && tint.y > 0.0f) ? tint.y : 0.0f,
        (std::isfinite(tint.z) && tint.z > 0.0f) ? tint.z : 0.0f
    );
    for (size_t i = 0; i < pixelCount; ++i) {
        float* p = rgba + i * 4;
        float r = (std::isfinite(p[0]) && p[0] > 0.0f) ? p[0] * safeTint.x : 0.0f;
        float g = (std::isfinite(p[1]) && p[1] > 0.0f) ? p[1] * safeTint.y : 0.0f;
        float b = (std::isfinite(p[2]) && p[2] > 0.0f) ? p[2] * safeTint.z : 0.0f;
        p[0] = r;
        p[1] = g;
        p[2] = b;
        p[3] = 1.0f;
    }
}

/// Converts unit direction to equirectangular UVs matching GLSL directionToEquirectangular().
inline glm::vec2 directionToEquirectangularCpu(const glm::vec3& rawDir) {
    float len = glm::length(rawDir);
    glm::vec3 dir = (len > 1e-8f) ? (rawDir / len) : glm::vec3(0.0f, 1.0f, 0.0f);
    float phi = std::atan2(dir.z, dir.x);
    float theta = std::acos(std::clamp(dir.y, -1.0f, 1.0f));
    return glm::vec2(
        (phi + detail::kEnvPi) / (2.0f * detail::kEnvPi),
        theta / detail::kEnvPi
    );
}

/// Converts equirectangular UVs to world-space unit direction (exact inverse).
inline glm::vec3 equirectangularToDirectionCpu(const glm::vec2& uv) {
    float phi = uv.x * (2.0f * detail::kEnvPi) - detail::kEnvPi;
    float theta = uv.y * detail::kEnvPi;
    float sinTheta = std::sin(theta);
    float cosTheta = std::cos(theta);
    return glm::vec3(sinTheta * std::cos(phi), cosTheta, sinTheta * std::sin(phi));
}

/// Builds a 2D Vose Alias Table for equirectangular RGBA32F pixels.
/// Capped at (maxGridW, maxGridH) (default 1024x512 -> <= 8 MB VRAM).
inline EnvImportanceTable buildEnvImportanceTable(
    const float* rgba,
    uint32_t srcW,
    uint32_t srcH,
    uint32_t maxGridW = 1024u,
    uint32_t maxGridH = 512u
) {
    EnvImportanceTable table{};
    if (!rgba || srcW == 0 || srcH == 0 || maxGridW == 0 || maxGridH == 0) {
        return table;
    }

    const uint32_t W = std::min(srcW, maxGridW);
    const uint32_t H = std::min(srcH, maxGridH);
    const size_t totalCells = static_cast<size_t>(W) * static_cast<size_t>(H);

    // 1. Box-downsample source luminance onto target importance grid (W x H)
    std::vector<double> cellLuma(totalCells, 0.0);
    for (uint32_t y = 0; y < H; ++y) {
        uint32_t y0 = static_cast<uint32_t>((static_cast<uint64_t>(y) * srcH) / H);
        uint32_t y1 = static_cast<uint32_t>(((static_cast<uint64_t>(y + 1) * srcH) + H - 1u) / H);
        y1 = std::max(y0 + 1u, std::min(y1, srcH));
        for (uint32_t x = 0; x < W; ++x) {
            uint32_t x0 = static_cast<uint32_t>((static_cast<uint64_t>(x) * srcW) / W);
            uint32_t x1 = static_cast<uint32_t>(((static_cast<uint64_t>(x + 1) * srcW) + W - 1u) / W);
            x1 = std::max(x0 + 1u, std::min(x1, srcW));

            double sum = 0.0;
            uint64_t count = 0;
            for (uint32_t sy = y0; sy < y1; ++sy) {
                const float* row = rgba + static_cast<size_t>(sy) * srcW * 4;
                for (uint32_t sx = x0; sx < x1; ++sx) {
                    const float r = row[sx * 4 + 0];
                    const float g = row[sx * 4 + 1];
                    const float b = row[sx * 4 + 2];
                    float l = 0.0f;
                    if (std::isfinite(r) && std::isfinite(g) && std::isfinite(b)) {
                        l = detail::envLuminance(std::max(r, 0.0f), std::max(g, 0.0f), std::max(b, 0.0f));
                    }
                    sum += static_cast<double>(l);
                    ++count;
                }
            }
            cellLuma[static_cast<size_t>(y) * W + x] = (count > 0) ? (sum / static_cast<double>(count)) : 0.0;
        }
    }

    // 2. Compute exact spherical luminance flux: Integral_{S^2} L(w) dw
    const double cellUvArea = 1.0 / static_cast<double>(totalCells);
    const double jacobianConst = 2.0 * detail::kEnvPiD * detail::kEnvPiD * cellUvArea;
    double totalFlux = 0.0;
    double lumaSum = 0.0;
    for (uint32_t y = 0; y < H; ++y) {
        double theta = detail::kEnvPiD * (static_cast<double>(y) + 0.5) / static_cast<double>(H);
        double dOmega = std::sin(theta) * jacobianConst;
        const size_t rowOff = static_cast<size_t>(y) * W;
        for (uint32_t x = 0; x < W; ++x) {
            double l = cellLuma[rowOff + x];
            lumaSum += l;
            totalFlux += l * dOmega;
        }
    }

    if (totalFlux <= 1e-12 || !std::isfinite(totalFlux)) {
        return table; // Disabled (black / zero env map)
    }

    // 3. Apply bilinear-bleed-safe 3x3 tent dilation (wrapping in U, clamping in V)
    //    plus a tiny positive floor (1e-6 * meanLuma) so linear texel bleed around
    //    sharp suns/softboxes never yields zero PDF or extreme L/p spikes.
    const double meanLuma = lumaSum / static_cast<double>(totalCells);
    const double tinyFloor = 1e-6 * std::max(meanLuma, 1e-12);
    std::vector<double> weights(totalCells, 0.0);
    double sumWeights = 0.0;
    for (uint32_t y = 0; y < H; ++y) {
        double theta = detail::kEnvPiD * (static_cast<double>(y) + 0.5) / static_cast<double>(H);
        double sinTheta = std::max(std::sin(theta), 1e-6);
        for (uint32_t x = 0; x < W; ++x) {
            double nbrSum = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                int yy = std::clamp(static_cast<int>(y) + dy, 0, static_cast<int>(H) - 1);
                for (int dx = -1; dx <= 1; ++dx) {
                    int xx = (static_cast<int>(x) + dx + static_cast<int>(W)) % static_cast<int>(W);
                    nbrSum += cellLuma[static_cast<size_t>(yy) * W + static_cast<size_t>(xx)];
                }
            }
            double center = cellLuma[static_cast<size_t>(y) * W + x];
            double smoothed = 0.5 * center + 0.5 * (nbrSum / 9.0);
            double w = std::max(smoothed, tinyFloor) * sinTheta;
            weights[static_cast<size_t>(y) * W + x] = w;
            sumWeights += w;
        }
    }

    if (sumWeights <= 0.0 || !std::isfinite(sumWeights)) {
        return table;
    }

    // 4. Construct exact O(1) Vose Alias Table
    table.header.width = W;
    table.header.height = H;
    table.header.envSelectProb = 0.0f; // Caller fills via computeEnvSelectProb()
    table.header.envTotalFlux = static_cast<float>(totalFlux);
    table.entries.resize(totalCells);

    std::vector<double> scaledProb(totalCells);
    std::vector<size_t> smallList;
    std::vector<size_t> largeList;
    smallList.reserve(totalCells);
    largeList.reserve(totalCells);

    const double invSum = 1.0 / sumWeights;
    const double fTotal = static_cast<double>(totalCells);
    for (size_t i = 0; i < totalCells; ++i) {
        double p = weights[i] * invSum;
        table.entries[i].pdf = static_cast<float>(p * fTotal);
        table.entries[i].pad = 0.0f;
        scaledProb[i] = p * fTotal;
        if (scaledProb[i] < 1.0) {
            smallList.push_back(i);
        } else {
            largeList.push_back(i);
        }
    }

    while (!smallList.empty() && !largeList.empty()) {
        size_t s = smallList.back();
        smallList.pop_back();
        size_t l = largeList.back();
        largeList.pop_back();

        table.entries[s].q = static_cast<float>(std::clamp(scaledProb[s], 0.0, 1.0));
        table.entries[s].alias = static_cast<uint32_t>(l);

        scaledProb[l] = (scaledProb[l] + scaledProb[s]) - 1.0;
        if (scaledProb[l] < 1.0) {
            smallList.push_back(l);
        } else {
            largeList.push_back(l);
        }
    }

    while (!largeList.empty()) {
        size_t l = largeList.back();
        largeList.pop_back();
        table.entries[l].q = 1.0f;
        table.entries[l].alias = static_cast<uint32_t>(l);
    }
    while (!smallList.empty()) {
        size_t s = smallList.back();
        smallList.pop_back();
        table.entries[s].q = 1.0f;
        table.entries[s].alias = static_cast<uint32_t>(s);
    }

    return table;
}

/// Estimates explicit light irradiance at scene scale vs environment irradiance
/// to determine the NEE mixture probability p_env in [0, 1].
inline float computeExplicitLightIrradianceEstimate(const std::vector<LightGPU>& lights, float sceneRadius) {
    if (lights.empty()) return 0.0f;
    const float safeR = std::max(sceneRadius, 1.0f);
    const float sceneArea = 4.0f * detail::kEnvPi * safeR * safeR;
    double irrSum = 0.0;
    for (const auto& l : lights) {
        uint32_t type = static_cast<uint32_t>(l.position.w);
        if (type == LIGHT_DIRECTIONAL) {
            float lum = std::max(detail::envLuminance(l.emission.r, l.emission.g, l.emission.b), 0.0f);
            irrSum += static_cast<double>(lum);
        } else {
            float flux = (l.sampling.w > 0.0f) ? l.sampling.w : calculateLightFlux(l);
            irrSum += static_cast<double>(std::max(flux, 0.0f)) / static_cast<double>(sceneArea);
        }
    }
    return static_cast<float>(irrSum);
}

/// Computes unbiased mixture probability p_env between environment NEE and explicit lights.
inline float computeEnvSelectProb(
    bool hasEnvMap,
    float envTotalFlux,
    float envIntensity,
    uint32_t numLights,
    float totalExplicitIrradiance,
    bool isExplicitOrDomeHdri,
    bool hasFallbackSun
) {
    if (!hasEnvMap || !(envTotalFlux > 1e-7f) || !(envIntensity > 0.0f)) {
        return 0.0f;
    }
    if (numLights == 0u) {
        return 1.0f;
    }
    if (isExplicitOrDomeHdri) {
        const float envIrr = (envTotalFlux * envIntensity) * 0.25f;
        const float denom = envIrr + std::max(totalExplicitIrradiance, 0.0f);
        const float frac = (denom > 1e-8f) ? (envIrr / denom) : 0.25f;
        return std::clamp(frac, 0.20f, 0.50f);
    }
    if (hasFallbackSun) {
        return 0.25f;
    }
    // Enclosed indoor / authored-light scene without explicit HDRI or DomeLight:
    // keep env NEE at 0 so interior tests pay zero sample-budget tax while BSDF still hits sky.
    return 0.0f;
}

/// Packs EnvImportanceTable into contiguous std430 bytes (at least 32 bytes).
inline std::vector<uint8_t> packEnvImportanceBufferBytes(const EnvImportanceTable& table) {
    const size_t entryCount = std::max<size_t>(1, table.entries.size());
    std::vector<uint8_t> bytes(sizeof(EnvImportanceHeader) + entryCount * sizeof(EnvAliasEntry), 0);
    std::memcpy(bytes.data(), &table.header, sizeof(EnvImportanceHeader));
    if (!table.entries.empty()) {
        std::memcpy(bytes.data() + sizeof(EnvImportanceHeader),
                    table.entries.data(),
                    table.entries.size() * sizeof(EnvAliasEntry));
    }
    return bytes;
}

/// CPU mirror of GLSL evalEnvMapPdf(vec3 dir) for unit testing.
inline float evalEnvMapPdfCpu(const EnvImportanceTable& table, const glm::vec3& rawDir) {
    const uint32_t W = table.header.width;
    const uint32_t H = table.header.height;
    if (W == 0 || H == 0 || !(table.header.envSelectProb > 0.0f) || table.entries.size() < static_cast<size_t>(W) * H) {
        return 0.0f;
    }
    float len = glm::length(rawDir);
    glm::vec3 dir = (len > 1e-8f) ? (rawDir / len) : glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec2 uv = directionToEquirectangularCpu(dir);
    uint32_t x = std::min(static_cast<uint32_t>(std::max(uv.x, 0.0f) * static_cast<float>(W)), W - 1u);
    uint32_t y = std::min(static_cast<uint32_t>(std::max(uv.y, 0.0f) * static_cast<float>(H)), H - 1u);
    float uvPdf = table.entries[static_cast<size_t>(y) * W + x].pdf;
    float sinTheta = std::max(std::sqrt(dir.x * dir.x + dir.z * dir.z), 1e-5f);
    return uvPdf / (2.0f * detail::kEnvPi * detail::kEnvPi * sinTheta);
}

/// CPU mirror of GLSL sampleEnvMapImportance(vec3 u, ...) for unit testing.
inline bool sampleEnvMapImportanceCpu(
    const EnvImportanceTable& table,
    float u0, float u1, float u2,
    glm::vec3& outDir,
    glm::vec2& outUv,
    float& outSolidAnglePdf
) {
    const uint32_t W = table.header.width;
    const uint32_t H = table.header.height;
    if (W == 0 || H == 0 || !(table.header.envSelectProb > 0.0f) || table.entries.size() < static_cast<size_t>(W) * H) {
        return false;
    }
    const uint32_t total = W * H;
    const float scaled = std::clamp(u0, 0.0f, 0.9999999f) * static_cast<float>(total);
    const uint32_t col = std::min(static_cast<uint32_t>(scaled), total - 1u);
    const float frac = scaled - static_cast<float>(col);
    const EnvAliasEntry& e = table.entries[col];
    const uint32_t cell = (frac < e.q) ? col : e.alias;
    const float uvPdf = (cell == col) ? e.pdf : table.entries[cell].pdf;
    const uint32_t cx = cell % W;
    const uint32_t cy = cell / W;
    const float jx = std::clamp(u1, 1e-4f, 1.0f - 1e-4f);
    const float jy = std::clamp(u2, 1e-4f, 1.0f - 1e-4f);
    outUv = glm::vec2((static_cast<float>(cx) + jx) / static_cast<float>(W),
                      (static_cast<float>(cy) + jy) / static_cast<float>(H));
    outDir = equirectangularToDirectionCpu(outUv);
    const float sinTheta = std::max(std::sqrt(outDir.x * outDir.x + outDir.z * outDir.z), 1e-5f);
    outSolidAnglePdf = uvPdf / (2.0f * detail::kEnvPi * detail::kEnvPi * sinTheta);
    return outSolidAnglePdf > 0.0f;
}

} // namespace pathways
