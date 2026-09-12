#pragma once

#include <glm/glm.hpp>
#include <cstdint>
#include <vector>
#include <algorithm>
#include <cmath>
#include <bit>

namespace pathways {

enum LightType : uint32_t {
    LIGHT_AREA_QUAD = 0,
    LIGHT_SPOT = 1,
    LIGHT_DIRECTIONAL = 2
};

struct LightGPU {
    glm::vec4 position; // xyz: corner or position, w: type
    glm::vec4 emission; // rgb: color * intensity, w: area
    glm::vec4 u;        // xyz: edge vector 1, w: spot inner angle cos
    glm::vec4 v;        // xyz: edge vector 2, w: spot outer angle cos
    glm::vec4 normal;   // xyz: normal or direction, w: padding
    glm::vec4 sampling; // x: q (threshold), y: uintBitsToFloat(aliasIdx), z: discretePdf, w: flux (96 bytes total)
};

inline float calculateLightFlux(const LightGPU& light) {
    float lum = 0.2126f * light.emission.r + 0.7152f * light.emission.g + 0.0722f * light.emission.b;
    if (lum <= 0.0f) return 0.0f;
    uint32_t type = static_cast<uint32_t>(light.position.w);
    if (type == LIGHT_AREA_QUAD) {
        float area = light.emission.w;
        if (area <= 0.0f) {
            glm::vec3 u(light.u);
            glm::vec3 v(light.v);
            area = glm::length(glm::cross(u, v));
        }
        return lum * area * 3.14159265359f;
    } else if (type == LIGHT_SPOT) {
        float outerCos = light.v.w;
        float solidAngle = 2.0f * 3.14159265359f * (1.0f - std::clamp(outerCos, -1.0f, 1.0f));
        return lum * solidAngle;
    } else { // LIGHT_DIRECTIONAL
        return lum * 3.14159265359f;
    }
}

inline void buildLightAliasTable(std::vector<LightGPU>& lights) {
    if (lights.empty()) return;
    size_t n = lights.size();
    if (n == 1) {
        float flux = calculateLightFlux(lights[0]);
        lights[0].sampling = glm::vec4(1.0f, std::bit_cast<float>(0u), 1.0f, flux);
        return;
    }

    std::vector<float> fluxes(n);
    double totalFlux = 0.0;
    for (size_t i = 0; i < n; ++i) {
        fluxes[i] = calculateLightFlux(lights[i]);
        totalFlux += fluxes[i];
    }

    if (totalFlux <= 1e-7) {
        float uniformPdf = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            lights[i].sampling = glm::vec4(1.0f, std::bit_cast<float>(static_cast<uint32_t>(i)), uniformPdf, 0.0f);
        }
        return;
    }

    std::vector<float> prob(n);
    std::vector<uint32_t> alias(n);
    std::vector<float> pdf(n);
    std::vector<float> scaledProb(n);
    std::vector<size_t> smallList;
    std::vector<size_t> largeList;
    smallList.reserve(n);
    largeList.reserve(n);

    for (size_t i = 0; i < n; ++i) {
        pdf[i] = static_cast<float>(fluxes[i] / totalFlux);
        scaledProb[i] = pdf[i] * static_cast<float>(n);
        if (scaledProb[i] < 1.0f - 1e-6f) {
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

        prob[s] = scaledProb[s];
        alias[s] = static_cast<uint32_t>(l);

        scaledProb[l] = (scaledProb[l] + scaledProb[s]) - 1.0f;
        if (scaledProb[l] < 1.0f - 1e-6f) {
            smallList.push_back(l);
        } else {
            largeList.push_back(l);
        }
    }

    while (!largeList.empty()) {
        size_t l = largeList.back();
        largeList.pop_back();
        prob[l] = 1.0f;
        alias[l] = static_cast<uint32_t>(l);
    }

    while (!smallList.empty()) {
        size_t s = smallList.back();
        smallList.pop_back();
        prob[s] = 1.0f;
        alias[s] = static_cast<uint32_t>(s);
    }

    for (size_t i = 0; i < n; ++i) {
        lights[i].sampling = glm::vec4(
            std::clamp(prob[i], 0.0f, 1.0f),
            std::bit_cast<float>(alias[i]),
            pdf[i],
            fluxes[i]
        );
    }
}

} // namespace pathways
