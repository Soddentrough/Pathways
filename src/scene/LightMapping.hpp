#pragma once

// Emissive-geometry <-> light-list correspondence (plan P1 §4.1/4.2).
//
// Unbiased NEE + BSDF-sampling MIS needs, for every emissive triangle hit by
// a BSDF-sampled ray, the light-sampling pdf of that exact point. This module
// matches emissive triangles to the area lights that cover them and records:
//
//   per triangle (TriangleShadeGPU::padding0 on the GPU):
//     TRI_LIGHT_NONE       (0)          not covered by any light: the BSDF hit
//                                        is the only estimator -> MIS weight 1
//     lightIdx + 1                       covered by exactly that light -> balance
//                                        heuristic against its NEE pdf
//     TRI_LIGHT_NEE_ONLY   (0xFFFFFFFF) covered, but the light cannot use MIS
//                                        (ambiguous instancing / partial
//                                        coverage): NEE owns it -> weight 0
//
//   per light (LightGPU::pad0.x as uint bits): LIGHT_FLAG_MIS when the light's
//   whole surface is made of matched, hittable triangles. Lights without the
//   flag are NEE-only (MIS weight 1 on the NEE side), e.g. analytic quads that
//   have no geometry, spot and directional lights.
//
// The weights of the two strategies therefore always sum to exactly one for
// every point on every emitter, which is the condition for unbiased MIS.

#include "scene/Light.hpp"
#include "scene/ProceduralScene.hpp"

#include <glm/glm.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace pathways {

constexpr uint32_t TRI_LIGHT_NONE = 0u;
constexpr uint32_t TRI_LIGHT_NEE_ONLY = 0xFFFFFFFFu;
constexpr uint32_t LIGHT_FLAG_MIS = 1u;

inline uint32_t getLightFlags(const LightGPU& l) { return std::bit_cast<uint32_t>(l.pad0.x); }
inline void setLightFlags(LightGPU& l, uint32_t f) { l.pad0.x = std::bit_cast<float>(f); }

struct LightMappingStats {
    uint32_t emissiveTriangles = 0;
    uint32_t matchedTriangles = 0;
    uint32_t neeOnlyTriangles = 0;
    uint32_t misLights = 0;
    uint32_t neeOnlyAreaLights = 0;
};

namespace detail {

struct AreaLightPlane {
    uint32_t index;
    glm::vec3 p0, u, v, n;
    float area;
    float uu, uv, vv, invDet;
    glm::vec3 bmin, bmax;
};

inline bool insideParallelogram(const AreaLightPlane& L, const glm::vec3& p, float planeEps, float uvEps) {
    glm::vec3 d = p - L.p0;
    if (std::abs(glm::dot(d, L.n)) > planeEps) return false;
    float du = glm::dot(d, L.u), dv = glm::dot(d, L.v);
    float a = (L.vv * du - L.uv * dv) * L.invDet;
    float b = (L.uu * dv - L.uv * du) * L.invDet;
    return a >= -uvEps && a <= 1.0f + uvEps && b >= -uvEps && b <= 1.0f + uvEps;
}

inline bool isEmissive(const MaterialGPU& m) {
    return std::max({m.emissive.r, m.emissive.g, m.emissive.b}) > 1e-3f;
}

// The BSDF-hit estimator sees the material's emission while NEE uses the
// light's constant radiance. MIS weights only make the combination unbiased
// when both strategies integrate the same function, so textured, procedural
// or mismatching emitters must stay NEE-only.
inline bool emissionMatchesLight(const MaterialGPU& m, const LightGPU& l) {
    if (m.emissiveTex != 0u) return false;
    if ((m.type & (MATERIAL_FLAG_PROCEDURAL_HOLO | MATERIAL_FLAG_HOLO_VIDEO)) != 0u) return false;
    for (int c = 0; c < 3; ++c) {
        float a = m.emissive[c], b = l.emission[c];
        if (std::abs(a - b) > 0.01f * std::max({std::abs(a), std::abs(b), 1e-6f})) return false;
    }
    return true;
}

} // namespace detail

// triLightMap is resized to scene.triangles.size(). Lights get LIGHT_FLAG_MIS
// set/cleared. Must run after partitionSceneGeometry() (final triangle order).
inline LightMappingStats buildEmissiveLightMapping(SceneData& scene, std::vector<uint32_t>& triLightMap) {
    LightMappingStats stats;
    const size_t numTris = scene.triangles.size();
    triLightMap.assign(numTris, TRI_LIGHT_NONE);

    std::vector<detail::AreaLightPlane> planes;
    for (uint32_t i = 0; i < scene.lights.size(); ++i) {
        LightGPU& l = scene.lights[i];
        setLightFlags(l, 0u);
        if (static_cast<uint32_t>(l.position.w) != LIGHT_AREA_QUAD) continue;
        detail::AreaLightPlane P{};
        P.index = i;
        P.p0 = glm::vec3(l.position);
        P.u = glm::vec3(l.u);
        P.v = glm::vec3(l.v);
        glm::vec3 c = glm::cross(P.u, P.v);
        float len = glm::length(c);
        if (len < 1e-10f) continue;
        P.n = c / len;
        P.area = len;
        P.uu = glm::dot(P.u, P.u);
        P.uv = glm::dot(P.u, P.v);
        P.vv = glm::dot(P.v, P.v);
        float det = P.uu * P.vv - P.uv * P.uv;
        if (std::abs(det) < 1e-20f) continue;
        P.invDet = 1.0f / det;
        glm::vec3 q[4] = {P.p0, P.p0 + P.u, P.p0 + P.v, P.p0 + P.u + P.v};
        P.bmin = glm::min(glm::min(q[0], q[1]), glm::min(q[2], q[3]));
        P.bmax = glm::max(glm::max(q[0], q[1]), glm::max(q[2], q[3]));
        planes.push_back(P);
    }

    // Uniform grid over light AABBs for O(1)-ish candidate lookup.
    glm::vec3 gmin(1e30f), gmax(-1e30f);
    float meanExtent = 0.0f;
    for (const auto& P : planes) {
        gmin = glm::min(gmin, P.bmin);
        gmax = glm::max(gmax, P.bmax);
        meanExtent += glm::length(P.bmax - P.bmin);
    }
    const float cell = planes.empty() ? 1.0f : std::max(meanExtent / planes.size(), 1e-3f);
    auto cellKey = [&](const glm::vec3& p) -> uint64_t {
        glm::ivec3 c = glm::ivec3(glm::floor((p - gmin) / cell));
        return (uint64_t(uint32_t(c.x) & 0x1FFFFF)) | (uint64_t(uint32_t(c.y) & 0x1FFFFF) << 21) |
               (uint64_t(uint32_t(c.z) & 0x1FFFFF) << 42);
    };
    std::unordered_map<uint64_t, std::vector<uint32_t>> grid;
    for (uint32_t pi = 0; pi < planes.size(); ++pi) {
        const auto& P = planes[pi];
        glm::ivec3 a = glm::ivec3(glm::floor((P.bmin - gmin) / cell));
        glm::ivec3 b = glm::ivec3(glm::floor((P.bmax - gmin) / cell));
        for (int z = a.z; z <= b.z; ++z)
            for (int y = a.y; y <= b.y; ++y)
                for (int x = a.x; x <= b.x; ++x)
                    grid[cellKey(gmin + (glm::vec3(x, y, z) + 0.5f) * cell)].push_back(pi);
    }

    // Instances: (triangle range, transform, material offset). Monolithic
    // scenes behave as one identity instance over all triangles.
    struct InstView { uint32_t first, count; glm::mat4 xf; uint32_t matOffset; };
    std::vector<InstView> views;
    if (!scene.instances.empty() && !scene.blasRanges.empty()) {
        for (const auto& inst : scene.instances) {
            if (inst.blasIndex >= scene.blasRanges.size()) continue;
            const auto& r = scene.blasRanges[inst.blasIndex];
            uint32_t mo = (inst.customIndex < scene.instanceData.size()) ? scene.instanceData[inst.customIndex].materialOffset : 0u;
            views.push_back({r.firstTriangle, r.triangleCount, inst.transform, mo});
        }
    } else {
        views.push_back({0u, static_cast<uint32_t>(numTris), glm::mat4(1.0f), 0u});
    }

    constexpr uint32_t UNSEEN = 0xFFFFFFFEu;
    std::vector<uint32_t> perTri(numTris, UNSEEN);
    std::vector<double> matchedArea(scene.lights.size(), 0.0);
    std::vector<uint8_t> lightAmbiguous(scene.lights.size(), 0);
    std::vector<uint8_t> emissionMismatch(scene.lights.size(), 0);
    std::vector<uint8_t> triEmissive(numTris, 0);

    for (const auto& V : views) {
        for (uint32_t t = V.first; t < V.first + V.count && t < numTris; ++t) {
            const TriangleGPU& tri = scene.triangles[t];
            uint32_t mid = tri.materialId + V.matOffset;
            if (mid >= scene.materials.size() || !detail::isEmissive(scene.materials[mid])) continue;
            triEmissive[t] = 1;
            glm::vec3 w0 = glm::vec3(V.xf * glm::vec4(glm::vec3(tri.v0.position), 1.0f));
            glm::vec3 w1 = glm::vec3(V.xf * glm::vec4(glm::vec3(tri.v1.position), 1.0f));
            glm::vec3 w2 = glm::vec3(V.xf * glm::vec4(glm::vec3(tri.v2.position), 1.0f));
            glm::vec3 cr = glm::cross(w1 - w0, w2 - w0);
            float twiceArea = glm::length(cr);
            uint32_t match = TRI_LIGHT_NONE;
            if (twiceArea > 1e-12f && !planes.empty()) {
                glm::vec3 tn = cr / twiceArea;
                glm::vec3 cen = (w0 + w1 + w2) / 3.0f;
                float scale = std::max({glm::length(w1 - w0), glm::length(w2 - w0), 1e-4f});
                auto it = grid.find(cellKey(cen));
                if (it != grid.end()) {
                    for (uint32_t pi : it->second) {
                        const auto& P = planes[pi];
                        if (std::abs(glm::dot(tn, P.n)) < 0.999f) continue;
                        float eps = 1e-3f * std::max(scale, std::sqrt(P.area));
                        if (detail::insideParallelogram(P, w0, eps, 1e-3f) &&
                            detail::insideParallelogram(P, w1, eps, 1e-3f) &&
                            detail::insideParallelogram(P, w2, eps, 1e-3f)) {
                            match = P.index + 1u;
                            matchedArea[P.index] += 0.5 * twiceArea;
                            if (!detail::emissionMatchesLight(scene.materials[mid], scene.lights[P.index])) {
                                emissionMismatch[P.index] = 1;
                            }
                            break;
                        }
                    }
                }
            }
            if (perTri[t] == UNSEEN) {
                perTri[t] = match;
            } else if (perTri[t] != match) {
                // Same BLAS triangle maps to different lights in different instances.
                if (perTri[t] != TRI_LIGHT_NONE && perTri[t] != TRI_LIGHT_NEE_ONLY) lightAmbiguous[perTri[t] - 1u] = 1;
                if (match != TRI_LIGHT_NONE) lightAmbiguous[match - 1u] = 1;
                perTri[t] = TRI_LIGHT_NEE_ONLY;
            }
        }
    }

    // A light can use MIS only if hittable geometry tiles its entire surface;
    // otherwise NEE samples points BSDF sampling can never reach.
    std::vector<uint8_t> lightMis(scene.lights.size(), 0);
    for (const auto& P : planes) {
        double rel = matchedArea[P.index] / std::max<double>(P.area, 1e-12);
        bool full = (rel > 0.995 && rel < 1.005);
        if (full && !lightAmbiguous[P.index] && !emissionMismatch[P.index]) {
            lightMis[P.index] = 1;
            setLightFlags(scene.lights[P.index], LIGHT_FLAG_MIS);
            stats.misLights++;
        } else {
            stats.neeOnlyAreaLights++;
        }
    }

    for (size_t t = 0; t < numTris; ++t) {
        if (!triEmissive[t]) continue;
        stats.emissiveTriangles++;
        uint32_t m = perTri[t];
        if (m == UNSEEN) m = TRI_LIGHT_NONE;
        if (m != TRI_LIGHT_NONE && m != TRI_LIGHT_NEE_ONLY && !lightMis[m - 1u]) m = TRI_LIGHT_NEE_ONLY;
        if (m == TRI_LIGHT_NEE_ONLY) stats.neeOnlyTriangles++;
        else if (m != TRI_LIGHT_NONE) stats.matchedTriangles++;
        triLightMap[t] = m;
    }
    return stats;
}

} // namespace pathways
