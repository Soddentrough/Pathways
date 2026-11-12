#pragma once

// Emissive instanced geometry as NEE area lights (one light per emitting quad per instance).
//
// Scenes built from BLAS prototypes + instance transforms (procedural cities, USD point
// instancers) carry most of their emitted power in emissive prototype triangles. Without this
// pass that power is reachable only by BSDF sampling (MIS weight 1), which at 1 spp is a
// low-probability, high-value event ("mostly black, rare bright speckle"). This pass pairs the
// emissive triangles of every prototype into parallelogram quads and appends one world-space
// LIGHT_AREA_QUAD per quad per instance, so light sampling can find them, and tags the
// triangles so BSDF hits resolve the matching light for balance-heuristic MIS:
//
//   TriangleShadeGPU::padding0 = TRI_LIGHT_INSTANCED_BIT | quadIndexInPrototype
//   InstanceGPU::flags          = firstLightOfInstance + 1      (0 = instance has no such lights)
//   light index                 = firstLightOfInstance + quadIndexInPrototype
//
// Consistency rules (see LightMapping.hpp / shaders/compute/mis_common.glsl):
//  * One-sided: a light emits from the quad's geometric front (winding normal). BSDF hits on the
//    back side get MIS weight 1 because NEE never samples that side, so the two-sided emission of
//    emissive surfaces is still integrated exactly once. (Two-sided proxies would spend half of
//    the NEE samples on back faces that typically face into solid geometry.)
//  * Only constant-emission materials: textured / procedural / video emitters would need the
//    emission evaluated at the sampled point; they stay BSDF-only (unbiased, noisier).
//  * Only triangles that pair into exact parallelograms become lights; an unpaired triangle would
//    need a triangle light type (a parallelogram would emit from area with no geometry). Unpaired
//    triangles stay BSDF-only.
//  * Selection is per prototype, never truncated per instance: a prototype triangle is covered in
//    every instance or in none, which is what makes the per-triangle tag valid.
//  * Lights carry LightGPU::pad0.w = 1 so buildEmissiveLightMapping() flags them MIS-capable
//    without plane matching.

#include "scene/Light.hpp"
#include "scene/ProceduralScene.hpp"

#include <cmath>
#include <cstdint>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pathways {

constexpr float kEmissiveGeometryLightMarker = 1.0f; // LightGPU::pad0.w

inline bool isEmissiveGeometryLight(const LightGPU& l) { return l.pad0.w > 0.5f; }

struct EmissiveGeometryLightStats {
    uint32_t protoQuads = 0;         // emitting quads over all prototypes (before instancing)
    uint32_t addedLights = 0;        // lights appended (one per quad per instance)
    uint32_t unpairedTriangles = 0;  // emissive prototype triangles left BSDF-only (no parallelogram partner)
    uint32_t textured = 0;           // emissive prototype triangles left BSDF-only (non-constant emission)
    const char* skipReason = nullptr; // non-null: feature not applied to this scene
};

namespace detail {

struct ProtoQuad {
    uint32_t tris[2] = {0xFFFFFFFFu, 0xFFFFFFFFu}; // global triangle indices covered by this quad/triangle
    glm::vec3 corner, u, v;                        // prototype space, cross(u, v) = winding normal
    glm::vec3 emission;
    bool isTriangle = false;
};

inline bool nearPts(const glm::vec3& a, const glm::vec3& b) { return glm::dot(a - b, a - b) < 1e-6f; }

inline std::vector<ProtoQuad> buildProtoQuads(const SceneData& scene, const BlasGeometryRange& range,
                                              uint32_t matOffset, EmissiveGeometryLightStats& stats) {
    struct Tri { glm::vec3 p[3]; glm::vec3 n; glm::vec3 em; uint32_t mat; uint32_t idx; bool paired = false; };
    std::vector<Tri> tris;
    for (uint32_t t = range.firstTriangle; t < range.firstTriangle + range.triangleCount && t < scene.triangles.size(); ++t) {
        const TriangleGPU& tr = scene.triangles[t];
        uint32_t mid = tr.materialId + matOffset;
        if (mid >= scene.materials.size()) continue;
        const MaterialGPU& m = scene.materials[mid];
        if (std::max({m.emissive.r, m.emissive.g, m.emissive.b}) <= 1e-3f) continue;
        if (m.emissiveTex != 0u || (m.type & (MATERIAL_FLAG_PROCEDURAL_HOLO | MATERIAL_FLAG_HOLO_VIDEO)) != 0u) {
            stats.textured++;
            continue;
        }
        Tri x;
        x.p[0] = glm::vec3(tr.v0.position);
        x.p[1] = glm::vec3(tr.v1.position);
        x.p[2] = glm::vec3(tr.v2.position);
        glm::vec3 c = glm::cross(x.p[1] - x.p[0], x.p[2] - x.p[0]);
        float len = glm::length(c);
        if (len < 1e-10f) continue;
        x.n = c / len;
        x.em = glm::vec3(m.emissive);
        x.mat = mid;
        x.idx = t;
        tris.push_back(x);
    }

    // Spatial hash on vertex positions keeps pairing near-linear for dense emissive prototypes.
    auto key = [](const glm::vec3& p) -> uint64_t {
        glm::ivec3 q = glm::ivec3(glm::round(p * 1000.0f));
        return (uint64_t(uint32_t(q.x) & 0x1FFFFF)) | (uint64_t(uint32_t(q.y) & 0x1FFFFF) << 21) |
               (uint64_t(uint32_t(q.z) & 0x1FFFFF) << 42);
    };
    std::unordered_map<uint64_t, std::vector<uint32_t>> byVertex;
    for (uint32_t i = 0; i < tris.size(); ++i)
        for (const auto& p : tris[i].p) byVertex[key(p)].push_back(i);

    std::vector<ProtoQuad> quads;
    for (uint32_t i = 0; i < tris.size(); ++i) {
        if (tris[i].paired) continue;
        bool done = false;
        for (const auto& pv : tris[i].p) {
            auto it = byVertex.find(key(pv));
            if (it == byVertex.end()) continue;
            for (uint32_t j : it->second) {
                if (j == i || tris[j].paired || tris[i].mat != tris[j].mat) continue;
                if (glm::dot(tris[i].n, tris[j].n) < 0.999f) continue;
                glm::vec3 shared[3], un1[3], un2[3];
                int ns = 0, n1 = 0, n2 = 0;
                for (const auto& a : tris[i].p) {
                    bool hit = false;
                    for (const auto& b : tris[j].p) if (nearPts(a, b)) { hit = true; break; }
                    if (hit) shared[ns++] = a; else un1[n1++] = a;
                }
                if (ns != 2 || n1 != 1) continue;
                for (const auto& b : tris[j].p) {
                    if (!nearPts(b, shared[0]) && !nearPts(b, shared[1])) un2[n2++] = b;
                }
                if (n2 != 1) continue;
                glm::vec3 e0 = shared[0], e1 = shared[1], vA = un1[0], vB = un2[0];
                float diagErr = glm::length((e0 + e1) - (vA + vB)); // parallelogram: diagonals bisect
                float scale = std::max({glm::length(vA - e0), glm::length(vB - e0), glm::length(e1 - e0), 1e-4f});
                if (diagErr / scale > 1e-3f) continue;
                glm::vec3 u = vA - e0, v = vB - e0;
                if (glm::length(glm::cross(u, v)) < 1e-10f) continue;
                if (glm::dot(glm::cross(u, v), tris[i].n) < 0.0f) std::swap(u, v);
                quads.push_back({{tris[i].idx, tris[j].idx}, e0, u, v, tris[i].em});
                tris[i].paired = tris[j].paired = true;
                done = true;
                break;
            }
            if (done) break;
        }
        if (!done) {
            glm::vec3 e0 = tris[i].p[0];
            glm::vec3 u = tris[i].p[1] - e0;
            glm::vec3 v = tris[i].p[2] - e0;
            glm::vec3 cr = glm::cross(u, v);
            if (glm::length(cr) >= 1e-10f) {
                if (glm::dot(cr, tris[i].n) < 0.0f) std::swap(u, v);
                quads.push_back({{tris[i].idx, 0xFFFFFFFFu}, e0, u, v, tris[i].em, true});
            } else {
                stats.unpairedTriangles++;
            }
        }
    }
    return quads;
}

} // namespace detail

// Appends one light per emitting quad per instance. Must run after partitionSceneGeometry()
// (final triangle order) and before buildEmissiveLightMapping() / the light tree build.
inline EmissiveGeometryLightStats appendEmissiveGeometryLights(SceneData& scene, size_t maxLights = 4000000) {
    EmissiveGeometryLightStats stats;
    if (scene.instances.empty() || scene.blasRanges.empty()) {
        stats.skipReason = "monolithic scene (loader / LightMapping path)";
        return stats;
    }
    if (scene.isMineshaftCoaster || !scene.animatedInstances.empty()) {
        stats.skipReason = "animated instances (world-space lights would go stale)";
        return stats;
    }

    // The per-triangle quad tag is prototype-level: every instance of a prototype must expose the
    // same material offset.
    std::unordered_map<uint32_t, uint32_t> protoMatOffset;
    for (const auto& inst : scene.instances) {
        if (inst.blasIndex >= scene.blasRanges.size()) continue;
        uint32_t mo = (inst.customIndex < scene.instanceData.size()) ? scene.instanceData[inst.customIndex].materialOffset : 0u;
        auto it = protoMatOffset.emplace(inst.blasIndex, mo);
        if (!it.second && it.first->second != mo) {
            stats.skipReason = "prototype instanced with differing material offsets";
            return stats;
        }
    }

    std::unordered_map<uint32_t, std::vector<detail::ProtoQuad>> protoQuads;
    std::vector<uint32_t> triQuad(scene.triangles.size(), 0xFFFFFFFFu);
    for (const auto& [blas, mo] : protoMatOffset) {
        auto quads = detail::buildProtoQuads(scene, scene.blasRanges[blas], mo, stats);
        for (uint32_t qi = 0; qi < quads.size(); ++qi)
            for (uint32_t t : quads[qi].tris)
                if (t < triQuad.size()) triQuad[t] = qi;
        stats.protoQuads += static_cast<uint32_t>(quads.size());
        protoQuads.emplace(blas, std::move(quads));
    }

    size_t total = scene.lights.size();
    for (const auto& inst : scene.instances) {
        auto it = protoQuads.find(inst.blasIndex);
        if (it != protoQuads.end()) total += it->second.size();
    }
    if (total > maxLights) {
        stats.skipReason = "light budget exceeded";
        return stats;
    }

    // Refuse scenes whose explicit area lights already cover emissive geometry: adding geometry
    // lights there would count that emission twice (fix the scene instead, as done for Cyber City).
    {
        struct Plane { glm::vec3 p0, u, v, n; float uu, uv, vv, invDet, slack; };
        std::vector<Plane> planes;
        glm::vec3 gmin(1e30f), gmax(-1e30f);
        float meanExtent = 0.0f;
        for (const auto& l : scene.lights) {
            if (static_cast<uint32_t>(l.position.w) != LIGHT_AREA_QUAD || isEmissiveGeometryLight(l)) continue;
            Plane P{};
            P.p0 = glm::vec3(l.position); P.u = glm::vec3(l.u); P.v = glm::vec3(l.v);
            glm::vec3 c = glm::cross(P.u, P.v);
            float len = glm::length(c);
            if (len < 1e-10f) continue;
            P.n = c / len;
            P.uu = glm::dot(P.u, P.u); P.uv = glm::dot(P.u, P.v); P.vv = glm::dot(P.v, P.v);
            float det = P.uu * P.vv - P.uv * P.uv;
            if (std::abs(det) < 1e-20f) continue;
            P.invDet = 1.0f / det;
            P.slack = 0.05f * std::sqrt(len) + 0.05f; // proxies are often offset a few cm from the surface
            for (glm::vec3 q : {P.p0, P.p0 + P.u, P.p0 + P.v, P.p0 + P.u + P.v}) { gmin = glm::min(gmin, q); gmax = glm::max(gmax, q); }
            meanExtent += std::sqrt(len);
            planes.push_back(P);
        }
        if (!planes.empty()) {
            const float cell = std::max(meanExtent / planes.size(), 0.25f);
            auto cellOf = [&](const glm::vec3& p) { return glm::ivec3(glm::floor((p - gmin) / cell)); };
            auto cellKey = [](glm::ivec3 c) -> uint64_t {
                return (uint64_t(uint32_t(c.x) & 0x1FFFFF)) | (uint64_t(uint32_t(c.y) & 0x1FFFFF) << 21) |
                       (uint64_t(uint32_t(c.z) & 0x1FFFFF) << 42);
            };
            std::unordered_map<uint64_t, std::vector<uint32_t>> grid;
            for (uint32_t pi = 0; pi < planes.size(); ++pi) {
                const Plane& P = planes[pi];
                glm::vec3 lo(1e30f), hi(-1e30f);
                for (glm::vec3 q : {P.p0, P.p0 + P.u, P.p0 + P.v, P.p0 + P.u + P.v}) { lo = glm::min(lo, q); hi = glm::max(hi, q); }
                glm::ivec3 a = cellOf(lo - glm::vec3(P.slack)), b = cellOf(hi + glm::vec3(P.slack));
                for (int z = a.z; z <= b.z; ++z) for (int y = a.y; y <= b.y; ++y) for (int x = a.x; x <= b.x; ++x)
                    grid[cellKey({x, y, z})].push_back(pi);
            }
            for (const auto& inst : scene.instances) {
                auto it = protoQuads.find(inst.blasIndex);
                if (it == protoQuads.end()) continue;
                for (const auto& q : it->second) {
                    glm::vec3 c = glm::vec3(inst.transform * glm::vec4(q.corner + 0.5f * (q.u + q.v), 1.0f));
                    auto g = grid.find(cellKey(cellOf(c)));
                    if (g == grid.end()) continue;
                    for (uint32_t pi : g->second) {
                        const Plane& P = planes[pi];
                        glm::vec3 d = c - P.p0;
                        if (std::abs(glm::dot(d, P.n)) > P.slack) continue;
                        float du = glm::dot(d, P.u), dv = glm::dot(d, P.v);
                        float s = (P.vv * du - P.uv * dv) * P.invDet, t = (P.uu * dv - P.uv * du) * P.invDet;
                        if (s >= -0.01f && s <= 1.01f && t >= -0.01f && t <= 1.01f) {
                            stats.skipReason = "explicit area lights duplicate emissive geometry (would double count)";
                            return stats;
                        }
                    }
                }
            }
        }
    }

    std::vector<LightGPU> added;
    added.reserve(total - scene.lights.size());
    std::vector<uint32_t> instanceBase(scene.instances.size(), 0u);
    for (size_t ii = 0; ii < scene.instances.size(); ++ii) {
        const auto& inst = scene.instances[ii];
        auto it = protoQuads.find(inst.blasIndex);
        if (it == protoQuads.end() || it->second.empty()) continue;
        instanceBase[ii] = static_cast<uint32_t>(scene.lights.size() + added.size()) + 1u;
        for (const auto& q : it->second) {
            LightGPU l{};
            l.pad0.w = kEmissiveGeometryLightMarker;
            glm::vec3 c0 = glm::vec3(inst.transform * glm::vec4(q.corner, 1.0f));
            glm::vec3 u = glm::vec3(inst.transform * glm::vec4(q.u, 0.0f));
            glm::vec3 v = glm::vec3(inst.transform * glm::vec4(q.v, 0.0f));
            glm::vec3 cr = glm::cross(u, v);
            float len = glm::length(cr);
            float area = q.isTriangle ? (0.5f * len) : len;
            l.position = glm::vec4(c0, q.isTriangle ? LIGHT_AREA_TRIANGLE : LIGHT_AREA_QUAD);
            if (len < 1e-10f) {
                // Degenerate after transform: zero-flux placeholder keeps the quad indexing dense.
                l.u = glm::vec4(1e-3f, 0.0f, 0.0f, 0.0f);
                l.v = glm::vec4(0.0f, 1e-3f, 0.0f, 0.0f);
                l.normal = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
                l.emission = glm::vec4(0.0f, 0.0f, 0.0f, 1e-6f);
            } else {
                l.u = glm::vec4(u, 0.0f);
                l.v = glm::vec4(v, 0.0f);
                l.normal = glm::vec4(cr / len, 0.0f);
                l.emission = glm::vec4(q.emission, area);
            }
            added.push_back(l);
        }
    }

    stats.addedLights = static_cast<uint32_t>(added.size());
    scene.lights.insert(scene.lights.end(), added.begin(), added.end());
    for (size_t ii = 0; ii < scene.instances.size(); ++ii) {
        const uint32_t cidx = scene.instances[ii].customIndex;
        if (cidx < scene.instanceData.size()) scene.instanceData[cidx].flags = instanceBase[ii];
    }
    scene.instancedEmissiveTriQuad = std::move(triQuad);
    return stats;
}

} // namespace pathways
