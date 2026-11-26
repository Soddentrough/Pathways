#pragma once

#include "scene/Light.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <bit>
#include <cstdint>

namespace pathways {

// 32-byte cache-line-paired GPU layout for hierarchical Light Tree nodes
// Sibling pairs [firstChild, firstChild + 1] are stored contiguously at even
// indices >= 2 so both children fit in one 64-byte cache line.
struct alignas(16) LightTreeNodeGPU {
    glm::vec4 centerFlux;       // xyz: bbox center, w: radiant flux (Phi)
    glm::uvec4 extentConeChild; // x: packHalf2x16(halfExt.x, halfExt.y)
                                // y: packHalf2x16(halfExt.z, coneAngleCos)
                                // z: packOct32(coneAxis)
                                // w: [31] isLeaf | [30:0] lightIdx (leaf) or even firstChild (internal)

    [[nodiscard]] bool isLeaf() const noexcept { return (extentConeChild.w & 0x80000000u) != 0u; }
    [[nodiscard]] uint32_t lightIndex() const noexcept { return extentConeChild.w & 0x7FFFFFFFu; }
    [[nodiscard]] uint32_t firstChild() const noexcept { return extentConeChild.w & 0x7FFFFFFFu; }
    [[nodiscard]] float flux() const noexcept { return centerFlux.w; }
    [[nodiscard]] glm::vec3 center() const noexcept { return glm::vec3(centerFlux); }
    [[nodiscard]] glm::vec3 halfExtent() const noexcept {
        glm::vec2 xy = glm::unpackHalf2x16(extentConeChild.x);
        glm::vec2 zc = glm::unpackHalf2x16(extentConeChild.y);
        return glm::vec3(xy.x, xy.y, zc.x);
    }
    [[nodiscard]] float coneAngleCos() const noexcept { return glm::unpackHalf2x16(extentConeChild.y).y; }
    [[nodiscard]] glm::vec3 bboxMin() const noexcept { return center() - halfExtent(); }
    [[nodiscard]] glm::vec3 bboxMax() const noexcept { return center() + halfExtent(); }
};
static_assert(sizeof(LightTreeNodeGPU) == 32, "LightTreeNodeGPU must be exactly 32 bytes");

namespace detail {

inline uint32_t packOct32Cone(glm::vec3 v) {
    float len = glm::length(v);
    if (len > 1e-6f) {
        v /= len;
    } else {
        v = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    float l1 = std::abs(v.x) + std::abs(v.y) + std::abs(v.z);
    if (l1 > 1e-6f) {
        v /= l1;
    } else {
        v = glm::vec3(0.0f, 1.0f, 0.0f);
    }
    glm::vec2 enc(v.x, v.y);
    if (v.z < 0.0f) {
        glm::vec2 s((enc.x >= 0.0f) ? 1.0f : -1.0f, (enc.y >= 0.0f) ? 1.0f : -1.0f);
        enc = (glm::vec2(1.0f) - glm::abs(glm::vec2(v.y, v.x))) * s;
    }
    return glm::packSnorm2x16(enc);
}

struct MergedBounds {
    glm::vec3 bMin = glm::vec3(0.0f);
    glm::vec3 bMax = glm::vec3(0.0f);
    float flux = 0.0f;
    glm::vec3 coneAxis = glm::vec3(0.0f, 1.0f, 0.0f);
    float coneAngleCos = -1.0f;
};

inline LightTreeNodeGPU packLightTreeNode(
    const glm::vec3& bMin,
    const glm::vec3& bMax,
    float flux,
    const glm::vec3& coneAxis,
    float coneCos,
    bool isLeaf,
    uint32_t childOrLight,
    glm::vec3* outActualMin = nullptr,
    glm::vec3* outActualMax = nullptr)
{
    glm::vec3 center = 0.5f * (bMin + bMax);
    glm::vec3 rawHalf = glm::max(bMax - center, center - bMin);
    glm::vec3 paddedHalf = glm::clamp(rawHalf * 1.002f + glm::vec3(1e-4f), glm::vec3(0.0f), glm::vec3(65504.0f));
    float clampedCos = std::clamp(coneCos, -1.0f, 1.0f);

    LightTreeNodeGPU node{};
    node.centerFlux = glm::vec4(center, flux);
    node.extentConeChild.x = glm::packHalf2x16(glm::vec2(paddedHalf.x, paddedHalf.y));
    node.extentConeChild.y = glm::packHalf2x16(glm::vec2(paddedHalf.z, clampedCos));
    node.extentConeChild.z = packOct32Cone(coneAxis);
    node.extentConeChild.w = (isLeaf ? 0x80000000u : 0u) | (childOrLight & 0x7FFFFFFFu);

    if (outActualMin || outActualMax) {
        glm::vec3 actualHalf = node.halfExtent();
        if (outActualMin) *outActualMin = center - actualHalf;
        if (outActualMax) *outActualMax = center + actualHalf;
    }
    return node;
}

struct LightPrimInfo {
    uint32_t lightIdx = 0;
    glm::vec3 bboxMin = glm::vec3(0.0f);
    glm::vec3 bboxMax = glm::vec3(0.0f);
    glm::vec3 centroid = glm::vec3(0.0f);
    glm::vec3 coneAxis = glm::vec3(0.0f, 1.0f, 0.0f);
    float coneAngleCos = 1.0f; // cosine of half-angle
    float flux = 0.0f;
};

inline LightPrimInfo makePrimInfo(uint32_t idx, const LightGPU& light) {
    LightPrimInfo info;
    info.lightIdx = idx;
    info.flux = calculateLightFlux(light);

    uint32_t type = static_cast<uint32_t>(light.position.w);
    glm::vec3 pos(light.position);

    if (type == LIGHT_AREA_QUAD || type == LIGHT_AREA_TRIANGLE) {
        glm::vec3 u(light.u);
        glm::vec3 v(light.v);
        glm::vec3 p0 = pos;
        glm::vec3 p1 = pos + u;
        glm::vec3 p2 = pos + v;
        if (type == LIGHT_AREA_TRIANGLE) {
            info.bboxMin = glm::min(glm::min(p0, p1), p2) - glm::vec3(0.005f);
            info.bboxMax = glm::max(glm::max(p0, p1), p2) + glm::vec3(0.005f);
            info.centroid = (p0 + p1 + p2) / 3.0f;
        } else {
            glm::vec3 p3 = pos + u + v;
            info.bboxMin = glm::min(glm::min(p0, p1), glm::min(p2, p3)) - glm::vec3(0.005f);
            info.bboxMax = glm::max(glm::max(p0, p1), glm::max(p2, p3)) + glm::vec3(0.005f);
            info.centroid = 0.25f * (p0 + p1 + p2 + p3);
        }
        info.coneAxis = glm::normalize(glm::vec3(light.normal));
        info.coneAngleCos = 0.0f; // Front hemisphere emission (cos 90 deg = 0.0)
    } else if (type == LIGHT_SPOT) {
        info.bboxMin = pos - glm::vec3(0.05f);
        info.bboxMax = pos + glm::vec3(0.05f);
        info.centroid = pos;
        info.coneAxis = glm::normalize(glm::vec3(light.normal));
        info.coneAngleCos = std::clamp(light.v.w, -1.0f, 1.0f); // Outer cone angle cos
    } else { // LIGHT_DIRECTIONAL
        info.bboxMin = pos - glm::vec3(1000.0f);
        info.bboxMax = pos + glm::vec3(1000.0f);
        info.centroid = pos;
        info.coneAxis = glm::normalize(glm::vec3(light.normal));
        info.coneAngleCos = 0.999f; // Directional beam
    }

    return info;
}

inline void combineBoundingCones(
    const glm::vec3& axisA, float cosA,
    const glm::vec3& axisB, float cosB,
    glm::vec3& outAxis, float& outCos)
{
    if (cosA <= -0.999f || cosB <= -0.999f) {
        outAxis = glm::vec3(0.0f, 1.0f, 0.0f);
        outCos = -1.0f; // Omnidirectional
        return;
    }

    float dotVal = std::clamp(glm::dot(axisA, axisB), -1.0f, 1.0f);
    float angleAB = std::acos(dotVal);
    float thetaA = std::acos(std::clamp(cosA, -1.0f, 1.0f));
    float thetaB = std::acos(std::clamp(cosB, -1.0f, 1.0f));

    if (angleAB + thetaB <= thetaA) {
        outAxis = axisA;
        outCos = cosA;
        return;
    }
    if (angleAB + thetaA <= thetaB) {
        outAxis = axisB;
        outCos = cosB;
        return;
    }

    float newTheta = 0.5f * (thetaA + thetaB + angleAB);
    if (newTheta >= 3.14159265f) {
        outAxis = glm::vec3(0.0f, 1.0f, 0.0f);
        outCos = -1.0f;
        return;
    }

    float rotAngle = newTheta - thetaA;
    glm::vec3 ortho = axisB - dotVal * axisA;
    float orthoLen = glm::length(ortho);
    if (orthoLen > 1e-5f) {
        outAxis = glm::normalize(axisA * std::cos(rotAngle) + (ortho / orthoLen) * std::sin(rotAngle));
    } else {
        outAxis = axisA;
    }
    outCos = std::cos(newTheta);
}

inline MergedBounds buildSubtree(
    std::vector<LightPrimInfo>& prims,
    size_t start,
    size_t end,
    std::vector<LightTreeNodeGPU>& nodes,
    uint32_t dstIdx)
{
    size_t count = end - start;
    if (count == 1) {
        const auto& p = prims[start];
        MergedBounds out{};
        out.flux = p.flux;
        out.coneAxis = p.coneAxis;
        out.coneAngleCos = p.coneAngleCos;
        nodes[dstIdx] = packLightTreeNode(
            p.bboxMin, p.bboxMax, p.flux, p.coneAxis, p.coneAngleCos,
            true, p.lightIdx, &out.bMin, &out.bMax
        );
        return out;
    }

    // Determine bounding box of centroids
    glm::vec3 cMin = prims[start].centroid;
    glm::vec3 cMax = prims[start].centroid;
    for (size_t i = start + 1; i < end; ++i) {
        cMin = glm::min(cMin, prims[i].centroid);
        cMax = glm::max(cMax, prims[i].centroid);
    }

    glm::vec3 extent = cMax - cMin;
    int axis = 0;
    if (extent.y > extent.x && extent.y > extent.z) axis = 1;
    else if (extent.z > extent.x && extent.z > extent.y) axis = 2;

    size_t mid = start + count / 2;
    std::nth_element(
        prims.begin() + start,
        prims.begin() + mid,
        prims.begin() + end,
        [axis](const LightPrimInfo& a, const LightPrimInfo& b) {
            return a.centroid[axis] < b.centroid[axis];
        }
    );

    uint32_t pairIdx = static_cast<uint32_t>(nodes.size());
    nodes.resize(pairIdx + 2);

    MergedBounds left = buildSubtree(prims, start, mid, nodes, pairIdx);
    MergedBounds right = buildSubtree(prims, mid, end, nodes, pairIdx + 1u);

    glm::vec3 bMin = glm::min(left.bMin, right.bMin);
    glm::vec3 bMax = glm::max(left.bMax, right.bMax);
    float totalFlux = left.flux + right.flux;

    glm::vec3 mergedAxis;
    float mergedCos;
    combineBoundingCones(
        left.coneAxis, left.coneAngleCos,
        right.coneAxis, right.coneAngleCos,
        mergedAxis, mergedCos
    );

    MergedBounds out{};
    out.flux = totalFlux;
    out.coneAxis = mergedAxis;
    out.coneAngleCos = mergedCos;
    nodes[dstIdx] = packLightTreeNode(
        bMin, bMax, totalFlux, mergedAxis, mergedCos,
        false, pairIdx, &out.bMin, &out.bMax
    );
    return out;
}

} // namespace detail

inline void buildLightTree(const std::vector<LightGPU>& lights, std::vector<LightTreeNodeGPU>& nodes) {
    nodes.clear();
    if (lights.empty()) {
        nodes.push_back(detail::packLightTreeNode(
            glm::vec3(-1.0f), glm::vec3(1.0f), 0.0f, glm::vec3(0.0f, 1.0f, 0.0f), -1.0f,
            true, 0u
        ));
        return;
    }

    std::vector<detail::LightPrimInfo> prims(lights.size());
    for (size_t i = 0; i < lights.size(); ++i) {
        prims[i] = detail::makePrimInfo(static_cast<uint32_t>(i), lights[i]);
    }

    if (lights.size() == 1) {
        nodes.resize(1);
        detail::buildSubtree(prims, 0, 1, nodes, 0u);
        return;
    }

    // Reserve 2*N nodes: root at 0, padding slot at 1, and every sibling pair [pairIdx, pairIdx+1]
    // starting at an even index >= 2 (64-byte cache-line aligned).
    nodes.reserve(lights.size() * 2 + 2);
    nodes.resize(2);
    detail::buildSubtree(prims, 0, prims.size(), nodes, 0u);
}

// Writes each light's root-to-leaf descent into LightGPU::pad0:
//   y: uintBitsToFloat(path bits, bit d = 1 -> right child at depth d)
//   z: uintBitsToFloat(depth)
// Shaders replay this path to evaluate the tree's selection pdf for a known
// light index (MIS weight of BSDF-sampled emitter hits). Depth > 32 cannot be
// encoded; such lights get depth 0xFFFFFFFF and the shader treats the tree pdf
// as unavailable (the engine then falls back to alias sampling).
inline bool computeLightTreePaths(const std::vector<LightTreeNodeGPU>& nodes, std::vector<LightGPU>& lights) {
    if (nodes.empty() || lights.empty()) return true;
    bool ok = true;
    struct Item { uint32_t node, bits, depth; };
    std::vector<Item> stack{{0u, 0u, 0u}};
    while (!stack.empty()) {
        Item it = stack.back();
        stack.pop_back();
        const auto& n = nodes[it.node];
        if (n.isLeaf()) {
            uint32_t li = n.lightIndex();
            if (li < lights.size()) {
                lights[li].pad0.y = std::bit_cast<float>(it.bits);
                lights[li].pad0.z = std::bit_cast<float>(it.depth <= 32u ? it.depth : 0xFFFFFFFFu);
                if (it.depth > 32u) ok = false;
            }
            continue;
        }
        uint32_t d = it.depth;
        uint32_t rightBit = (d < 32u) ? (1u << d) : 0u;
        uint32_t first = n.firstChild();
        stack.push_back({first, it.bits, d + 1u});
        stack.push_back({first + 1u, it.bits | rightBit, d + 1u});
    }
    return ok;
}

} // namespace pathways
