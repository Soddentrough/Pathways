#pragma once

#include "scene/Light.hpp"
#include <glm/glm.hpp>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdint>

namespace pathways {

// 64-byte GPU layout for hierarchical Light Tree nodes
struct alignas(16) LightTreeNodeGPU {
    glm::vec4 bboxMin;   // xyz: bboxMin, w: radiant flux (Phi)
    glm::vec4 bboxMax;   // xyz: bboxMax, w: coneAngleCos
    glm::vec4 coneAxis;  // xyz: cone center axis, w: padding
    glm::uvec4 children; // x: leftChild (or lightIdx if leaf), y: rightChild (0xFFFFFFFF if leaf), z: isLeaf (1 or 0), w: padding
};

namespace detail {

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

    if (type == LIGHT_AREA_QUAD) {
        glm::vec3 u(light.u);
        glm::vec3 v(light.v);
        glm::vec3 p0 = pos;
        glm::vec3 p1 = pos + u;
        glm::vec3 p2 = pos + v;
        glm::vec3 p3 = pos + u + v;

        info.bboxMin = glm::min(glm::min(p0, p1), glm::min(p2, p3)) - glm::vec3(0.005f);
        info.bboxMax = glm::max(glm::max(p0, p1), glm::max(p2, p3)) + glm::vec3(0.005f);
        info.centroid = 0.25f * (p0 + p1 + p2 + p3);
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

inline uint32_t buildSubtree(
    std::vector<LightPrimInfo>& prims,
    size_t start,
    size_t end,
    std::vector<LightTreeNodeGPU>& nodes)
{
    uint32_t nodeIdx = static_cast<uint32_t>(nodes.size());
    nodes.emplace_back();

    size_t count = end - start;
    if (count == 1) {
        const auto& p = prims[start];
        LightTreeNodeGPU leaf{};
        leaf.bboxMin = glm::vec4(p.bboxMin, p.flux);
        leaf.bboxMax = glm::vec4(p.bboxMax, p.coneAngleCos);
        leaf.coneAxis = glm::vec4(p.coneAxis, 0.0f);
        leaf.children = glm::uvec4(p.lightIdx, 0xFFFFFFFFu, 1u, 0u);
        nodes[nodeIdx] = leaf;
        return nodeIdx;
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

    uint32_t leftChild = buildSubtree(prims, start, mid, nodes);
    uint32_t rightChild = buildSubtree(prims, mid, end, nodes);

    const auto& left = nodes[leftChild];
    const auto& right = nodes[rightChild];

    glm::vec3 bMin = glm::min(glm::vec3(left.bboxMin), glm::vec3(right.bboxMin));
    glm::vec3 bMax = glm::max(glm::vec3(left.bboxMax), glm::vec3(right.bboxMax));
    float totalFlux = left.bboxMin.w + right.bboxMin.w;

    glm::vec3 mergedAxis;
    float mergedCos;
    combineBoundingCones(
        glm::vec3(left.coneAxis), left.bboxMax.w,
        glm::vec3(right.coneAxis), right.bboxMax.w,
        mergedAxis, mergedCos
    );

    LightTreeNodeGPU internalNode{};
    internalNode.bboxMin = glm::vec4(bMin, totalFlux);
    internalNode.bboxMax = glm::vec4(bMax, mergedCos);
    internalNode.coneAxis = glm::vec4(mergedAxis, 0.0f);
    internalNode.children = glm::uvec4(leftChild, rightChild, 0u, 0u);

    nodes[nodeIdx] = internalNode;
    return nodeIdx;
}

} // namespace detail

inline void buildLightTree(const std::vector<LightGPU>& lights, std::vector<LightTreeNodeGPU>& nodes) {
    nodes.clear();
    if (lights.empty()) {
        LightTreeNodeGPU dummy{};
        dummy.bboxMin = glm::vec4(-1.0f, -1.0f, -1.0f, 0.0f);
        dummy.bboxMax = glm::vec4(1.0f, 1.0f, 1.0f, -1.0f);
        dummy.coneAxis = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
        dummy.children = glm::uvec4(0u, 0xFFFFFFFFu, 1u, 0u);
        nodes.push_back(dummy);
        return;
    }

    std::vector<detail::LightPrimInfo> prims(lights.size());
    for (size_t i = 0; i < lights.size(); ++i) {
        prims[i] = detail::makePrimInfo(static_cast<uint32_t>(i), lights[i]);
    }

    nodes.reserve(lights.size() * 2);
    detail::buildSubtree(prims, 0, prims.size(), nodes);
}

} // namespace pathways
