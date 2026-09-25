#include "scene/ProceduralScene.hpp"
#include <fstream>
#include <cmath>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>

namespace pathways {

static void addQuad(std::vector<TriangleGPU>& triangles,
                    glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3,
                    glm::vec3 normal, uint32_t matId,
                    glm::vec2 uv0 = glm::vec2(0.0f, 1.0f),
                    glm::vec2 uv1 = glm::vec2(1.0f, 1.0f),
                    glm::vec2 uv2 = glm::vec2(1.0f, 0.0f),
                    glm::vec2 uv3 = glm::vec2(0.0f, 0.0f)) {
    glm::vec3 up = std::abs(normal.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 tanDir = glm::normalize(glm::cross(up, normal));
    glm::vec4 tangentVec = glm::vec4(tanDir, 1.0f);

    TriangleGPU t1{};
    t1.v0.position = glm::vec4(p0, uv0.x);
    t1.v0.normal = glm::vec4(normal, uv0.y);
    t1.v0.tangent = tangentVec;
    t1.v1.position = glm::vec4(p1, uv1.x);
    t1.v1.normal = glm::vec4(normal, uv1.y);
    t1.v1.tangent = tangentVec;
    t1.v2.position = glm::vec4(p2, uv2.x);
    t1.v2.normal = glm::vec4(normal, uv2.y);
    t1.v2.tangent = tangentVec;
    t1.materialId = matId;

    TriangleGPU t2{};
    t2.v0.position = glm::vec4(p0, uv0.x);
    t2.v0.normal = glm::vec4(normal, uv0.y);
    t2.v0.tangent = tangentVec;
    t2.v1.position = glm::vec4(p2, uv2.x);
    t2.v1.normal = glm::vec4(normal, uv2.y);
    t2.v1.tangent = tangentVec;
    t2.v2.position = glm::vec4(p3, uv3.x);
    t2.v2.normal = glm::vec4(normal, uv3.y);
    t2.v2.tangent = tangentVec;
    t2.materialId = matId;

    triangles.push_back(t1);
    triangles.push_back(t2);
}

static void addBox(std::vector<TriangleGPU>& triangles,
                   glm::vec3 center, glm::vec3 size, float rotationY, uint32_t matId) {
    float rad = glm::radians(rotationY);
    float cosR = std::cos(rad);
    float sinR = std::sin(rad);

    auto rot = [cosR, sinR](glm::vec3 p) -> glm::vec3 {
        return glm::vec3(p.x * cosR - p.z * sinR, p.y, p.x * sinR + p.z * cosR);
    };

    glm::vec3 h = size * 0.5f;
    glm::vec3 p[8] = {
        center + rot(glm::vec3(-h.x, -h.y, -h.z)),
        center + rot(glm::vec3( h.x, -h.y, -h.z)),
        center + rot(glm::vec3( h.x,  h.y, -h.z)),
        center + rot(glm::vec3(-h.x,  h.y, -h.z)),
        center + rot(glm::vec3(-h.x, -h.y,  h.z)),
        center + rot(glm::vec3( h.x, -h.y,  h.z)),
        center + rot(glm::vec3( h.x,  h.y,  h.z)),
        center + rot(glm::vec3(-h.x,  h.y,  h.z))
    };

    // Front
    addQuad(triangles, p[4], p[5], p[6], p[7], rot(glm::vec3(0, 0, 1)), matId);
    // Back
    addQuad(triangles, p[1], p[0], p[3], p[2], rot(glm::vec3(0, 0, -1)), matId);
    // Top
    addQuad(triangles, p[7], p[6], p[2], p[3], rot(glm::vec3(0, 1, 0)), matId);
    // Bottom
    addQuad(triangles, p[0], p[1], p[5], p[4], rot(glm::vec3(0, -1, 0)), matId);
    // Left
    addQuad(triangles, p[0], p[4], p[7], p[3], rot(glm::vec3(-1, 0, 0)), matId);
    // Right
    addQuad(triangles, p[5], p[1], p[2], p[6], rot(glm::vec3(1, 0, 0)), matId);
}

static void addVoxelBox(std::vector<TriangleGPU>& triangles,
                        glm::vec3 center, glm::vec3 size, uint32_t matId,
                        glm::vec2 uvMin, glm::vec2 uvMax) {
    glm::vec3 h = size * 0.5f;
    float x0 = center.x - h.x;
    float x1 = center.x + h.x;
    float y0 = center.y - h.y;
    float y1 = center.y + h.y;
    float z0 = center.z - h.z;
    float z1 = center.z + h.z;

    // 1. Front Face (+Z normal) - facing front (+Z)
    // p0: bottom-left, p1: bottom-right, p2: top-right, p3: top-left
    addQuad(triangles,
            glm::vec3(x0, y0, z1),
            glm::vec3(x1, y0, z1),
            glm::vec3(x1, y1, z1),
            glm::vec3(x0, y1, z1),
            glm::vec3(0.0f, 0.0f, 1.0f), matId,
            glm::vec2(uvMin.x, uvMax.y),
            glm::vec2(uvMax.x, uvMax.y),
            glm::vec2(uvMax.x, uvMin.y),
            glm::vec2(uvMin.x, uvMin.y));

    // 2. Back Face (-Z normal) - facing back (-Z)
    addQuad(triangles,
            glm::vec3(x1, y0, z0),
            glm::vec3(x0, y0, z0),
            glm::vec3(x0, y1, z0),
            glm::vec3(x1, y1, z0),
            glm::vec3(0.0f, 0.0f, -1.0f), matId,
            glm::vec2(uvMax.x, uvMax.y),
            glm::vec2(uvMin.x, uvMax.y),
            glm::vec2(uvMin.x, uvMin.y),
            glm::vec2(uvMax.x, uvMin.y));

    // 3. Right Face (+X normal)
    addQuad(triangles,
            glm::vec3(x1, y0, z1),
            glm::vec3(x1, y0, z0),
            glm::vec3(x1, y1, z0),
            glm::vec3(x1, y1, z1),
            glm::vec3(1.0f, 0.0f, 0.0f), matId,
            glm::vec2(uvMax.x, uvMax.y),
            glm::vec2(uvMax.x, uvMax.y),
            glm::vec2(uvMax.x, uvMin.y),
            glm::vec2(uvMax.x, uvMin.y));

    // 4. Left Face (-X normal)
    addQuad(triangles,
            glm::vec3(x0, y0, z0),
            glm::vec3(x0, y0, z1),
            glm::vec3(x0, y1, z1),
            glm::vec3(x0, y1, z0),
            glm::vec3(-1.0f, 0.0f, 0.0f), matId,
            glm::vec2(uvMin.x, uvMax.y),
            glm::vec2(uvMin.x, uvMax.y),
            glm::vec2(uvMin.x, uvMin.y),
            glm::vec2(uvMin.x, uvMin.y));

    // 5. Top Face (+Y normal)
    addQuad(triangles,
            glm::vec3(x0, y1, z1),
            glm::vec3(x1, y1, z1),
            glm::vec3(x1, y1, z0),
            glm::vec3(x0, y1, z0),
            glm::vec3(0.0f, 1.0f, 0.0f), matId,
            glm::vec2(uvMin.x, uvMin.y),
            glm::vec2(uvMax.x, uvMin.y),
            glm::vec2(uvMax.x, uvMin.y),
            glm::vec2(uvMin.x, uvMin.y));

    // 6. Bottom Face (-Y normal)
    addQuad(triangles,
            glm::vec3(x0, y0, z0),
            glm::vec3(x1, y0, z0),
            glm::vec3(x1, y0, z1),
            glm::vec3(x0, y0, z1),
            glm::vec3(0.0f, -1.0f, 0.0f), matId,
            glm::vec2(uvMin.x, uvMax.y),
            glm::vec2(uvMax.x, uvMax.y),
            glm::vec2(uvMax.x, uvMax.y),
            glm::vec2(uvMin.x, uvMax.y));
}

static void addSphere(std::vector<TriangleGPU>& triangles,
                      glm::vec3 center, float radius, uint32_t matId,
                      int rings = 24, int sectors = 24) {
    float const R = 1.0f / static_cast<float>(rings - 1);
    float const S = 1.0f / static_cast<float>(sectors - 1);
    float const PI = 3.14159265358979323846f;

    auto makeTri = [matId](glm::vec3 pA, glm::vec3 nA, glm::vec2 uvA,
                           glm::vec3 pB, glm::vec3 nB, glm::vec2 uvB,
                           glm::vec3 pC, glm::vec3 nC, glm::vec2 uvC) {
        TriangleGPU tri{};
        glm::vec3 up = std::abs(nA.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        tri.v0.position = glm::vec4(pA, uvA.x);
        tri.v0.normal = glm::vec4(nA, uvA.y);
        tri.v0.tangent = glm::vec4(glm::normalize(glm::cross(up, nA)), 1.0f);

        up = std::abs(nB.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        tri.v1.position = glm::vec4(pB, uvB.x);
        tri.v1.normal = glm::vec4(nB, uvB.y);
        tri.v1.tangent = glm::vec4(glm::normalize(glm::cross(up, nB)), 1.0f);

        up = std::abs(nC.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        tri.v2.position = glm::vec4(pC, uvC.x);
        tri.v2.normal = glm::vec4(nC, uvC.y);
        tri.v2.tangent = glm::vec4(glm::normalize(glm::cross(up, nC)), 1.0f);

        tri.materialId = matId;
        return tri;
    };

    for (int r = 0; r < rings - 1; ++r) {
        for (int s = 0; s < sectors - 1; ++s) {
            float y0 = std::cos(PI * r * R);
            float y1 = std::cos(PI * (r + 1) * R);
            float r0 = std::sin(PI * r * R);
            float r1 = std::sin(PI * (r + 1) * R);

            float x00 = r0 * std::sin(2.0f * PI * s * S);
            float z00 = r0 * std::cos(2.0f * PI * s * S);

            float x10 = r1 * std::sin(2.0f * PI * s * S);
            float z10 = r1 * std::cos(2.0f * PI * s * S);

            float x01 = r0 * std::sin(2.0f * PI * (s + 1) * S);
            float z01 = r0 * std::cos(2.0f * PI * (s + 1) * S);

            float x11 = r1 * std::sin(2.0f * PI * (s + 1) * S);
            float z11 = r1 * std::cos(2.0f * PI * (s + 1) * S);

            glm::vec3 n00 = glm::normalize(glm::vec3(x00, y0, z00));
            glm::vec3 n10 = glm::normalize(glm::vec3(x10, y1, z10));
            glm::vec3 n01 = glm::normalize(glm::vec3(x01, y0, z01));
            glm::vec3 n11 = glm::normalize(glm::vec3(x11, y1, z11));

            glm::vec3 p00 = center + n00 * radius;
            glm::vec3 p10 = center + n10 * radius;
            glm::vec3 p01 = center + n01 * radius;
            glm::vec3 p11 = center + n11 * radius;

            glm::vec2 uv00 = glm::vec2(s * S, r * R);
            glm::vec2 uv10 = glm::vec2(s * S, (r + 1) * R);
            glm::vec2 uv01 = glm::vec2((s + 1) * S, r * R);
            glm::vec2 uv11 = glm::vec2((s + 1) * S, (r + 1) * R);

            if (r != 0) {
                triangles.push_back(makeTri(p00, n00, uv00, p01, n01, uv01, p10, n10, uv10));
            }
            if (r != rings - 2) {
                triangles.push_back(makeTri(p01, n01, uv01, p11, n11, uv11, p10, n10, uv10));
            }
        }
    }
}

static void addCylinder(std::vector<TriangleGPU>& triangles,
                        glm::vec3 center, float radius, float height, uint32_t matId,
                        int segments = 12) {
    float halfH = height * 0.5f;
    glm::vec3 topCenter = center + glm::vec3(0.0f, halfH, 0.0f);
    glm::vec3 bottomCenter = center - glm::vec3(0.0f, halfH, 0.0f);
    float const PI = 3.14159265358979323846f;

    for (int i = 0; i < segments; ++i) {
        float a0 = (2.0f * PI * static_cast<float>(i)) / static_cast<float>(segments);
        float a1 = (2.0f * PI * static_cast<float>(i + 1)) / static_cast<float>(segments);

        glm::vec3 n0 = glm::vec3(std::cos(a0), 0.0f, std::sin(a0));
        glm::vec3 n1 = glm::vec3(std::cos(a1), 0.0f, std::sin(a1));
        glm::vec3 nSide = glm::normalize(n0 + n1);

        glm::vec3 p0 = bottomCenter + n0 * radius;
        glm::vec3 p1 = bottomCenter + n1 * radius;
        glm::vec3 p2 = topCenter + n1 * radius;
        glm::vec3 p3 = topCenter + n0 * radius;

        // Side quad
        addQuad(triangles, p0, p1, p2, p3, nSide, matId);

        // Top cap triangle
        TriangleGPU topTri{};
        topTri.v0.position = glm::vec4(topCenter, 0.5f);
        topTri.v0.normal = glm::vec4(0.0f, 1.0f, 0.0f, 0.5f);
        topTri.v0.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        topTri.v1.position = glm::vec4(p2, 1.0f);
        topTri.v1.normal = glm::vec4(0.0f, 1.0f, 0.0f, 1.0f);
        topTri.v1.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        topTri.v2.position = glm::vec4(p3, 0.0f);
        topTri.v2.normal = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
        topTri.v2.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        topTri.materialId = matId;
        triangles.push_back(topTri);

        // Bottom cap triangle
        TriangleGPU botTri{};
        botTri.v0.position = glm::vec4(bottomCenter, 0.5f);
        botTri.v0.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.5f);
        botTri.v0.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        botTri.v1.position = glm::vec4(p0, 0.0f);
        botTri.v1.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
        botTri.v1.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        botTri.v2.position = glm::vec4(p1, 1.0f);
        botTri.v2.normal = glm::vec4(0.0f, -1.0f, 0.0f, 1.0f);
        botTri.v2.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
        botTri.materialId = matId;
        triangles.push_back(botTri);
    }
}

static void addFrustum(std::vector<TriangleGPU>& triangles,
                       glm::vec3 center, glm::vec2 baseSize, glm::vec2 topSize, float height,
                       uint32_t matId) {
    float halfH = height * 0.5f;
    glm::vec2 hb = baseSize * 0.5f;
    glm::vec2 ht = topSize * 0.5f;

    glm::vec3 b0 = center + glm::vec3(-hb.x, -halfH, -hb.y);
    glm::vec3 b1 = center + glm::vec3( hb.x, -halfH, -hb.y);
    glm::vec3 b2 = center + glm::vec3( hb.x, -halfH,  hb.y);
    glm::vec3 b3 = center + glm::vec3(-hb.x, -halfH,  hb.y);

    glm::vec3 t0 = center + glm::vec3(-ht.x,  halfH, -ht.y);
    glm::vec3 t1 = center + glm::vec3( ht.x,  halfH, -ht.y);
    glm::vec3 t2 = center + glm::vec3( ht.x,  halfH,  ht.y);
    glm::vec3 t3 = center + glm::vec3(-ht.x,  halfH,  ht.y);

    // Front (z = +h)
    glm::vec3 nFront = glm::normalize(glm::cross(b2 - b3, t3 - b3));
    addQuad(triangles, b3, b2, t2, t3, nFront, matId);

    // Back (z = -h)
    glm::vec3 nBack = glm::normalize(glm::cross(b0 - b1, t1 - b1));
    addQuad(triangles, b1, b0, t0, t1, nBack, matId);

    // Right (x = +h)
    glm::vec3 nRight = glm::normalize(glm::cross(b1 - b2, t2 - b2));
    addQuad(triangles, b2, b1, t1, t2, nRight, matId);

    // Left (x = -h)
    glm::vec3 nLeft = glm::normalize(glm::cross(b3 - b0, t0 - b0));
    addQuad(triangles, b0, b3, t3, t0, nLeft, matId);

    // Top
    addQuad(triangles, t3, t2, t1, t0, glm::vec3(0.0f, 1.0f, 0.0f), matId);
    // Bottom
    addQuad(triangles, b0, b1, b2, b3, glm::vec3(0.0f, -1.0f, 0.0f), matId);
}

static void addTorus(std::vector<TriangleGPU>& triangles,
                     glm::vec3 center, float majorRadius, float minorRadius, uint32_t matId,
                     int rings = 36, int sectors = 18,
                     glm::vec3 normalAxis = glm::vec3(0.0f, 0.0f, 1.0f)) {
    float const PI = 3.14159265358979323846f;
    glm::vec3 nAxis = glm::normalize(normalAxis);
    glm::vec3 uAxis = std::abs(nAxis.y) < 0.99f ? glm::normalize(glm::cross(nAxis, glm::vec3(0.0f, 1.0f, 0.0f)))
                                                : glm::normalize(glm::cross(nAxis, glm::vec3(1.0f, 0.0f, 0.0f)));
    glm::vec3 vAxis = glm::normalize(glm::cross(nAxis, uAxis));

    auto getTorusPoint = [&](float u, float v) -> std::pair<glm::vec3, glm::vec3> {
        float cosU = std::cos(u);
        float sinU = std::sin(u);
        float cosV = std::cos(v);
        float sinV = std::sin(v);

        glm::vec3 ringCenter = center + (uAxis * cosU + vAxis * sinU) * majorRadius;
        glm::vec3 radialDir = uAxis * cosU + vAxis * sinU;
        glm::vec3 norm = radialDir * cosV + nAxis * sinV;
        glm::vec3 pos = ringCenter + norm * minorRadius;
        return {pos, norm};
    };

    auto makeTri = [matId](glm::vec3 pA, glm::vec3 nA, glm::vec2 uvA,
                           glm::vec3 pB, glm::vec3 nB, glm::vec2 uvB,
                           glm::vec3 pC, glm::vec3 nC, glm::vec2 uvC) {
        TriangleGPU tri{};
        glm::vec3 up = std::abs(nA.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        tri.v0.position = glm::vec4(pA, uvA.x);
        tri.v0.normal = glm::vec4(nA, uvA.y);
        tri.v0.tangent = glm::vec4(glm::normalize(glm::cross(up, nA)), 1.0f);

        up = std::abs(nB.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        tri.v1.position = glm::vec4(pB, uvB.x);
        tri.v1.normal = glm::vec4(nB, uvB.y);
        tri.v1.tangent = glm::vec4(glm::normalize(glm::cross(up, nB)), 1.0f);

        up = std::abs(nC.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
        tri.v2.position = glm::vec4(pC, uvC.x);
        tri.v2.normal = glm::vec4(nC, uvC.y);
        tri.v2.tangent = glm::vec4(glm::normalize(glm::cross(up, nC)), 1.0f);

        tri.materialId = matId;
        return tri;
    };

    for (int r = 0; r < rings; ++r) {
        float u0 = (2.0f * PI * r) / static_cast<float>(rings);
        float u1 = (2.0f * PI * (r + 1)) / static_cast<float>(rings);
        float uTex0 = static_cast<float>(r) / static_cast<float>(rings);
        float uTex1 = static_cast<float>(r + 1) / static_cast<float>(rings);

        for (int s = 0; s < sectors; ++s) {
            float v0 = (2.0f * PI * s) / static_cast<float>(sectors);
            float v1 = (2.0f * PI * (s + 1)) / static_cast<float>(sectors);
            float vTex0 = static_cast<float>(s) / static_cast<float>(sectors);
            float vTex1 = static_cast<float>(s + 1) / static_cast<float>(sectors);

            auto [p00, n00] = getTorusPoint(u0, v0);
            auto [p10, n10] = getTorusPoint(u1, v0);
            auto [p01, n01] = getTorusPoint(u0, v1);
            auto [p11, n11] = getTorusPoint(u1, v1);

            triangles.push_back(makeTri(p00, n00, glm::vec2(uTex0, vTex0),
                                        p10, n10, glm::vec2(uTex1, vTex0),
                                        p11, n11, glm::vec2(uTex1, vTex1)));
            triangles.push_back(makeTri(p00, n00, glm::vec2(uTex0, vTex0),
                                        p11, n11, glm::vec2(uTex1, vTex1),
                                        p01, n01, glm::vec2(uTex0, vTex1)));
        }
    }
}

static std::vector<MaterialGPU> createDefaultCornellBoxMaterials() {
    std::vector<MaterialGPU> materials;

    // Materials:
    // 0: White diffuse (walls/floor/ceiling)
    MaterialGPU matWhite{};
    matWhite.albedo = glm::vec4(0.75f, 0.75f, 0.75f, 1.0f);
    matWhite.roughness = 0.9f;
    matWhite.metallic = 0.0f;
    matWhite.type = MATERIAL_DIFFUSE;
    materials.push_back(matWhite);

    // 1: Red diffuse (left wall)
    MaterialGPU matRed{};
    matRed.albedo = glm::vec4(0.75f, 0.12f, 0.12f, 1.0f);
    matRed.roughness = 0.9f;
    matRed.metallic = 0.0f;
    matRed.type = MATERIAL_DIFFUSE;
    materials.push_back(matRed);

    // 2: Green diffuse (right wall)
    MaterialGPU matGreen{};
    matGreen.albedo = glm::vec4(0.12f, 0.75f, 0.15f, 1.0f);
    matGreen.roughness = 0.9f;
    matGreen.metallic = 0.0f;
    matGreen.type = MATERIAL_DIFFUSE;
    materials.push_back(matGreen);

    // 3: Emissive Light (ceiling area light)
    MaterialGPU matLight{};
    matLight.albedo = glm::vec4(1.0f);
    matLight.emissive = glm::vec4(18.0f, 18.0f, 15.0f, 1.0f);
    matLight.type = MATERIAL_EMISSIVE;
    materials.push_back(matLight);

    // 4: Dielectric Refraction (Glass Sphere)
    MaterialGPU matGlass{};
    matGlass.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    matGlass.roughness = 0.01f;
    matGlass.metallic = 0.0f;
    matGlass.ior = 1.52f; // Crown glass
    matGlass.transmission = 1.0f;
    matGlass.thickness = 1.0f;
    matGlass.type = MATERIAL_DIELECTRIC;
    materials.push_back(matGlass);

    // 5: Metallic Specular (Mirror / Metal Sphere)
    MaterialGPU matMetal{};
    matMetal.albedo = glm::vec4(0.95f, 0.85f, 0.65f, 1.0f); // Gold / polished brass tint
    matMetal.roughness = 0.04f;
    matMetal.metallic = 1.0f;
    matMetal.type = MATERIAL_METALLIC;
    materials.push_back(matMetal);

    // 6: Blue diffuse box
    MaterialGPU matBlue{};
    matBlue.albedo = glm::vec4(0.2f, 0.35f, 0.8f, 1.0f);
    matBlue.roughness = 0.8f;
    matBlue.metallic = 0.0f;
    matBlue.type = MATERIAL_DIFFUSE;
    materials.push_back(matBlue);

    return materials;
}

SceneData ProceduralScene::createCornellBox() {
    SceneData scene;
    scene.materials = createDefaultCornellBoxMaterials();

    auto recordRange = [&](const std::string& name, uint32_t startTri) {
        if (scene.triangles.size() <= startTri) return;
        MeshRange mr{};
        mr.name = name;
        mr.firstTriangle = startTri;
        mr.triangleCount = static_cast<uint32_t>(scene.triangles.size() - startTri);
        for (uint32_t i = startTri; i < scene.triangles.size(); ++i) {
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v0.position));
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v1.position));
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v2.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v0.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v1.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v2.position));
        }
        scene.meshRanges.push_back(mr);
    };

    // Geometry: Box dimensions -1.0 to 1.0
    // Floor (y = 0.0)
    uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3(-1.0f, 0.0f,  1.0f),
            glm::vec3( 1.0f, 0.0f,  1.0f),
            glm::vec3( 1.0f, 0.0f, -1.0f),
            glm::vec3(-1.0f, 0.0f, -1.0f),
            glm::vec3(0.0f, 1.0f, 0.0f), 0);
    recordRange("Floor", tStart);

    // Ceiling (y = 2.0)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3(-1.0f, 2.0f, -1.0f),
            glm::vec3( 1.0f, 2.0f, -1.0f),
            glm::vec3( 1.0f, 2.0f,  1.0f),
            glm::vec3(-1.0f, 2.0f,  1.0f),
            glm::vec3(0.0f, -1.0f, 0.0f), 0);
    recordRange("Ceiling", tStart);

    // Back wall (z = -1.0)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3(-1.0f, 0.0f, -1.0f),
            glm::vec3( 1.0f, 0.0f, -1.0f),
            glm::vec3( 1.0f, 2.0f, -1.0f),
            glm::vec3(-1.0f, 2.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f), 0);
    recordRange("Back Wall", tStart);

    // Left wall (x = -1.0, Red)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3(-1.0f, 0.0f,  1.0f),
            glm::vec3(-1.0f, 0.0f, -1.0f),
            glm::vec3(-1.0f, 2.0f, -1.0f),
            glm::vec3(-1.0f, 2.0f,  1.0f),
            glm::vec3(1.0f, 0.0f, 0.0f), 1);
    recordRange("Left Wall (Red)", tStart);

    // Right wall (x = 1.0, Green)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3( 1.0f, 0.0f, -1.0f),
            glm::vec3( 1.0f, 0.0f,  1.0f),
            glm::vec3( 1.0f, 2.0f,  1.0f),
            glm::vec3( 1.0f, 2.0f, -1.0f),
            glm::vec3(-1.0f, 0.0f, 0.0f), 2);
    recordRange("Right Wall (Green)", tStart);

    // Ceiling Area Light Quad (centered at y = 1.99, size 0.6 x 0.6)
    float lw = 0.35f;
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3(-lw, 1.99f, -lw),
            glm::vec3( lw, 1.99f, -lw),
            glm::vec3( lw, 1.99f,  lw),
            glm::vec3(-lw, 1.99f,  lw),
            glm::vec3(0.0f, -1.0f, 0.0f), 3);
    recordRange("Ceiling Light", tStart);

    // Add Area Light structure for Next Event Estimation
    LightGPU areaLight{};
    areaLight.position = glm::vec4(-lw, 1.99f, -lw, LIGHT_AREA_QUAD);
    areaLight.u = glm::vec4(2.0f * lw, 0.0f, 0.0f, 0.0f);
    areaLight.v = glm::vec4(0.0f, 0.0f, 2.0f * lw, 0.0f);
    areaLight.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
    areaLight.emission = glm::vec4(18.0f, 18.0f, 15.0f, (2.0f * lw) * (2.0f * lw));
    scene.lights.push_back(areaLight);

    // Add a Spot Light for directional accent and soft shadows
    LightGPU spotLight{};
    spotLight.position = glm::vec4(0.4f, 1.8f, 0.5f, LIGHT_SPOT);
    spotLight.normal = glm::vec4(glm::normalize(glm::vec3(-0.4f, -1.4f, -0.7f)), 0.0f);
    spotLight.emission = glm::vec4(25.0f, 22.0f, 18.0f, 1.0f);
    spotLight.u = glm::vec4(0.0f, 0.0f, 0.0f, std::cos(glm::radians(25.0f))); // inner angle cos
    spotLight.v = glm::vec4(0.0f, 0.0f, 0.0f, std::cos(glm::radians(35.0f))); // outer angle cos
    scene.lights.push_back(spotLight);

    // Interior objects:
    // 1. Tall diffuse blue box on the right
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addBox(scene.triangles, glm::vec3(0.35f, 0.6f, -0.3f), glm::vec3(0.55f, 1.2f, 0.55f), 22.0f, 6);
    recordRange("Tall Blue Box", tStart);

    // 2. Glass Sphere (Dielectric Refraction) on the left
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addSphere(scene.triangles, glm::vec3(-0.4f, 0.35f, -0.35f), 0.35f, 4);
    recordRange("Glass Sphere", tStart);

    // 3. Metallic / Mirror Sphere in the foreground
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addSphere(scene.triangles, glm::vec3(0.1f, 0.25f, 0.35f), 0.25f, 5);
    scene.hasDielectrics = true;
    scene.dielectricBoundsMin = glm::vec3(-0.4f, 0.35f, -0.35f) - glm::vec3(0.35f);
    scene.dielectricBoundsMax = glm::vec3(-0.4f, 0.35f, -0.35f) + glm::vec3(0.35f);

    scene.hasCamera = true;
    scene.cameraPosition = glm::vec3(0.0f, 1.0f, 2.7f);
    scene.cameraTarget = glm::vec3(0.0f, 1.0f, 0.0f);
    scene.cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    scene.cameraFov = 45.0f;

    scene.boundsMin = glm::vec3(-1.0f, 0.0f, -1.0f);
    scene.boundsMax = glm::vec3(1.0f, 2.0f, 1.0f);
    scene.sceneRadius = 2.0f;
    scene.focalBoundsMin = scene.boundsMin;
    scene.focalBoundsMax = scene.boundsMax;
    scene.focalRadius = scene.sceneRadius;
    scene.focalDistance = glm::length(scene.cameraPosition - scene.cameraTarget);
    scene.centralTarget = scene.cameraTarget;

    return scene;
}

SceneData ProceduralScene::createManyLightsScene(uint32_t gridDim) {
    SceneData scene;
    scene.materials = createDefaultCornellBoxMaterials();

    auto recordRange = [&](const std::string& name, uint32_t startIdx) {
        MeshRange mr;
        mr.name = name;
        mr.firstTriangle = startIdx;
        mr.triangleCount = static_cast<uint32_t>(scene.triangles.size()) - startIdx;
        mr.minBound = glm::vec3(1e30f);
        mr.maxBound = glm::vec3(-1e30f);
        for (size_t i = startIdx; i < scene.triangles.size(); ++i) {
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v0.position));
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v1.position));
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v2.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v0.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v1.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v2.position));
        }
        scene.meshRanges.push_back(mr);
    };

    // Geometry: Box dimensions -1.0 to 1.0
    // Floor (y = 0.0)
    uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3(-1.0f, 0.0f,  1.0f),
            glm::vec3( 1.0f, 0.0f,  1.0f),
            glm::vec3( 1.0f, 0.0f, -1.0f),
            glm::vec3(-1.0f, 0.0f, -1.0f),
            glm::vec3(0.0f, 1.0f, 0.0f), 0);
    recordRange("Floor", tStart);

    // Ceiling (y = 2.0)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3(-1.0f, 2.0f, -1.0f),
            glm::vec3( 1.0f, 2.0f, -1.0f),
            glm::vec3( 1.0f, 2.0f,  1.0f),
            glm::vec3(-1.0f, 2.0f,  1.0f),
            glm::vec3(0.0f, -1.0f, 0.0f), 0);
    recordRange("Ceiling", tStart);

    // Back wall (z = -1.0)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3(-1.0f, 0.0f, -1.0f),
            glm::vec3( 1.0f, 0.0f, -1.0f),
            glm::vec3( 1.0f, 2.0f, -1.0f),
            glm::vec3(-1.0f, 2.0f, -1.0f),
            glm::vec3(0.0f, 0.0f, 1.0f), 0);
    recordRange("Back Wall", tStart);

    // Left wall (x = -1.0, Red)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3(-1.0f, 0.0f,  1.0f),
            glm::vec3(-1.0f, 0.0f, -1.0f),
            glm::vec3(-1.0f, 2.0f, -1.0f),
            glm::vec3(-1.0f, 2.0f,  1.0f),
            glm::vec3(1.0f, 0.0f, 0.0f), 1);
    recordRange("Left Wall (Red)", tStart);

    // Right wall (x = 1.0, Green)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3( 1.0f, 0.0f, -1.0f),
            glm::vec3( 1.0f, 0.0f,  1.0f),
            glm::vec3( 1.0f, 2.0f,  1.0f),
            glm::vec3( 1.0f, 2.0f, -1.0f),
            glm::vec3(-1.0f, 0.0f, 0.0f), 2);
    recordRange("Right Wall (Green)", tStart);

    // Grid of ceiling lights (y = 1.99)
    uint32_t N = std::clamp(gridDim, 2u, 16u);
    float step = 1.6f / static_cast<float>(N);
    float halfSize = std::min(step * 0.35f, 0.04f);
    float area = (2.0f * halfSize) * (2.0f * halfSize);

    tStart = static_cast<uint32_t>(scene.triangles.size());
    for (uint32_t gz = 0; gz < N; ++gz) {
        for (uint32_t gx = 0; gx < N; ++gx) {
            float cx = -0.8f + (static_cast<float>(gx) + 0.5f) * step;
            float cz = -0.8f + (static_cast<float>(gz) + 0.5f) * step;

            // Generate pleasing varied spectral colors across the grid
            float uHue = static_cast<float>(gz * N + gx) / static_cast<float>(N * N);
            float hue = uHue * 6.0f;
            int hIdx = static_cast<int>(hue) % 6;
            float f = hue - std::floor(hue);
            glm::vec3 col(1.0f);
            if (hIdx == 0) col = glm::vec3(1.0f, f, 0.2f);
            else if (hIdx == 1) col = glm::vec3(1.0f - f, 1.0f, 0.2f);
            else if (hIdx == 2) col = glm::vec3(0.2f, 1.0f, f);
            else if (hIdx == 3) col = glm::vec3(0.2f, 1.0f - f, 1.0f);
            else if (hIdx == 4) col = glm::vec3(f, 0.2f, 1.0f);
            else col = glm::vec3(1.0f, 0.2f, 1.0f - f);

            // Add visible emissive quad mesh on ceiling
            addQuad(scene.triangles,
                    glm::vec3(cx - halfSize, 1.99f, cz - halfSize),
                    glm::vec3(cx + halfSize, 1.99f, cz - halfSize),
                    glm::vec3(cx + halfSize, 1.99f, cz + halfSize),
                    glm::vec3(cx - halfSize, 1.99f, cz + halfSize),
                    glm::vec3(0.0f, -1.0f, 0.0f), 3);

            // Add corresponding physical area light
            LightGPU light{};
            light.position = glm::vec4(cx - halfSize, 1.99f, cz - halfSize, LIGHT_AREA_QUAD);
            light.u = glm::vec4(2.0f * halfSize, 0.0f, 0.0f, 0.0f);
            light.v = glm::vec4(0.0f, 0.0f, 2.0f * halfSize, 0.0f);
            light.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
            float intensity = 15.0f;
            light.emission = glm::vec4(col * intensity, area);
            scene.lights.push_back(light);
        }
    }
    recordRange("Many Ceiling Lights", tStart);

    // Interior objects:
    // 1. Tall diffuse blue box on the right
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addBox(scene.triangles, glm::vec3(0.35f, 0.6f, -0.3f), glm::vec3(0.55f, 1.2f, 0.55f), 22.0f, 6);
    recordRange("Tall Blue Box", tStart);

    // 2. Glass Sphere (Dielectric Refraction) on the left
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addSphere(scene.triangles, glm::vec3(-0.4f, 0.35f, -0.35f), 0.35f, 4);
    recordRange("Glass Sphere", tStart);

    // 3. Metallic / Mirror Sphere in the foreground
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addSphere(scene.triangles, glm::vec3(0.1f, 0.25f, 0.35f), 0.25f, 5);
    recordRange("Mirror Sphere", tStart);

    scene.hasCamera = true;
    scene.cameraPosition = glm::vec3(0.0f, 1.0f, 2.7f);
    scene.cameraTarget = glm::vec3(0.0f, 1.0f, 0.0f);
    scene.cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    scene.cameraFov = 45.0f;

    scene.boundsMin = glm::vec3(-1.0f, 0.0f, -1.0f);
    scene.boundsMax = glm::vec3(1.0f, 2.0f, 1.0f);
    scene.sceneRadius = 2.0f;
    scene.focalBoundsMin = scene.boundsMin;
    scene.focalBoundsMax = scene.boundsMax;
    scene.focalRadius = scene.sceneRadius;
    scene.focalDistance = glm::length(scene.cameraPosition - scene.cameraTarget);
    scene.centralTarget = scene.cameraTarget;

    return scene;
}

static std::vector<MaterialGPU> createCyberCityMaterials() {
    std::vector<MaterialGPU> materials;
    materials.reserve(48);

    // --- 0 to 9: Core Architectural & Structural Materials ---
    // 0: Dark Structural Foundation Concrete (Diffuse)
    MaterialGPU mat0{};
    mat0.albedo = glm::vec4(0.14f, 0.14f, 0.16f, 1.0f);
    mat0.roughness = 0.88f;
    mat0.metallic = 0.0f;
    mat0.type = MATERIAL_DIFFUSE;
    materials.push_back(mat0);

    // 1: Carbon Steel Structural Framing / Girders (Metallic)
    MaterialGPU mat1{};
    mat1.albedo = glm::vec4(0.22f, 0.24f, 0.26f, 1.0f);
    mat1.roughness = 0.25f;
    mat1.metallic = 0.94f;
    mat1.type = MATERIAL_METALLIC;
    materials.push_back(mat1);

    // 2: Mirror Polished Titanium / Chrome Panels (Conductor)
    MaterialGPU mat2{};
    mat2.albedo = glm::vec4(0.92f, 0.94f, 0.97f, 1.0f);
    mat2.roughness = 0.020f;
    mat2.metallic = 1.0f;
    mat2.type = MATERIAL_METALLIC;
    materials.push_back(mat2);

    // 3: Brushed Brass / Gold Trim (Metallic)
    MaterialGPU mat3{};
    mat3.albedo = glm::vec4(0.95f, 0.78f, 0.35f, 1.0f);
    mat3.roughness = 0.18f;
    mat3.metallic = 1.0f;
    mat3.type = MATERIAL_METALLIC;
    materials.push_back(mat3);

    // 4: Wet Reflective Plaza Pavement (Diffuse + Procedural Puddle Mask)
    MaterialGPU mat4{};
    mat4.albedo = glm::vec4(0.06f, 0.06f, 0.08f, 1.0f);
    mat4.roughness = 0.35f;
    mat4.metallic = 0.0f;
    mat4.clearcoat = 0.0f;
    mat4.type = MATERIAL_DIFFUSE | MATERIAL_FLAG_PROCEDURAL_PUDDLE;
    materials.push_back(mat4);

    // 5: Transmissive Structural Crown Glass (Dielectric Refraction)
    MaterialGPU mat5{};
    mat5.albedo = glm::vec4(0.98f, 1.0f, 0.99f, 1.0f);
    mat5.roughness = 0.012f;
    mat5.metallic = 0.0f;
    mat5.ior = 1.52f;
    mat5.transmission = 1.0f;
    mat5.thickness = 0.15f;
    mat5.type = MATERIAL_DIELECTRIC;
    materials.push_back(mat5);

    // 6: Transmissive Tinted Cyan Canopy Glass (Dielectric Refraction)
    MaterialGPU mat6{};
    mat6.albedo = glm::vec4(0.25f, 0.90f, 0.98f, 1.0f);
    mat6.roughness = 0.030f;
    mat6.metallic = 0.0f;
    mat6.ior = 1.55f;
    mat6.transmission = 0.95f;
    mat6.thickness = 0.20f;
    mat6.type = MATERIAL_DIELECTRIC;
    materials.push_back(mat6);

    // 7: Dark Anodized Bronze / Weathered Metal (Metallic)
    MaterialGPU mat7{};
    mat7.albedo = glm::vec4(0.35f, 0.25f, 0.18f, 1.0f);
    mat7.roughness = 0.32f;
    mat7.metallic = 0.90f;
    mat7.type = MATERIAL_METALLIC;
    materials.push_back(mat7);

    // 8: Basalt / Rough Granite Foundation Blocks (Diffuse)
    MaterialGPU mat8{};
    mat8.albedo = glm::vec4(0.10f, 0.10f, 0.11f, 1.0f);
    mat8.roughness = 0.92f;
    mat8.metallic = 0.0f;
    mat8.type = MATERIAL_DIFFUSE;
    materials.push_back(mat8);

    // 9: White Ceramic Architectural Cladding Tiles (Diffuse + Clearcoat)
    MaterialGPU mat9{};
    mat9.albedo = glm::vec4(0.92f, 0.92f, 0.94f, 1.0f);
    mat9.roughness = 0.20f;
    mat9.metallic = 0.02f;
    mat9.clearcoat = 0.85f;
    mat9.clearcoatRoughness = 0.04f;
    mat9.type = MATERIAL_DIFFUSE;
    materials.push_back(mat9);

    // --- 10 to 19: Advanced Conductors & Dielectrics ---
    // 10: Copper Conduits & Busbars (Metallic)
    MaterialGPU mat10{};
    mat10.albedo = glm::vec4(0.95f, 0.55f, 0.35f, 1.0f);
    mat10.roughness = 0.14f;
    mat10.metallic = 1.0f;
    mat10.type = MATERIAL_METALLIC;
    materials.push_back(mat10);

    // 11: Smoked Obsidian Glass (Dielectric with Absorption)
    MaterialGPU mat11{};
    mat11.albedo = glm::vec4(0.20f, 0.20f, 0.25f, 1.0f);
    mat11.roughness = 0.025f;
    mat11.metallic = 0.0f;
    mat11.ior = 1.65f;
    mat11.transmission = 0.75f;
    mat11.thickness = 0.40f;
    mat11.type = MATERIAL_DIELECTRIC;
    materials.push_back(mat11);

    // 12: Frosted Amber Glass (Dielectric Rough)
    MaterialGPU mat12{};
    mat12.albedo = glm::vec4(0.95f, 0.75f, 0.30f, 1.0f);
    mat12.roughness = 0.22f;
    mat12.metallic = 0.0f;
    mat12.ior = 1.48f;
    mat12.transmission = 0.85f;
    mat12.thickness = 0.25f;
    mat12.type = MATERIAL_DIELECTRIC;
    materials.push_back(mat12);

    // 13: Emerald Crystal Spire (High-Index Dielectric)
    MaterialGPU mat13{};
    mat13.albedo = glm::vec4(0.15f, 0.95f, 0.50f, 1.0f);
    mat13.roughness = 0.015f;
    mat13.metallic = 0.0f;
    mat13.ior = 1.76f;
    mat13.transmission = 0.90f;
    mat13.thickness = 0.50f;
    mat13.type = MATERIAL_DIELECTRIC;
    materials.push_back(mat13);

    // 14: Ruby Laser Optical Glass (Dielectric)
    MaterialGPU mat14{};
    mat14.albedo = glm::vec4(0.98f, 0.15f, 0.25f, 1.0f);
    mat14.roughness = 0.018f;
    mat14.metallic = 0.0f;
    mat14.ior = 1.77f;
    mat14.transmission = 0.88f;
    mat14.thickness = 0.50f;
    mat14.type = MATERIAL_DIELECTRIC;
    materials.push_back(mat14);

    // 15: Dispersive Diamond Prism (High IOR Dielectric + Dispersion)
    MaterialGPU mat15{};
    mat15.albedo = glm::vec4(1.0f);
    mat15.roughness = 0.008f;
    mat15.metallic = 0.0f;
    mat15.ior = 2.42f;
    mat15.transmission = 1.0f;
    mat15.thickness = 0.30f;
    mat15.dispersion = 0.08f;
    mat15.type = MATERIAL_DIELECTRIC;
    materials.push_back(mat15);

    // 16: Iridescent Anti-Reflective Coating (Dielectric + Iridescence)
    MaterialGPU mat16{};
    mat16.albedo = glm::vec4(0.90f, 0.95f, 1.0f, 1.0f);
    mat16.roughness = 0.020f;
    mat16.metallic = 0.0f;
    mat16.ior = 1.50f;
    mat16.transmission = 0.90f;
    mat16.thickness = 0.25f;
    mat16.iridescence = 0.85f;
    mat16.iridescenceIor = 1.33f;
    mat16.iridescenceThickness = 0.45f;
    mat16.type = MATERIAL_DIELECTRIC;
    materials.push_back(mat16);

    // 17: Anisotropic Brushed Platinum Panels (Metallic + Anisotropy)
    MaterialGPU mat17{};
    mat17.albedo = glm::vec4(0.85f, 0.86f, 0.88f, 1.0f);
    mat17.roughness = 0.12f;
    mat17.metallic = 1.0f;
    mat17.anisotropyStrength = 0.85f;
    mat17.anisotropyRotation = 0.5f;
    mat17.type = MATERIAL_METALLIC;
    materials.push_back(mat17);

    // 18: Rose Gold Filigree / Accent Panels (Metallic)
    MaterialGPU mat18{};
    mat18.albedo = glm::vec4(0.96f, 0.72f, 0.68f, 1.0f);
    mat18.roughness = 0.08f;
    mat18.metallic = 1.0f;
    mat18.type = MATERIAL_METALLIC;
    materials.push_back(mat18);

    // 19: Gunmetal Alloy Armor / Structural Nodes (Metallic)
    MaterialGPU mat19{};
    mat19.albedo = glm::vec4(0.28f, 0.30f, 0.32f, 1.0f);
    mat19.roughness = 0.35f;
    mat19.metallic = 0.88f;
    mat19.type = MATERIAL_METALLIC;
    materials.push_back(mat19);

    // --- 20 to 29: Secondary Composites, Utilities & Fabrics ---
    // 20: Dark Asphalt Roadway Surface (Diffuse + Procedural Puddle Mask)
    MaterialGPU mat20{};
    mat20.albedo = glm::vec4(0.07f, 0.07f, 0.08f, 1.0f);
    mat20.roughness = 0.55f;
    mat20.metallic = 0.0f;
    mat20.clearcoat = 0.0f;
    mat20.type = MATERIAL_DIFFUSE | MATERIAL_FLAG_PROCEDURAL_PUDDLE;
    materials.push_back(mat20);

    // 21: Weathered Rusted Iron (Diffuse Rough)
    MaterialGPU mat21{};
    mat21.albedo = glm::vec4(0.55f, 0.25f, 0.15f, 1.0f);
    mat21.roughness = 0.88f;
    mat21.metallic = 0.15f;
    mat21.type = MATERIAL_DIFFUSE;
    materials.push_back(mat21);

    // 22: Rubber Transit Track Dampers (Diffuse Deep Matte)
    MaterialGPU mat22{};
    mat22.albedo = glm::vec4(0.08f, 0.08f, 0.09f, 1.0f);
    mat22.roughness = 0.95f;
    mat22.metallic = 0.0f;
    mat22.type = MATERIAL_DIFFUSE;
    materials.push_back(mat22);

    // 23: Velvet Sheen Lounge Fabrics (Diffuse + Sheen)
    MaterialGPU mat23{};
    mat23.albedo = glm::vec4(0.35f, 0.08f, 0.18f, 1.0f);
    mat23.roughness = 0.65f;
    mat23.metallic = 0.0f;
    mat23.sheenColor = glm::vec3(0.85f, 0.20f, 0.45f);
    mat23.sheenRoughness = 0.45f;
    mat23.type = MATERIAL_DIFFUSE;
    materials.push_back(mat23);

    // 24: High-Gloss Yellow Hazard Enamel (Diffuse + Clearcoat)
    MaterialGPU mat24{};
    mat24.albedo = glm::vec4(0.95f, 0.80f, 0.05f, 1.0f);
    mat24.roughness = 0.12f;
    mat24.metallic = 0.0f;
    mat24.clearcoat = 0.90f;
    mat24.clearcoatRoughness = 0.03f;
    mat24.type = MATERIAL_DIFFUSE;
    materials.push_back(mat24);

    // 25: Matte Black Stealth Paneling (Diffuse Absorber)
    MaterialGPU mat25{};
    mat25.albedo = glm::vec4(0.025f, 0.025f, 0.030f, 1.0f);
    mat25.roughness = 0.95f;
    mat25.metallic = 0.0f;
    mat25.type = MATERIAL_DIFFUSE;
    materials.push_back(mat25);

    // 26: Solar Photovoltaic Cell Panels (Metallic Specular Deep Blue)
    MaterialGPU mat26{};
    mat26.albedo = glm::vec4(0.08f, 0.12f, 0.32f, 1.0f);
    mat26.roughness = 0.04f;
    mat26.metallic = 0.80f;
    mat26.type = MATERIAL_METALLIC;
    materials.push_back(mat26);

    // 27: Galvanized Steel Truss Structures (Metallic)
    MaterialGPU mat27{};
    mat27.albedo = glm::vec4(0.60f, 0.62f, 0.64f, 1.0f);
    mat27.roughness = 0.42f;
    mat27.metallic = 0.85f;
    mat27.type = MATERIAL_METALLIC;
    materials.push_back(mat27);

    // 28: Perforated Acoustic / Ventilation Mesh (Diffuse)
    MaterialGPU mat28{};
    mat28.albedo = glm::vec4(0.25f, 0.26f, 0.28f, 1.0f);
    mat28.roughness = 0.70f;
    mat28.metallic = 0.30f;
    mat28.type = MATERIAL_DIFFUSE;
    materials.push_back(mat28);

    // 29: Polished Jade Ornamental Architectural Trim (Clearcoat)
    MaterialGPU mat29{};
    mat29.albedo = glm::vec4(0.12f, 0.48f, 0.28f, 1.0f);
    mat29.roughness = 0.15f;
    mat29.metallic = 0.0f;
    mat29.clearcoat = 0.85f;
    mat29.clearcoatRoughness = 0.05f;
    mat29.type = MATERIAL_DIFFUSE;
    materials.push_back(mat29);

    // --- 30 to 47: 18 Spectral Emissive Neons, Lasers & Displays ---
    // 30: Lower-Level Billboard Video Display (Flat 2D Emission)
    MaterialGPU mat30{};
    mat30.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    mat30.emissive = glm::vec4(1.2f, 1.2f, 1.2f, 1.0f);
    mat30.emissiveTex = 1;
    mat30.type = MATERIAL_EMISSIVE;
    materials.push_back(mat30);

    // 31: Neon Magenta 650nm (Holographic Billboard)
    MaterialGPU mat31{};
    mat31.albedo = glm::vec4(1.0f, 0.1f, 0.6f, 1.0f);
    mat31.emissive = glm::vec4(42.0f, 2.0f, 22.0f, 1.0f);
    mat31.type = MATERIAL_EMISSIVE | MATERIAL_FLAG_PROCEDURAL_HOLO;
    materials.push_back(mat31);

    // 32: Neon Blaze Orange 600nm (Holographic Billboard)
    MaterialGPU mat32{};
    mat32.albedo = glm::vec4(1.0f, 0.4f, 0.05f, 1.0f);
    mat32.emissive = glm::vec4(45.0f, 15.0f, 1.5f, 1.0f);
    mat32.type = MATERIAL_EMISSIVE | MATERIAL_FLAG_PROCEDURAL_HOLO;
    materials.push_back(mat32);

    // 33: Neon Acid Green 520nm
    MaterialGPU mat33{};
    mat33.albedo = glm::vec4(0.2f, 1.0f, 0.2f, 1.0f);
    mat33.emissive = glm::vec4(5.0f, 40.0f, 5.0f, 1.0f);
    mat33.type = MATERIAL_EMISSIVE;
    materials.push_back(mat33);

    // 34: Ruby Laser Warning Beacon 700nm
    MaterialGPU mat34{};
    mat34.albedo = glm::vec4(1.0f, 0.05f, 0.05f, 1.0f);
    mat34.emissive = glm::vec4(60.0f, 2.0f, 2.0f, 1.0f);
    mat34.type = MATERIAL_EMISSIVE;
    materials.push_back(mat34);

    // 35: Electric Gold Conduit / Maglev Rail 580nm
    MaterialGPU mat35{};
    mat35.albedo = glm::vec4(1.0f, 0.85f, 0.3f, 1.0f);
    mat35.emissive = glm::vec4(38.0f, 28.0f, 5.0f, 1.0f);
    mat35.type = MATERIAL_EMISSIVE;
    materials.push_back(mat35);

    // 36: Rooftop Holographic Video Projection Display (Thin-Walled Translucent Holographic Shader)
    MaterialGPU mat36{};
    mat36.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    mat36.emissive = glm::vec4(1.0f, 1.1f, 1.3f, 1.0f);
    mat36.emissiveTex = 1;
    mat36.ior = 1.0f;
    mat36.transmission = 0.0f;
    mat36.thickness = 0.0f;
    mat36.roughness = 0.0f;
    mat36.type = MATERIAL_EMISSIVE | MATERIAL_FLAG_HOLO_VIDEO;
    materials.push_back(mat36);

    // 37: Deep Cobalt Blue 450nm
    MaterialGPU mat37{};
    mat37.albedo = glm::vec4(0.05f, 0.2f, 1.0f, 1.0f);
    mat37.emissive = glm::vec4(2.0f, 8.0f, 50.0f, 1.0f);
    mat37.type = MATERIAL_EMISSIVE;
    materials.push_back(mat37);

    // 38: Hot Pink Hologram
    MaterialGPU mat38{};
    mat38.albedo = glm::vec4(1.0f, 0.2f, 0.7f, 1.0f);
    mat38.emissive = glm::vec4(48.0f, 8.0f, 32.0f, 1.0f);
    mat38.type = MATERIAL_EMISSIVE | MATERIAL_FLAG_PROCEDURAL_HOLO;
    materials.push_back(mat38);

    // 39: Mint Phosphor Luminescence
    MaterialGPU mat39{};
    mat39.albedo = glm::vec4(0.3f, 1.0f, 0.7f, 1.0f);
    mat39.emissive = glm::vec4(10.0f, 45.0f, 28.0f, 1.0f);
    mat39.type = MATERIAL_EMISSIVE;
    materials.push_back(mat39);

    // 40: Ultraviolet Blacklight Glow
    MaterialGPU mat40{};
    mat40.albedo = glm::vec4(0.5f, 0.05f, 0.95f, 1.0f);
    mat40.emissive = glm::vec4(18.0f, 1.0f, 38.0f, 1.0f);
    mat40.type = MATERIAL_EMISSIVE;
    materials.push_back(mat40);

    // 41: Warm Sodium Vapor Streetlight
    MaterialGPU mat41{};
    mat41.albedo = glm::vec4(1.0f, 0.75f, 0.2f, 1.0f);
    mat41.emissive = glm::vec4(35.0f, 22.0f, 2.0f, 1.0f);
    mat41.type = MATERIAL_EMISSIVE;
    materials.push_back(mat41);

    // 42: High-Intensity Xenon White
    MaterialGPU mat42{};
    mat42.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    mat42.emissive = glm::vec4(45.0f, 48.0f, 52.0f, 1.0f);
    mat42.type = MATERIAL_EMISSIVE;
    materials.push_back(mat42);

    // 43: Amber Hazard Flasher
    MaterialGPU mat43{};
    mat43.albedo = glm::vec4(1.0f, 0.6f, 0.05f, 1.0f);
    mat43.emissive = glm::vec4(40.0f, 20.0f, 0.5f, 1.0f);
    mat43.type = MATERIAL_EMISSIVE;
    materials.push_back(mat43);

    // 44: Crimson Strobe Beacon
    MaterialGPU mat44{};
    mat44.albedo = glm::vec4(1.0f, 0.05f, 0.15f, 1.0f);
    mat44.emissive = glm::vec4(55.0f, 1.0f, 5.0f, 1.0f);
    mat44.type = MATERIAL_EMISSIVE;
    materials.push_back(mat44);

    // 45: Electric Turquoise Display
    MaterialGPU mat45{};
    mat45.albedo = glm::vec4(0.05f, 0.95f, 0.85f, 1.0f);
    mat45.emissive = glm::vec4(1.0f, 38.0f, 35.0f, 1.0f);
    mat45.type = MATERIAL_EMISSIVE;
    materials.push_back(mat45);

    // 46: Lime Yellow Neon Sign
    MaterialGPU mat46{};
    mat46.albedo = glm::vec4(0.85f, 1.0f, 0.15f, 1.0f);
    mat46.emissive = glm::vec4(35.0f, 42.0f, 4.0f, 1.0f);
    mat46.type = MATERIAL_EMISSIVE;
    materials.push_back(mat46);

    // 47: Ice Blue Transit Headlight
    MaterialGPU mat47{};
    mat47.albedo = glm::vec4(0.7f, 0.85f, 1.0f, 1.0f);
    mat47.emissive = glm::vec4(28.0f, 38.0f, 50.0f, 1.0f);
    mat47.type = MATERIAL_EMISSIVE;
    materials.push_back(mat47);

    return materials;
}

enum CyberBlasType : uint32_t {
    CYBER_BLAS_TOWER_BASE = 0,
    CYBER_BLAS_TOWER_MID_A = 1,
    CYBER_BLAS_TOWER_MID_B = 2,
    CYBER_BLAS_TOWER_MID_C = 3,
    CYBER_BLAS_TOWER_CROWN = 4,
    CYBER_BLAS_ROOFTOP_HVAC = 5,
    CYBER_BLAS_SKYBRIDGE = 6,
    CYBER_BLAS_COMM_GANTRY = 7,
    CYBER_BLAS_TRANSIT_GUIDEWAY = 8,
    CYBER_BLAS_SKY_CAB = 9,
    CYBER_BLAS_BILLBOARD_CYAN = 10,
    CYBER_BLAS_BILLBOARD_MAGENTA = 11,
    CYBER_BLAS_BILLBOARD_ORANGE = 12,
    CYBER_BLAS_BILLBOARD_VIOLET = 13,
    CYBER_BLAS_SOLAR_ROOF = 14,
    CYBER_BLAS_PLAZA_DISTRICT = 15,
    CYBER_BLAS_ROAD_AVENUE = 16,
    CYBER_BLAS_ROAD_INTERSECTION = 17,
    CYBER_BLAS_PERIMETER_GROUND = 18,
    CYBER_BLAS_HOLO_PROJECTOR = 19,
    CYBER_BLAS_COUNT = 20
};

static void addWindowGrid(std::vector<TriangleGPU>& triangles,
                          glm::vec3 origin, glm::vec3 uAxis, glm::vec3 vAxis,
                          int rows, int cols, float spanU, float spanV,
                          uint32_t mullionMat, uint32_t glassMat, uint32_t emissiveMat,
                          uint32_t seed, uint32_t sillMat = 3, uint32_t accentMat = 30) {
    float stepU = spanU / static_cast<float>(cols);
    float stepV = spanV / static_cast<float>(rows);
    glm::vec3 normal = glm::normalize(glm::cross(uAxis, vAxis));

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            glm::vec3 p = origin + uAxis * ((static_cast<float>(c) + 0.5f) * stepU)
                                 + vAxis * ((static_cast<float>(r) + 0.5f) * stepV);
            // Window mullion frame
            addBox(triangles, p, glm::vec3(stepU * 0.94f, stepV * 0.94f, 0.12f), 0.0f, mullionMat);

            // Recessed window sill (adds micro-facet depth)
            glm::vec3 pSill = p + normal * 0.02f - vAxis * (stepV * 0.42f);
            addBox(triangles, pSill, glm::vec3(stepU * 0.88f, 0.04f, 0.08f), 0.0f, sillMat);

            // Recessed pane
            uint32_t pSeed = (seed + static_cast<uint32_t>(r * 19 + c * 31)) % 100u;
            uint32_t paneMat = (pSeed < 24u) ? emissiveMat : glassMat;
            glm::vec3 pGlass = p + normal * 0.04f;
            glm::vec3 halfU = uAxis * (stepU * 0.38f);
            glm::vec3 halfV = vAxis * (stepV * 0.38f);
            addQuad(triangles,
                    pGlass - halfU - halfV,
                    pGlass + halfU - halfV,
                    pGlass + halfU + halfV,
                    pGlass - halfU + halfV,
                    normal, paneMat);

            // Micro LED indicator / architectural accent strip on select windows
            if (pSeed > 78u) {
                glm::vec3 pLed = p + normal * 0.05f + vAxis * (stepV * 0.40f);
                addBox(triangles, pLed, glm::vec3(stepU * 0.70f, 0.02f, 0.02f), 0.0f, accentMat);
            }
        }
    }
}

static void addDiagridLattice(std::vector<TriangleGPU>& triangles,
                              glm::vec3 origin, glm::vec3 uAxis, glm::vec3 vAxis,
                              int segments, float spanU, float spanV,
                              uint32_t trussMat, uint32_t nodeMat = 2) {
    float stepU = spanU / static_cast<float>(segments);
    for (int s = 0; s < segments; ++s) {
        glm::vec3 b0 = origin + uAxis * (static_cast<float>(s) * stepU);
        glm::vec3 b1 = origin + uAxis * (static_cast<float>(s + 1) * stepU);
        glm::vec3 t0 = b0 + vAxis * spanV;
        glm::vec3 t1 = b1 + vAxis * spanV;
        addBox(triangles, (b0 + t1) * 0.5f, glm::vec3(stepU * 0.10f, spanV * 1.02f, 0.14f), 45.0f, trussMat);
        addBox(triangles, (b1 + t0) * 0.5f, glm::vec3(stepU * 0.10f, spanV * 1.02f, 0.14f), -45.0f, trussMat);
        // Gusset / connection nodes
        addBox(triangles, (b0 + b1) * 0.5f, glm::vec3(stepU * 0.16f, 0.12f, 0.16f), 0.0f, nodeMat);
        addBox(triangles, (t0 + t1) * 0.5f, glm::vec3(stepU * 0.16f, 0.12f, 0.16f), 0.0f, nodeMat);
    }
}

SceneData ProceduralScene::createCyberCityScene() {
    SceneData scene;
    scene.materials = createCyberCityMaterials();

    // Ingest video billboard & hologram texture (640x360 RGBA)
    {
        TextureData videoTex;
        videoTex.width = 640;
        videoTex.height = 360;
        videoTex.isSrgb = false;

        videoTex.pixels.resize(640 * 360 * 4);

        bool loaded = false;
        const std::vector<std::string> candidates = {
            "scenes/cyber_city/cyber_city_frame_0.raw",
            "../scenes/cyber_city/cyber_city_frame_0.raw",
            "../../scenes/cyber_city/cyber_city_frame_0.raw"
        };
        for (const auto& path : candidates) {
            std::ifstream file(path, std::ios::binary);
            if (file) {
                file.read(reinterpret_cast<char*>(videoTex.pixels.data()), videoTex.pixels.size());
                if (file.gcount() == static_cast<std::streamsize>(videoTex.pixels.size())) {
                    loaded = true;
                    break;
                }
            }
        }

        if (!loaded) {
            for (uint32_t py = 0; py < 360; ++py) {
                for (uint32_t px = 0; px < 640; ++px) {
                    size_t idx = (py * 640 + px) * 4;
                    videoTex.pixels[idx + 0] = static_cast<uint8_t>(px * 255 / 640);
                    videoTex.pixels[idx + 1] = static_cast<uint8_t>(200);
                    videoTex.pixels[idx + 2] = static_cast<uint8_t>(py * 255 / 360);
                    videoTex.pixels[idx + 3] = 255;
                }
            }
        }
        scene.textures.push_back(std::move(videoTex));
    }

    auto recordBlasPrototype = [&](const std::string& name, uint32_t startTri) {
        uint32_t triCount = static_cast<uint32_t>(scene.triangles.size() - startTri);
        BlasGeometryRange range{};
        range.firstTriangle = startTri;
        range.triangleCount = triCount;
        range.numOpaqueTriangles = 0; // populated during partitioning
        scene.blasRanges.push_back(range);

        MeshRange mr{};
        mr.name = name;
        mr.firstTriangle = startTri;
        mr.triangleCount = triCount;
        for (uint32_t i = startTri; i < scene.triangles.size(); ++i) {
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v0.position));
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v1.position));
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v2.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v0.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v1.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v2.position));
        }
        scene.meshRanges.push_back(mr);
    };

    // =========================================================================
    // 1. Build 16 Modular BLAS Geometry Prototypes
    // =========================================================================

    // Prototype 0: Tower Base Block (CYBER_BLAS_TOWER_BASE)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Foundation core block (Mat 0: Foundation Concrete)
        addBox(scene.triangles, glm::vec3(0.0f, 4.0f, 0.0f), glm::vec3(12.0f, 7.6f, 12.0f), 0.0f, 0);
        // Basalt plinth base step (Mat 8: Basalt / Rough Granite Foundation Blocks)
        addBox(scene.triangles, glm::vec3(0.0f, 0.2f, 0.0f), glm::vec3(13.6f, 0.4f, 13.6f), 0.0f, 8);
        // Secondary stepped plinth with anti-slip brass nosing (Mat 3: Brushed Brass)
        addBox(scene.triangles, glm::vec3(0.0f, 0.4f, 0.0f), glm::vec3(13.0f, 0.1f, 13.0f), 0.0f, 3);
        // 4 heavy corner columns (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(-5.2f, 4.0f, -5.2f), glm::vec3(2.0f, 8.0f, 2.0f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3( 5.2f, 4.0f, -5.2f), glm::vec3(2.0f, 8.0f, 2.0f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3(-5.2f, 4.0f,  5.2f), glm::vec3(2.0f, 8.0f, 2.0f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3( 5.2f, 4.0f,  5.2f), glm::vec3(2.0f, 8.0f, 2.0f), 0.0f, 1);
        // Corner column decorative fluting (Mat 2: Titanium Chrome)
        for (float sx : {-5.2f, 5.2f}) {
            for (float sz : {-5.2f, 5.2f}) {
                addBox(scene.triangles, glm::vec3(sx + (sx > 0.0f ? 1.05f : -1.05f), 4.0f, sz), glm::vec3(0.1f, 7.8f, 1.6f), 0.0f, 2);
                addBox(scene.triangles, glm::vec3(sx, 4.0f, sz + (sz > 0.0f ? 1.05f : -1.05f)), glm::vec3(1.6f, 7.8f, 0.1f), 0.0f, 2);
            }
        }
        // Diagrid corner trusses (Mat 27: Galvanized Steel)
        addDiagridLattice(scene.triangles, glm::vec3(-5.8f, 0.4f, 5.8f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 3, 2.0f, 7.2f, 27);
        addDiagridLattice(scene.triangles, glm::vec3( 3.8f, 0.4f, 5.8f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 3, 2.0f, 7.2f, 27);
        addDiagridLattice(scene.triangles, glm::vec3(-5.8f, 0.4f, -5.8f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 3, 2.0f, 7.2f, 27);
        addDiagridLattice(scene.triangles, glm::vec3( 3.8f, 0.4f, -5.8f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 3, 2.0f, 7.2f, 27);
        // Entrance portals on all 4 facades (Mat 3: Brushed Brass, Mat 29: Polished Jade trim)
        addBox(scene.triangles, glm::vec3( 0.0f, 2.7f,  6.1f), glm::vec3(4.5f, 4.6f, 0.4f), 0.0f, 3);
        addBox(scene.triangles, glm::vec3( 0.0f, 2.7f, -6.1f), glm::vec3(4.5f, 4.6f, 0.4f), 0.0f, 3);
        addBox(scene.triangles, glm::vec3( 6.1f, 2.7f,  0.0f), glm::vec3(0.4f, 4.6f, 4.5f), 0.0f, 3);
        addBox(scene.triangles, glm::vec3(-6.1f, 2.7f,  0.0f), glm::vec3(0.4f, 4.6f, 4.5f), 0.0f, 3);
        addBox(scene.triangles, glm::vec3( 0.0f, 4.8f,  6.15f), glm::vec3(4.8f, 0.3f, 0.2f), 0.0f, 29);
        addBox(scene.triangles, glm::vec3( 0.0f, 4.8f, -6.15f), glm::vec3(4.8f, 0.3f, 0.2f), 0.0f, 29);
        addBox(scene.triangles, glm::vec3( 6.15f, 4.8f,  0.0f), glm::vec3(0.2f, 0.3f, 4.8f), 0.0f, 29);
        addBox(scene.triangles, glm::vec3(-6.15f, 4.8f,  0.0f), glm::vec3(0.2f, 0.3f, 4.8f), 0.0f, 29);
        // Cantilevered glass entrance canopies (Mat 6: Cyan Canopy Glass, Mat 2: Titanium frame)
        addBox(scene.triangles, glm::vec3( 0.0f, 4.95f,  6.8f), glm::vec3(4.6f, 0.08f, 1.4f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3( 0.0f, 4.95f, -6.8f), glm::vec3(4.6f, 0.08f, 1.4f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3( 6.8f, 4.95f,  0.0f), glm::vec3(1.4f, 0.08f, 4.6f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3(-6.8f, 4.95f,  0.0f), glm::vec3(1.4f, 0.08f, 4.6f), 0.0f, 6);
        // Revolving entrance door cylinders with Crown Glass (Mat 5: Dielectric Crown Glass)
        addCylinder(scene.triangles, glm::vec3( 0.0f, 1.8f,  6.0f), 1.2f, 2.8f, 5, 12);
        addCylinder(scene.triangles, glm::vec3( 0.0f, 1.8f, -6.0f), 1.2f, 2.8f, 5, 12);
        addCylinder(scene.triangles, glm::vec3( 6.0f, 1.8f,  0.0f), 1.2f, 2.8f, 5, 12);
        addCylinder(scene.triangles, glm::vec3(-6.0f, 1.8f,  0.0f), 1.2f, 2.8f, 5, 12);
        // Lower facade window grids: 4 facades x (4 rows x 5 cols = 20 windows per facade)
        // (Mat 1: Steel framing, Mat 7: Smoked Obsidian glass, Mat 41: Warm Sodium interior glow, Mat 18: Rose gold sill, Mat 35: Gold accent)
        addWindowGrid(scene.triangles, glm::vec3(-4.6f, 1.2f,  6.08f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 4, 5, 9.2f, 4.8f, 1, 7, 41, 101, 18, 35);
        addWindowGrid(scene.triangles, glm::vec3(-4.6f, 1.2f, -6.08f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 4, 5, 9.2f, 4.8f, 1, 7, 41, 103, 18, 35);
        addWindowGrid(scene.triangles, glm::vec3( 6.08f, 1.2f, -4.6f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 4, 5, 9.2f, 4.8f, 1, 7, 41, 107, 18, 35);
        addWindowGrid(scene.triangles, glm::vec3(-6.08f, 1.2f, -4.6f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 4, 5, 9.2f, 4.8f, 1, 7, 41, 109, 18, 35);
        // Emissive entrance beacon strips (Mat 45: Electric Turquoise 490nm, Mat 35: Electric Gold)
        addBox(scene.triangles, glm::vec3( 0.0f, 2.36f,  6.25f), glm::vec3(3.2f, 3.88f, 0.12f), 0.0f, 45);
        addBox(scene.triangles, glm::vec3( 0.0f, 2.36f, -6.25f), glm::vec3(3.2f, 3.88f, 0.12f), 0.0f, 45);
        addBox(scene.triangles, glm::vec3( 6.25f, 2.36f,  0.0f), glm::vec3(0.12f, 3.88f, 3.2f), 0.0f, 45);
        addBox(scene.triangles, glm::vec3(-6.25f, 2.36f,  0.0f), glm::vec3(0.12f, 3.88f, 3.2f), 0.0f, 45);
        addBox(scene.triangles, glm::vec3( 0.0f, 0.42f,  6.25f), glm::vec3(3.4f, 0.10f, 0.12f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3( 0.0f, 0.42f, -6.25f), glm::vec3(3.4f, 0.10f, 0.12f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3( 6.25f, 0.42f,  0.0f), glm::vec3(0.12f, 0.10f, 3.4f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3(-6.25f, 0.42f,  0.0f), glm::vec3(0.12f, 0.10f, 3.4f), 0.0f, 35);
        recordBlasPrototype("Tower Base Block", tStart);
    }

    // Prototype 1: Tower Mid Module A (Curtain Wall & Aerodynamic Fins)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Core elevator shaft (Mat 22: Matte Charcoal Composite)
        addBox(scene.triangles, glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(9.8f, 5.8f, 9.8f), 0.0f, 22);
        // Floor collar band (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 0.2f, 0.0f), glm::vec3(10.6f, 0.4f, 10.6f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3(0.0f, 5.8f, 0.0f), glm::vec3(10.6f, 0.4f, 10.6f), 0.0f, 1);
        // 4 Exterior vertical aerodynamic fins (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3( 0.0f, 3.0f,  5.2f), glm::vec3(0.8f, 6.0f, 0.6f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3( 0.0f, 3.0f, -5.2f), glm::vec3(0.8f, 6.0f, 0.6f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3( 5.2f, 3.0f,  0.0f), glm::vec3(0.6f, 6.0f, 0.8f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3(-5.2f, 3.0f,  0.0f), glm::vec3(0.6f, 6.0f, 0.8f), 0.0f, 2);
        // Horizontal aerodynamic wing strakes (Mat 18: Rose Gold)
        addBox(scene.triangles, glm::vec3( 0.0f, 3.0f,  5.5f), glm::vec3(4.0f, 0.12f, 0.4f), 0.0f, 18);
        addBox(scene.triangles, glm::vec3( 0.0f, 3.0f, -5.5f), glm::vec3(4.0f, 0.12f, 0.4f), 0.0f, 18);
        addBox(scene.triangles, glm::vec3( 5.5f, 3.0f,  0.0f), glm::vec3(0.4f, 0.12f, 4.0f), 0.0f, 18);
        addBox(scene.triangles, glm::vec3(-5.5f, 3.0f,  0.0f), glm::vec3(0.4f, 0.12f, 4.0f), 0.0f, 18);
        // Vertical neon circuit conduits (Mat 35: Electric Gold)
        addBox(scene.triangles, glm::vec3(-4.95f, 3.0f, -4.95f), glm::vec3(0.2f, 6.0f, 0.2f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3( 4.95f, 3.0f,  4.95f), glm::vec3(0.2f, 6.0f, 0.2f), 0.0f, 35);
        // External copper conduit raceways (Mat 10: Copper Conduits)
        addBox(scene.triangles, glm::vec3( 4.95f, 3.0f, -4.95f), glm::vec3(0.15f, 6.0f, 0.15f), 0.0f, 10);
        addBox(scene.triangles, glm::vec3(-4.95f, 3.0f,  4.95f), glm::vec3(0.15f, 6.0f, 0.15f), 0.0f, 10);
        // Dense Curtain Wall Window Grids on 4 facades (6 rows x 6 cols = 36 windows per facade)
        // (Mat 17: Anisotropic Platinum mullions, Mat 16: Iridescent Coated Glass, Mat 30: Neon Cyan office glow, Mat 18: Rose Gold sill, Mat 31: Magenta accent)
        addWindowGrid(scene.triangles, glm::vec3(-4.5f, 0.5f,  4.98f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 6, 9.0f, 5.0f, 17, 16, 30, 201, 18, 31);
        addWindowGrid(scene.triangles, glm::vec3(-4.5f, 0.5f, -4.98f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 6, 9.0f, 5.0f, 17, 16, 30, 203, 18, 31);
        addWindowGrid(scene.triangles, glm::vec3( 4.98f, 0.5f, -4.5f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 6, 9.0f, 5.0f, 17, 16, 30, 205, 18, 31);
        addWindowGrid(scene.triangles, glm::vec3(-4.98f, 0.5f, -4.5f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 6, 9.0f, 5.0f, 17, 16, 30, 207, 18, 31);
        recordBlasPrototype("Tower Mid Module A", tStart);
    }

    // Prototype 2: Tower Mid Module B (Terraced Garden & Exoskeleton)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Cantilever terrace slab (Mat 21: High-Gloss Ceramic, Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(11.8f, 1.0f, 11.8f), 0.0f, 21);
        addBox(scene.triangles, glm::vec3(0.0f, 0.05f, 0.0f), glm::vec3(11.9f, 0.1f, 11.9f), 0.0f, 1);
        // Recessed core (Mat 19: Gunmetal Alloy Armor)
        addBox(scene.triangles, glm::vec3(0.0f, 3.5f, 0.0f), glm::vec3(8.8f, 5.0f, 8.8f), 0.0f, 19);
        // 4 Terrace structural pillars (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(-5.2f, 3.0f, -5.2f), glm::vec3(0.8f, 5.0f, 0.8f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3( 5.2f, 3.0f, -5.2f), glm::vec3(0.8f, 5.0f, 0.8f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3(-5.2f, 3.0f,  5.2f), glm::vec3(0.8f, 5.0f, 0.8f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3( 5.2f, 3.0f,  5.2f), glm::vec3(0.8f, 5.0f, 0.8f), 0.0f, 2);
        // Balustrade glass railings (Mat 6: Cyan Glass) & Rose Gold trim (Mat 18)
        addBox(scene.triangles, glm::vec3( 0.0f, 1.6f,  5.85f), glm::vec3(11.4f, 1.2f, 0.08f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3( 0.0f, 1.6f, -5.85f), glm::vec3(11.4f, 1.2f, 0.08f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3( 5.85f, 1.6f,  0.0f), glm::vec3(0.08f, 1.2f, 11.4f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3(-5.85f, 1.6f,  0.0f), glm::vec3(0.08f, 1.2f, 11.4f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3( 0.0f, 2.22f,  5.85f), glm::vec3(11.5f, 0.06f, 0.12f), 0.0f, 18);
        addBox(scene.triangles, glm::vec3( 0.0f, 2.22f, -5.85f), glm::vec3(11.5f, 0.06f, 0.12f), 0.0f, 18);
        addBox(scene.triangles, glm::vec3( 5.85f, 2.22f,  0.0f), glm::vec3(0.12f, 0.06f, 11.5f), 0.0f, 18);
        addBox(scene.triangles, glm::vec3(-5.85f, 2.22f,  0.0f), glm::vec3(0.12f, 0.06f, 11.5f), 0.0f, 18);
        // Planter boxes with Jade trim (Mat 29) and White Ceramic (Mat 9)
        addBox(scene.triangles, glm::vec3(-3.8f, 1.3f,  5.0f), glm::vec3(2.2f, 0.6f, 0.8f), 0.0f, 9);
        addBox(scene.triangles, glm::vec3(-3.8f, 1.65f, 5.0f), glm::vec3(2.3f, 0.1f, 0.9f), 0.0f, 29);
        addBox(scene.triangles, glm::vec3( 3.8f, 1.3f, -5.0f), glm::vec3(2.2f, 0.6f, 0.8f), 0.0f, 9);
        addBox(scene.triangles, glm::vec3( 3.8f, 1.65f, -5.0f), glm::vec3(2.3f, 0.1f, 0.9f), 0.0f, 29);
        // Corner Dispersive Diamond Prism sculptures (Mat 15: Dispersive Diamond) and Emerald Crystal finials (Mat 13: Emerald)
        addSphere(scene.triangles, glm::vec3(-5.4f, 2.6f, -5.4f), 0.35f, 15, 14, 14);
        addSphere(scene.triangles, glm::vec3( 5.4f, 2.6f,  5.4f), 0.35f, 15, 14, 14);
        addSphere(scene.triangles, glm::vec3(-5.4f, 2.6f,  5.4f), 0.35f, 13, 14, 14);
        addSphere(scene.triangles, glm::vec3( 5.4f, 2.6f, -5.4f), 0.35f, 13, 14, 14);
        // Frosted Amber privacy screens (Mat 12: Frosted Amber Glass)
        addBox(scene.triangles, glm::vec3(-4.0f, 2.8f,  4.6f), glm::vec3(1.8f, 3.2f, 0.06f), 0.0f, 12);
        addBox(scene.triangles, glm::vec3( 4.0f, 2.8f, -4.6f), glm::vec3(1.8f, 3.2f, 0.06f), 0.0f, 12);
        // Diagrid Exoskeleton Trusses on East & West facades (Mat 27: Galvanized Steel)
        addDiagridLattice(scene.triangles, glm::vec3(-4.2f, 1.0f,  4.5f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 8.4f, 4.8f, 27);
        addDiagridLattice(scene.triangles, glm::vec3(-4.2f, 1.0f, -4.5f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 8.4f, 4.8f, 27);
        // Window grids on North & South recessed core (6 rows x 6 cols)
        addWindowGrid(scene.triangles, glm::vec3( 4.45f, 1.2f, -4.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 6, 8.0f, 4.5f, 1, 6, 31, 301, 10, 38);
        addWindowGrid(scene.triangles, glm::vec3(-4.45f, 1.2f, -4.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 6, 8.0f, 4.5f, 1, 6, 32, 303, 10, 38);
        // Neon perimeter rim strips (Mat 31: Magenta, Mat 32: Orange)
        addBox(scene.triangles, glm::vec3(0.0f, 0.8f,  5.95f), glm::vec3(11.6f, 0.2f, 0.1f), 0.0f, 31);
        addBox(scene.triangles, glm::vec3(0.0f, 0.8f, -5.95f), glm::vec3(11.6f, 0.2f, 0.1f), 0.0f, 32);
        recordBlasPrototype("Tower Mid Module B", tStart);
    }

    // Prototype 3: Tower Mid Module C (High-Tech Industrial Core)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Heavy carbon fiber sheathing core (Mat 19: Gunmetal Alloy Armor)
        addBox(scene.triangles, glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(10.2f, 5.8f, 10.2f), 0.0f, 19);
        // Floor collar (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 0.2f, 0.0f), glm::vec3(10.8f, 0.4f, 10.8f), 0.0f, 1);
        // Exposed acoustic/ventilation mesh grilles (Mat 28: Perforated Mesh)
        addBox(scene.triangles, glm::vec3( 0.0f, 4.5f,  5.18f), glm::vec3(7.2f, 1.8f, 0.12f), 0.0f, 28);
        addBox(scene.triangles, glm::vec3( 0.0f, 4.5f, -5.18f), glm::vec3(7.2f, 1.8f, 0.12f), 0.0f, 28);
        addBox(scene.triangles, glm::vec3( 5.18f, 4.5f,  0.0f), glm::vec3(0.12f, 1.8f, 7.2f), 0.0f, 28);
        addBox(scene.triangles, glm::vec3(-5.18f, 4.5f,  0.0f), glm::vec3(0.12f, 1.8f, 7.2f), 0.0f, 28);
        // High-voltage copper busbars (Mat 10: Copper Conduits)
        addBox(scene.triangles, glm::vec3(-4.8f, 3.0f,  5.22f), glm::vec3(0.25f, 5.8f, 0.15f), 0.0f, 10);
        addBox(scene.triangles, glm::vec3( 4.8f, 3.0f,  5.22f), glm::vec3(0.25f, 5.8f, 0.15f), 0.0f, 10);
        addBox(scene.triangles, glm::vec3(-4.8f, 3.0f, -5.22f), glm::vec3(0.25f, 5.8f, 0.15f), 0.0f, 10);
        addBox(scene.triangles, glm::vec3( 4.8f, 3.0f, -5.22f), glm::vec3(0.25f, 5.8f, 0.15f), 0.0f, 10);
        // Server window banks (6 rows x 6 cols): Mat 17 (Chromium), Mat 11 (Smoked Obsidian), Mat 33 (Acid Green status), Mat 2 (Titanium sill), Mat 45 (Turquoise accent)
        addWindowGrid(scene.triangles, glm::vec3(-4.2f, 0.8f,  5.15f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 6, 8.4f, 3.2f, 17, 11, 33, 401, 2, 45);
        addWindowGrid(scene.triangles, glm::vec3(-4.2f, 0.8f, -5.15f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 6, 8.4f, 3.2f, 17, 11, 33, 403, 2, 45);
        addWindowGrid(scene.triangles, glm::vec3( 5.15f, 0.8f, -4.2f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 6, 8.4f, 3.2f, 17, 11, 33, 405, 2, 45);
        addWindowGrid(scene.triangles, glm::vec3(-5.15f, 0.8f, -4.2f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 6, 8.4f, 3.2f, 17, 11, 33, 407, 2, 45);
        // Crimson strobe safety beacon (Mat 44) & UV Blacklight glow node (Mat 40)
        addBox(scene.triangles, glm::vec3( 5.15f, 5.6f,  5.15f), glm::vec3(0.3f, 0.3f, 0.3f), 0.0f, 44);
        addBox(scene.triangles, glm::vec3(-5.15f, 5.6f, -5.15f), glm::vec3(0.3f, 0.3f, 0.3f), 0.0f, 44);
        addSphere(scene.triangles, glm::vec3( 5.15f, 0.4f, -5.15f), 0.25f, 40, 10, 10);
        addSphere(scene.triangles, glm::vec3(-5.15f, 0.4f,  5.15f), 0.25f, 40, 10, 10);
        recordBlasPrototype("Tower Mid Module C", tStart);
    }

    // Prototype 4: Tower Crown & Spire
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Penthouse roof base (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(9.6f, 2.0f, 9.6f), 0.0f, 1);
        // Angular pyramid frustum (Mat 17: Anisotropic Brushed Platinum)
        addFrustum(scene.triangles, glm::vec3(0.0f, 3.0f, 0.0f), glm::vec2(9.6f), glm::vec2(5.4f), 2.0f, 17);
        // Mezzanine deck (Mat 3: Brushed Brass, Mat 18: Rose gold border)
        addBox(scene.triangles, glm::vec3(0.0f, 4.2f, 0.0f), glm::vec3(5.6f, 0.4f, 5.6f), 0.0f, 3);
        addBox(scene.triangles, glm::vec3(0.0f, 4.42f, 0.0f), glm::vec3(5.8f, 0.08f, 5.8f), 0.0f, 18);
        // Geodesic observation dome (Mat 15: Dispersive Diamond Prism, IOR 2.42, dispersion 0.08)
        addSphere(scene.triangles, glm::vec3(0.0f, 6.2f, 0.0f), 2.4f, 15, 28, 28);
        // Iridescent observation ring deck (Mat 16: Iridescent AR Coating)
        addCylinder(scene.triangles, glm::vec3(0.0f, 4.45f, 0.0f), 2.8f, 0.10f, 16, 24);
        // Emerald crystal finials (Mat 13: Emerald Crystal Spire)
        addSphere(scene.triangles, glm::vec3(-2.2f, 4.6f, -2.2f), 0.4f, 13, 14, 14);
        addSphere(scene.triangles, glm::vec3( 2.2f, 4.6f, -2.2f), 0.4f, 13, 14, 14);
        addSphere(scene.triangles, glm::vec3(-2.2f, 4.6f,  2.2f), 0.4f, 13, 14, 14);
        addSphere(scene.triangles, glm::vec3( 2.2f, 4.6f,  2.2f), 0.4f, 13, 14, 14);
        // Interior core pedestal (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(0.0f, 5.2f, 0.0f), glm::vec3(0.8f, 1.0f, 0.8f), 0.0f, 2);
        // High-altitude antenna mast (Mat 2: Titanium Chrome)
        addCylinder(scene.triangles, glm::vec3(0.0f, 12.0f, 0.0f), 0.18f, 7.8f, 2, 16);
        // Structural cross arms (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 11.0f, 0.0f), glm::vec3(2.4f, 0.15f, 0.15f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3(0.0f, 13.5f, 0.0f), glm::vec3(1.6f, 0.15f, 0.15f), 0.0f, 1);
        // Dipole antenna arrays on cross arms (Mat 10: Copper Conduits)
        for (float dx : {-1.0f, 1.0f}) {
            addCylinder(scene.triangles, glm::vec3(dx, 11.0f, 0.0f), 0.03f, 1.2f, 10, 8);
            addCylinder(scene.triangles, glm::vec3(dx * 0.7f, 13.5f, 0.0f), 0.03f, 0.8f, 10, 8);
        }
        // Satellite communications dishes (Mat 19: Gunmetal Alloy, Mat 3: Brass feed)
        addCylinder(scene.triangles, glm::vec3(0.8f, 8.5f, 0.8f), 0.6f, 0.2f, 19, 14);
        addBox(scene.triangles, glm::vec3(0.8f, 8.5f, 1.0f), glm::vec3(0.08f, 0.08f, 0.25f), 0.0f, 3);
        // Xenon floodlight fixtures (Mat 42: Xenon White)
        addBox(scene.triangles, glm::vec3( 1.1f, 10.9f, 0.0f), glm::vec3(0.2f, 0.2f, 0.2f), 0.0f, 42);
        addBox(scene.triangles, glm::vec3(-1.1f, 10.9f, 0.0f), glm::vec3(0.2f, 0.2f, 0.2f), 0.0f, 42);
        // Ruby laser dielectric focusing core (Mat 14: Ruby Laser Optical Glass)
        addSphere(scene.triangles, glm::vec3(0.0f, 16.0f, 0.0f), 0.35f, 14, 14, 14);
        // Ruby laser beacon tip (Mat 34: Ruby Laser Warning Beacon)
        addBox(scene.triangles, glm::vec3(0.0f, 16.4f, 0.0f), glm::vec3(0.3f, 0.3f, 0.3f), 0.0f, 34);
        recordBlasPrototype("Tower Crown & Spire", tStart);
    }

    // Prototype 5: Rooftop HVAC & Industrial Machinery Pod
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Corten steel mounting skid (Mat 21: Weathered Rusted Iron)
        addBox(scene.triangles, glm::vec3(0.0f, 0.4f, 0.0f), glm::vec3(8.0f, 0.8f, 6.0f), 0.0f, 21);
        // Dual large chiller cylinders (Mat 1: Carbon Steel)
        addCylinder(scene.triangles, glm::vec3(-2.2f, 2.2f, 0.0f), 1.4f, 3.2f, 1, 16);
        addCylinder(scene.triangles, glm::vec3( 2.2f, 2.2f, 0.0f), 1.4f, 3.2f, 1, 16);
        // 8 Radial fan blades per chiller (Mat 2: Titanium Chrome)
        for (int b = 0; b < 8; ++b) {
            float angle = static_cast<float>(b) * 45.0f;
            addBox(scene.triangles, glm::vec3(-2.2f, 3.75f, 0.0f), glm::vec3(0.1f, 0.05f, 2.2f), angle, 2);
            addBox(scene.triangles, glm::vec3( 2.2f, 3.75f, 0.0f), glm::vec3(0.1f, 0.05f, 2.2f), angle, 2);
        }
        // Perforated ventilation exhaust intakes (Mat 28: Perforated Mesh)
        addBox(scene.triangles, glm::vec3(-2.2f, 3.9f, 0.0f), glm::vec3(2.4f, 0.3f, 2.4f), 0.0f, 28);
        addBox(scene.triangles, glm::vec3( 2.2f, 3.9f, 0.0f), glm::vec3(2.4f, 0.3f, 2.4f), 0.0f, 28);
        // Polished copper pipe manifolds (Mat 10: Copper Conduits & Busbars)
        addCylinder(scene.triangles, glm::vec3(0.0f, 2.0f, -1.8f), 0.25f, 5.2f, 10, 12);
        addCylinder(scene.triangles, glm::vec3(0.0f, 2.6f, -1.8f), 0.25f, 5.2f, 10, 12);
        // Emergency pressure relief valve wheels (Mat 24: High-Gloss Yellow Hazard)
        addCylinder(scene.triangles, glm::vec3(-1.0f, 2.9f, -1.8f), 0.3f, 0.08f, 24, 12);
        addCylinder(scene.triangles, glm::vec3( 1.0f, 2.9f, -1.8f), 0.3f, 0.08f, 24, 12);
        // Gunmetal ductwork housing (Mat 19: Gunmetal Alloy Armor)
        addBox(scene.triangles, glm::vec3(0.0f, 1.8f, 1.8f), glm::vec3(3.2f, 2.2f, 1.8f), 0.0f, 19);
        // Amber hazard flasher (Mat 43: Amber Hazard)
        addBox(scene.triangles, glm::vec3(-3.6f, 1.0f, -2.6f), glm::vec3(0.3f, 0.4f, 0.3f), 0.0f, 43);
        addBox(scene.triangles, glm::vec3( 3.6f, 1.0f,  2.6f), glm::vec3(0.3f, 0.4f, 0.3f), 0.0f, 43);
        recordBlasPrototype("Rooftop HVAC", tStart);
    }

    // Prototype 6: Glass Skybridge (CYBER_BLAS_SKYBRIDGE)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Structural end collars (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(-7.8f, 1.6f, 0.0f), glm::vec3(0.6f, 3.4f, 3.8f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3( 7.8f, 1.6f, 0.0f), glm::vec3(0.6f, 3.4f, 3.8f), 0.0f, 1);
        // 4 Titanium longitudinal beams (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(0.0f, 0.12f, -1.75f), glm::vec3(15.2f, 0.25f, 0.25f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3(0.0f, 0.12f,  1.75f), glm::vec3(15.2f, 0.25f, 0.25f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3(0.0f, 3.08f, -1.75f), glm::vec3(15.2f, 0.25f, 0.25f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3(0.0f, 3.08f,  1.75f), glm::vec3(15.2f, 0.25f, 0.25f), 0.0f, 2);
        // Diagrid truss sides along both flanks with gusset nodes (Mat 27: Galvanized Steel, Mat 3: Brass node)
        addDiagridLattice(scene.triangles, glm::vec3(-7.2f, 0.2f,  1.74f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 8, 14.4f, 2.8f, 27, 3);
        addDiagridLattice(scene.triangles, glm::vec3(-7.2f, 0.2f, -1.74f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 8, 14.4f, 2.8f, 27, 3);
        // Floor and ceiling structural crown glass (Mat 5: Transmissive Crown Glass)
        addBox(scene.triangles, glm::vec3(0.0f, 0.20f, 0.0f), glm::vec3(15.0f, 0.08f, 3.2f), 0.0f, 5);
        addBox(scene.triangles, glm::vec3(0.0f, 3.00f, 0.0f), glm::vec3(15.0f, 0.08f, 3.2f), 0.0f, 5);
        // Sidewall transmissive panels (Mat 6: Cyan Canopy Glass)
        addBox(scene.triangles, glm::vec3(0.0f, 1.60f,  1.70f), glm::vec3(15.0f, 2.6f, 0.08f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3(0.0f, 1.60f, -1.70f), glm::vec3(15.0f, 2.6f, 0.08f), 0.0f, 6);
        // Interior walkway lounge carpet (Mat 23: Velvet Sheen)
        addBox(scene.triangles, glm::vec3(0.0f, 0.25f, 0.0f), glm::vec3(14.8f, 0.02f, 1.8f), 0.0f, 23);
        // Interior rose gold handrails (Mat 18: Rose Gold Filigree)
        addBox(scene.triangles, glm::vec3(0.0f, 1.05f, -1.55f), glm::vec3(14.8f, 0.06f, 0.06f), 0.0f, 18);
        addBox(scene.triangles, glm::vec3(0.0f, 1.05f,  1.55f), glm::vec3(14.8f, 0.06f, 0.06f), 0.0f, 18);
        // Interior neon guide strips (Mat 45: Electric Turquoise)
        addBox(scene.triangles, glm::vec3(0.0f, 0.26f, -0.9f), glm::vec3(14.8f, 0.04f, 0.15f), 0.0f, 45);
        addBox(scene.triangles, glm::vec3(0.0f, 0.26f,  0.9f), glm::vec3(14.8f, 0.04f, 0.15f), 0.0f, 45);
        // Ceiling recessed xenon troffer lights (Mat 42: Xenon White)
        for (int i = -3; i <= 3; ++i) {
            float xpos = static_cast<float>(i) * 2.0f;
            addBox(scene.triangles, glm::vec3(xpos, 2.95f, 0.0f), glm::vec3(1.2f, 0.04f, 0.6f), 0.0f, 42);
        }
        // Under-bridge electric turquoise display glow strip (Mat 45: Electric Turquoise)
        addBox(scene.triangles, glm::vec3(0.0f, -0.05f, 0.0f), glm::vec3(15.0f, 0.10f, 0.40f), 0.0f, 45);
        recordBlasPrototype("Glass Skybridge", tStart);
    }

    // Prototype 7: Comm Gantry & Microwave Relay (CYBER_BLAS_COMM_GANTRY)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Structural mast (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 4.0f, 0.0f), glm::vec3(1.6f, 8.0f, 1.6f), 0.0f, 1);
        // Diagrid lattice bracing (Mat 27: Galvanized Steel, Mat 3: Brass node)
        addDiagridLattice(scene.triangles, glm::vec3(-0.9f, 0.0f, 0.9f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 2, 1.8f, 7.8f, 27, 3);
        // Flange platforms (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(2.4f, 0.2f, 2.4f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3(0.0f, 6.0f, 0.0f), glm::vec3(2.0f, 0.2f, 2.0f), 0.0f, 2);
        // Microwave receiver dishes (Mat 19: Gunmetal Alloy Armor)
        addCylinder(scene.triangles, glm::vec3(0.0f, 6.5f, 1.0f), 0.9f, 0.3f, 19, 16);
        addCylinder(scene.triangles, glm::vec3(1.0f, 4.5f, 0.0f), 0.7f, 0.3f, 19, 16);
        // Gold plated transceivers (Mat 18: Rose Gold Filigree)
        addBox(scene.triangles, glm::vec3(0.0f, 6.5f, 1.3f), glm::vec3(0.18f, 0.18f, 0.4f), 0.0f, 18);
        addBox(scene.triangles, glm::vec3(1.3f, 4.5f, 0.0f), glm::vec3(0.4f, 0.18f, 0.18f), 0.0f, 18);
        // Ultraviolet blacklight holo-relay node (Mat 40: Ultraviolet Blacklight Glow)
        addSphere(scene.triangles, glm::vec3(0.0f, 7.5f, 0.0f), 0.45f, 40, 14, 14);
        // Top strobe beacon (Mat 44: Crimson Strobe)
        addBox(scene.triangles, glm::vec3(0.0f, 8.25f, 0.0f), glm::vec3(0.35f, 0.4f, 0.35f), 0.0f, 44);
        recordBlasPrototype("Comm Gantry", tStart);
    }

    // Prototype 8: Elevated Maglev Transit Guideway (CYBER_BLAS_TRANSIT_GUIDEWAY)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Foundation concrete support girder (Mat 0: Foundation Concrete)
        addBox(scene.triangles, glm::vec3(0.0f, 0.3f, 0.0f), glm::vec3(16.0f, 0.6f, 4.8f), 0.0f, 0);
        // High-traction rubber guideway bed (Mat 22: Rubber Transit Dampers)
        addBox(scene.triangles, glm::vec3(0.0f, 0.62f, 0.0f), glm::vec3(16.0f, 0.06f, 4.6f), 0.0f, 22);
        // Matte black stealth shoulders (Mat 25: Matte Black Stealth Paneling)
        addBox(scene.triangles, glm::vec3(0.0f, 0.64f, -2.0f), glm::vec3(16.0f, 0.04f, 0.6f), 0.0f, 25);
        addBox(scene.triangles, glm::vec3(0.0f, 0.64f,  2.0f), glm::vec3(16.0f, 0.04f, 0.6f), 0.0f, 25);
        // Dual electromagnetic levitation rails (Mat 35: Electric Gold Emissive Conduit)
        addBox(scene.triangles, glm::vec3(0.0f, 0.72f, -1.1f), glm::vec3(16.0f, 0.12f, 0.4f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3(0.0f, 0.72f,  1.1f), glm::vec3(16.0f, 0.12f, 0.4f), 0.0f, 35);
        // Center copper power rail (Mat 10: Copper Conduits & Busbars)
        addBox(scene.triangles, glm::vec3(0.0f, 0.68f, 0.0f), glm::vec3(16.0f, 0.08f, 0.3f), 0.0f, 10);
        // Titanium aerodynamic crash barriers (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(0.0f, 0.95f, -2.35f), glm::vec3(16.0f, 0.8f, 0.2f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3(0.0f, 0.95f,  2.35f), glm::vec3(16.0f, 0.8f, 0.2f), 0.0f, 2);
        // High-gloss yellow hazard stripes on crash barriers (Mat 24: High-Gloss Yellow Hazard)
        for (int i = -3; i <= 3; ++i) {
            float xpos = static_cast<float>(i) * 2.2f;
            addBox(scene.triangles, glm::vec3(xpos, 0.95f, -2.24f), glm::vec3(0.8f, 0.3f, 0.03f), 0.0f, 24);
            addBox(scene.triangles, glm::vec3(xpos, 0.95f,  2.24f), glm::vec3(0.8f, 0.3f, 0.03f), 0.0f, 24);
        }
        // Electric turquoise guideway edge markers (Mat 45: Electric Turquoise)
        addBox(scene.triangles, glm::vec3(0.0f, 0.98f, -2.22f), glm::vec3(16.0f, 0.08f, 0.06f), 0.0f, 45);
        addBox(scene.triangles, glm::vec3(0.0f, 0.98f,  2.22f), glm::vec3(16.0f, 0.08f, 0.06f), 0.0f, 45);
        // Cantilever support arms underneath (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, -0.4f, 0.0f), glm::vec3(1.2f, 0.8f, 4.4f), 0.0f, 1);
        recordBlasPrototype("Transit Guideway", tStart);
    }

    // Prototype 9: Autonomous Aerodynamic Sky Cab (CYBER_BLAS_SKY_CAB)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Stealth composite chassis (Mat 25: Matte Black Stealth Paneling)
        addBox(scene.triangles, glm::vec3(0.0f, 0.4f, 0.0f), glm::vec3(4.2f, 0.7f, 2.0f), 0.0f, 25);
        // Titanium aerodynamic canards and delta winglets (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(-0.4f, 0.38f, 0.0f), glm::vec3(2.4f, 0.12f, 3.4f), 0.0f, 2);
        // Smoked obsidian glass teardrop cockpit canopy (Mat 11: Smoked Obsidian Glass)
        addSphere(scene.triangles, glm::vec3(0.6f, 0.85f, 0.0f), 0.82f, 11, 20, 20);
        // Interior HUD cockpit display (Mat 45: Electric Turquoise Display)
        addBox(scene.triangles, glm::vec3(0.7f, 0.72f, 0.0f), glm::vec3(0.15f, 0.15f, 0.35f), 0.0f, 45);
        // Gunmetal VTOL thruster nacelles (Mat 19: Gunmetal Alloy Armor)
        addCylinder(scene.triangles, glm::vec3(-1.8f, 0.45f, -1.1f), 0.38f, 0.8f, 19, 14);
        addCylinder(scene.triangles, glm::vec3(-1.8f, 0.45f,  1.1f), 0.38f, 0.8f, 19, 14);
        // Platinum stator vanes inside thrusters (Mat 17: Anisotropic Brushed Platinum)
        for (int v = 0; v < 4; ++v) {
            float vAngle = static_cast<float>(v) * 45.0f;
            addBox(scene.triangles, glm::vec3(-1.8f, 0.45f, -1.1f), glm::vec3(0.04f, 0.65f, 0.65f), vAngle, 17);
            addBox(scene.triangles, glm::vec3(-1.8f, 0.45f,  1.1f), glm::vec3(0.04f, 0.65f, 0.65f), vAngle, 17);
        }
        // Cyan plasma thruster exhaust rings (Mat 45: Electric Turquoise)
        addBox(scene.triangles, glm::vec3(-2.22f, 0.45f, -1.1f), glm::vec3(0.08f, 0.42f, 0.42f), 0.0f, 45);
        addBox(scene.triangles, glm::vec3(-2.22f, 0.45f,  1.1f), glm::vec3(0.08f, 0.42f, 0.42f), 0.0f, 45);
        // Ice blue forward transit headlights (Mat 47: Ice Blue Transit Headlight)
        addBox(scene.triangles, glm::vec3(2.05f, 0.38f, -0.65f), glm::vec3(0.18f, 0.18f, 0.25f), 0.0f, 47);
        addBox(scene.triangles, glm::vec3(2.05f, 0.38f,  0.65f), glm::vec3(0.18f, 0.18f, 0.25f), 0.0f, 47);
        // Port & Starboard navigation strobes (Mat 34 Red Port, Mat 33 Green Starboard)
        addBox(scene.triangles, glm::vec3(-0.4f, 0.45f, -1.72f), glm::vec3(0.12f, 0.12f, 0.12f), 0.0f, 34);
        addBox(scene.triangles, glm::vec3(-0.4f, 0.45f,  1.72f), glm::vec3(0.12f, 0.12f, 0.12f), 0.0f, 33);
        // Tail strobe beacon (Mat 42: Xenon White)
        addBox(scene.triangles, glm::vec3(-2.05f, 0.55f, 0.0f), glm::vec3(0.15f, 0.25f, 0.6f), 0.0f, 42);
        recordBlasPrototype("Sky Cab", tStart);
    }

    // Prototype 10: High-Rise Billboard Cyan (CYBER_BLAS_BILLBOARD_CYAN)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Rear armature & truss (Mat 1: Carbon Steel, Mat 27: Galvanized Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, -0.15f), glm::vec3(8.0f, 4.5f, 0.2f), 0.0f, 1);
        addDiagridLattice(scene.triangles, glm::vec3(-3.8f, -2.0f, -0.24f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 4, 7.6f, 4.0f, 27);
        // Gold border bezel (Mat 3: Brushed Brass)
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, 0.02f), glm::vec3(8.3f, 4.8f, 0.15f), 0.0f, 3);
        // Violet secondary accent border (Mat 37: Deep Cobalt)
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, 0.08f), glm::vec3(7.9f, 4.4f, 0.06f), 0.0f, 37);
        // Primary emissive quad (Mat 30: Neon Cyan 480nm)
        addQuad(scene.triangles,
                glm::vec3(-3.8f, -2.05f, 0.12f),
                glm::vec3( 3.8f, -2.05f, 0.12f),
                glm::vec3( 3.8f,  2.05f, 0.12f),
                glm::vec3(-3.8f,  2.05f, 0.12f),
                glm::vec3(0.0f, 0.0f, 1.0f), 30);
        recordBlasPrototype("Billboard Cyan", tStart);
    }

    // Prototype 11: Vertical Billboard Magenta (CYBER_BLAS_BILLBOARD_MAGENTA)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Frame (Mat 1: Carbon Steel, Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, -0.15f), glm::vec3(4.5f, 8.0f, 0.2f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, 0.02f), glm::vec3(4.8f, 8.3f, 0.15f), 0.0f, 2);
        // Hot pink secondary border (Mat 38: Hot Pink)
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, 0.08f), glm::vec3(4.4f, 7.9f, 0.06f), 0.0f, 38);
        // Primary emissive quad (Mat 31: Neon Magenta 650nm)
        addQuad(scene.triangles,
                glm::vec3(-2.05f, -3.8f, 0.12f),
                glm::vec3( 2.05f, -3.8f, 0.12f),
                glm::vec3( 2.05f,  3.8f, 0.12f),
                glm::vec3(-2.05f,  3.8f, 0.12f),
                glm::vec3(0.0f, 0.0f, 1.0f), 31);
        recordBlasPrototype("Billboard Magenta", tStart);
    }

    // Prototype 12: Panoramic Billboard Orange (CYBER_BLAS_BILLBOARD_ORANGE)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Frame (Mat 1: Carbon Steel, Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, -0.15f), glm::vec3(7.0f, 3.2f, 0.2f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, 0.02f), glm::vec3(7.3f, 3.5f, 0.15f), 0.0f, 2);
        // Lime yellow secondary border (Mat 46: Lime Yellow Neon)
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, 0.08f), glm::vec3(6.9f, 3.1f, 0.06f), 0.0f, 46);
        // Primary emissive quad (Mat 32: Neon Blaze Orange 600nm)
        addQuad(scene.triangles,
                glm::vec3(-3.3f, -1.45f, 0.12f),
                glm::vec3( 3.3f, -1.45f, 0.12f),
                glm::vec3( 3.3f,  1.45f, 0.12f),
                glm::vec3(-3.3f,  1.45f, 0.12f),
                glm::vec3(0.0f, 0.0f, 1.0f), 32);
        recordBlasPrototype("Billboard Orange", tStart);
    }

    // Prototype 13: Cyber Square Billboard Violet (CYBER_BLAS_BILLBOARD_VIOLET)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Frame (Mat 1: Carbon Steel, Mat 3: Brushed Brass)
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, -0.15f), glm::vec3(5.5f, 5.5f, 0.2f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, 0.02f), glm::vec3(5.8f, 5.8f, 0.15f), 0.0f, 3);
        // Deep cobalt secondary border (Mat 37: Deep Cobalt Blue)
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, 0.08f), glm::vec3(5.3f, 5.3f, 0.06f), 0.0f, 37);
        // Primary emissive quad (Mat 36: Deep Violet Holo 405nm)
        addQuad(scene.triangles,
                glm::vec3(-2.5f, -2.5f, 0.12f),
                glm::vec3( 2.5f, -2.5f, 0.12f),
                glm::vec3( 2.5f,  2.5f, 0.12f),
                glm::vec3(-2.5f,  2.5f, 0.12f),
                glm::vec3(0.0f, 0.0f, 1.0f), 36);
        recordBlasPrototype("Billboard Violet", tStart);
    }

    // Prototype 14: Rooftop Photovoltaic Solar Array (CYBER_BLAS_SOLAR_ROOF)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Galvanized steel rack structure (Mat 27: Galvanized Steel)
        addBox(scene.triangles, glm::vec3(-3.5f, 0.8f, -2.0f), glm::vec3(0.2f, 1.6f, 0.2f), 0.0f, 27);
        addBox(scene.triangles, glm::vec3( 3.5f, 0.8f, -2.0f), glm::vec3(0.2f, 1.6f, 0.2f), 0.0f, 27);
        addBox(scene.triangles, glm::vec3(-3.5f, 0.5f,  2.0f), glm::vec3(0.2f, 1.0f, 0.2f), 0.0f, 27);
        addBox(scene.triangles, glm::vec3( 3.5f, 0.5f,  2.0f), glm::vec3(0.2f, 1.0f, 0.2f), 0.0f, 27);
        addBox(scene.triangles, glm::vec3(0.0f, 1.5f, -2.0f), glm::vec3(7.2f, 0.15f, 0.15f), 0.0f, 27);
        addBox(scene.triangles, glm::vec3(0.0f, 0.9f,  2.0f), glm::vec3(7.2f, 0.15f, 0.15f), 0.0f, 27);
        // Array of 8 angled solar photovoltaic cell panels (Mat 26: Solar Photovoltaic Blue)
        for (int px = -3; px <= 3; px += 2) {
            float ox = static_cast<float>(px) * 1.0f;
            addBox(scene.triangles, glm::vec3(ox, 1.35f, -0.9f), glm::vec3(1.7f, 0.06f, 1.8f), -15.0f, 26);
            addBox(scene.triangles, glm::vec3(ox, 0.95f,  0.9f), glm::vec3(1.7f, 0.06f, 1.8f), -15.0f, 26);
        }
        // Central inverter housing (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 0.4f, 0.0f), glm::vec3(1.4f, 0.8f, 1.0f), 0.0f, 1);
        // Copper heatsink fins on inverter (Mat 10: Copper Conduits)
        for (float hx = -0.5f; hx <= 0.5f; hx += 0.25f) {
            addBox(scene.triangles, glm::vec3(hx, 0.85f, 0.0f), glm::vec3(0.04f, 0.15f, 0.8f), 0.0f, 10);
        }
        // Mint phosphor and amber status indicator lights (Mat 39: Mint Phosphor, Mat 43: Amber Hazard)
        addBox(scene.triangles, glm::vec3(-0.3f, 0.6f, 0.52f), glm::vec3(0.25f, 0.12f, 0.04f), 0.0f, 39);
        addBox(scene.triangles, glm::vec3( 0.3f, 0.6f, 0.52f), glm::vec3(0.25f, 0.12f, 0.04f), 0.0f, 43);
        recordBlasPrototype("Solar Roof", tStart);
    }

    // Prototype 15: Ground District Plaza Tile 16m x 16m (CYBER_BLAS_PLAZA_DISTRICT)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Foundation concrete bed (Mat 0: Foundation Concrete) with subsurface overlap
        addBox(scene.triangles, glm::vec3(0.0f, -0.35f, 0.0f), glm::vec3(16.2f, 0.7f, 16.2f), 0.0f, 0);
        // Wet reflective plaza pavement surface (Mat 4: Wet Reflective Pavement)
        addBox(scene.triangles, glm::vec3(0.0f, 0.02f, 0.0f), glm::vec3(16.0f, 0.04f, 16.0f), 0.0f, 4);
        // Rough basalt curb borders (Mat 8: Basalt Foundation)
        addBox(scene.triangles, glm::vec3( 7.9f, 0.08f, 0.0f), glm::vec3(0.2f, 0.16f, 16.0f), 0.0f, 8);
        addBox(scene.triangles, glm::vec3(-7.9f, 0.08f, 0.0f), glm::vec3(0.2f, 0.16f, 16.0f), 0.0f, 8);
        addBox(scene.triangles, glm::vec3(0.0f, 0.08f,  7.9f), glm::vec3(16.0f, 0.16f, 0.2f), 0.0f, 8);
        addBox(scene.triangles, glm::vec3(0.0f, 0.08f, -7.9f), glm::vec3(16.0f, 0.16f, 0.2f), 0.0f, 8);
        // Glazed ceramic planter boxes (Mat 9: White Ceramic Cladding) with Jade trim (Mat 29)
        addBox(scene.triangles, glm::vec3(-4.5f, 0.4f, -4.5f), glm::vec3(2.4f, 0.8f, 2.4f), 0.0f, 9);
        addBox(scene.triangles, glm::vec3(-4.5f, 0.82f, -4.5f), glm::vec3(2.5f, 0.08f, 2.5f), 0.0f, 29);
        addBox(scene.triangles, glm::vec3( 4.5f, 0.4f,  4.5f), glm::vec3(2.4f, 0.8f, 2.4f), 0.0f, 9);
        addBox(scene.triangles, glm::vec3( 4.5f, 0.82f,  4.5f), glm::vec3(2.5f, 0.08f, 2.5f), 0.0f, 29);
        // Ornamental crystal art sculptures inside planters (Mat 13: Emerald, Mat 14: Ruby)
        addSphere(scene.triangles, glm::vec3(-4.5f, 1.15f, -4.5f), 0.35f, 13, 12, 12);
        addSphere(scene.triangles, glm::vec3( 4.5f, 1.15f,  4.5f), 0.35f, 14, 12, 12);
        // Recessed ground luminescent and pedestrian guidance strips
        // (Mat 45: Electric Turquoise 490nm, Mat 35: Electric Gold)
        addBox(scene.triangles, glm::vec3(0.0f, 0.03f, 0.0f), glm::vec3(15.6f, 0.02f, 0.25f), 0.0f, 45);
        addBox(scene.triangles, glm::vec3(0.0f, 0.03f, -4.5f), glm::vec3(0.25f, 0.02f, 7.0f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3(0.0f, 0.03f,  4.5f), glm::vec3(0.25f, 0.02f, 7.0f), 0.0f, 35);
        // Titanium streetlight poles (Mat 2: Titanium Chrome) with Yellow hazard base collars (Mat 24)
        addCylinder(scene.triangles, glm::vec3( 6.8f, 2.5f,  6.8f), 0.08f, 5.0f, 2, 10);
        addCylinder(scene.triangles, glm::vec3(-6.8f, 2.5f, -6.8f), 0.08f, 5.0f, 2, 10);
        addCylinder(scene.triangles, glm::vec3( 6.8f, 0.3f,  6.8f), 0.16f, 0.6f, 24, 10);
        addCylinder(scene.triangles, glm::vec3(-6.8f, 0.3f, -6.8f), 0.16f, 0.6f, 24, 10);
        // Streetlight lamp fixtures (Mat 41: Warm Sodium Vapor Streetlight)
        addBox(scene.triangles, glm::vec3( 6.5f, 4.9f,  6.5f), glm::vec3(0.6f, 0.15f, 0.35f), 0.0f, 41);
        addBox(scene.triangles, glm::vec3(-6.5f, 4.9f, -6.5f), glm::vec3(0.6f, 0.15f, 0.35f), 0.0f, 41);
        recordBlasPrototype("Plaza District", tStart);
    }

    // Prototype 16: Multi-Lane Urban Arterial Avenue 16m x 16m (CYBER_BLAS_ROAD_AVENUE)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Foundation concrete bed (Mat 0: Foundation Concrete) with subsurface overlap
        addBox(scene.triangles, glm::vec3(0.0f, -0.35f, 0.0f), glm::vec3(16.2f, 0.7f, 16.2f), 0.0f, 0);

        // Asphalt roadway surface (Mat 20: Rough Basalt / Dark Asphalt)
        addBox(scene.triangles, glm::vec3(0.0f, 0.02f, 0.0f), glm::vec3(10.0f, 0.04f, 16.0f), 0.0f, 20);

        // Double yellow center divider lines (Mat 35: Electric Gold Emissive)
        addBox(scene.triangles, glm::vec3(-0.15f, 0.045f, 0.0f), glm::vec3(0.12f, 0.01f, 15.9f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3( 0.15f, 0.045f, 0.0f), glm::vec3(0.12f, 0.01f, 15.9f), 0.0f, 35);

        // Dashed white lane division lines (Mat 42: High-Intensity Xenon White Emissive)
        for (int d = -2; d <= 1; ++d) {
            float zCenter = (static_cast<float>(d) + 0.5f) * 4.0f;
            addBox(scene.triangles, glm::vec3(-2.6f, 0.045f, zCenter), glm::vec3(0.14f, 0.01f, 2.2f), 0.0f, 42);
            addBox(scene.triangles, glm::vec3( 2.6f, 0.045f, zCenter), glm::vec3(0.14f, 0.01f, 2.2f), 0.0f, 42);
        }

        // Solid white outer road shoulder stripes (Mat 9: White Ceramic Cladding)
        addBox(scene.triangles, glm::vec3(-4.85f, 0.045f, 0.0f), glm::vec3(0.15f, 0.01f, 15.9f), 0.0f, 9);
        addBox(scene.triangles, glm::vec3( 4.85f, 0.045f, 0.0f), glm::vec3(0.15f, 0.01f, 15.9f), 0.0f, 9);

        // Raised basalt curbs (Mat 8: Basalt Foundation)
        addBox(scene.triangles, glm::vec3(-5.15f, 0.10f, 0.0f), glm::vec3(0.35f, 0.20f, 16.0f), 0.0f, 8);
        addBox(scene.triangles, glm::vec3( 5.15f, 0.10f, 0.0f), glm::vec3(0.35f, 0.20f, 16.0f), 0.0f, 8);

        // Recessed drainage utility grates along gutter (Mat 28: Perforated Mesh)
        addBox(scene.triangles, glm::vec3(-4.70f, 0.038f, 0.0f), glm::vec3(0.16f, 0.01f, 15.8f), 0.0f, 28);
        addBox(scene.triangles, glm::vec3( 4.70f, 0.038f, 0.0f), glm::vec3(0.16f, 0.01f, 15.8f), 0.0f, 28);

        // Pedestrian sidewalk pavement (Mat 4: Wet Reflective Pavement)
        addBox(scene.triangles, glm::vec3(-6.65f, 0.08f, 0.0f), glm::vec3(2.65f, 0.16f, 16.0f), 0.0f, 4);
        addBox(scene.triangles, glm::vec3( 6.65f, 0.08f, 0.0f), glm::vec3(2.65f, 0.16f, 16.0f), 0.0f, 4);

        // Recessed neon pedestrian guidance strips (Mat 45: Electric Turquoise 490nm)
        addBox(scene.triangles, glm::vec3(-6.65f, 0.165f, 0.0f), glm::vec3(0.18f, 0.01f, 15.8f), 0.0f, 45);
        addBox(scene.triangles, glm::vec3( 6.65f, 0.165f, 0.0f), glm::vec3(0.18f, 0.01f, 15.8f), 0.0f, 45);

        // Streetlight titanium masts on both sidewalks (Mat 2: Titanium Chrome)
        addCylinder(scene.triangles, glm::vec3(-7.2f, 3.2f, 0.0f), 0.09f, 6.4f, 2, 10);
        addCylinder(scene.triangles, glm::vec3( 7.2f, 3.2f, 0.0f), 0.09f, 6.4f, 2, 10);

        // Overhanging curved cantilever arms (reaching over the road curbs)
        addBox(scene.triangles, glm::vec3(-6.1f, 6.35f, 0.0f), glm::vec3(2.3f, 0.12f, 0.12f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3( 6.1f, 6.35f, 0.0f), glm::vec3(2.3f, 0.12f, 0.12f), 0.0f, 2);

        // Streetlight lamp fixture heads (Mat 41: Warm Sodium Vapor Streetlight)
        addBox(scene.triangles, glm::vec3(-4.9f, 6.25f, 0.0f), glm::vec3(0.75f, 0.12f, 0.35f), 0.0f, 41);
        addBox(scene.triangles, glm::vec3( 4.9f, 6.25f, 0.0f), glm::vec3(0.75f, 0.12f, 0.35f), 0.0f, 41);

        recordBlasPrototype("Road Avenue", tStart);
    }

    // Prototype 17: Urban 4-Way Intersection 16m x 16m (CYBER_BLAS_ROAD_INTERSECTION)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Foundation concrete bed (Mat 0) with subsurface overlap
        addBox(scene.triangles, glm::vec3(0.0f, -0.35f, 0.0f), glm::vec3(16.2f, 0.7f, 16.2f), 0.0f, 0);

        // Asphalt crossing area (Mat 20)
        addBox(scene.triangles, glm::vec3(0.0f, 0.02f, 0.0f), glm::vec3(16.0f, 0.04f, 16.0f), 0.0f, 20);

        // Zebra pedestrian crosswalks (Mat 9: White Ceramic Cladding) on all 4 directions
        for (int s = -4; s <= 4; ++s) {
            float xPos = static_cast<float>(s) * 1.0f;
            addBox(scene.triangles, glm::vec3(xPos, 0.045f,  6.0f), glm::vec3(0.45f, 0.01f, 2.0f), 0.0f, 9);
            addBox(scene.triangles, glm::vec3(xPos, 0.045f, -6.0f), glm::vec3(0.45f, 0.01f, 2.0f), 0.0f, 9);
        }
        for (int s = -4; s <= 4; ++s) {
            float zPos = static_cast<float>(s) * 1.0f;
            addBox(scene.triangles, glm::vec3( 6.0f, 0.045f, zPos), glm::vec3(2.0f, 0.01f, 0.45f), 0.0f, 9);
            addBox(scene.triangles, glm::vec3(-6.0f, 0.045f, zPos), glm::vec3(2.0f, 0.01f, 0.45f), 0.0f, 9);
        }

        // Corner sidewalks with wet pavement (Mat 4) and basalt curbs (Mat 8)
        for (float sx : {-1.0f, 1.0f}) {
            for (float sz : {-1.0f, 1.0f}) {
                glm::vec3 cPos(sx * 6.5f, 0.08f, sz * 6.5f);
                addBox(scene.triangles, cPos, glm::vec3(2.6f, 0.16f, 2.6f), 0.0f, 4);
                addBox(scene.triangles, cPos + glm::vec3(-sx * 1.35f, 0.02f, 0.0f), glm::vec3(0.2f, 0.20f, 2.8f), 0.0f, 8);
                addBox(scene.triangles, cPos + glm::vec3(0.0f, 0.02f, -sz * 1.35f), glm::vec3(2.8f, 0.20f, 0.2f), 0.0f, 8);
                // Corner illuminated safety bollard (Mat 2 Titanium + Mat 43 Amber + Mat 45 Cyan)
                addCylinder(scene.triangles, cPos + glm::vec3(-sx * 0.8f, 0.5f, -sz * 0.8f), 0.10f, 1.0f, 2, 8);
                addBox(scene.triangles, cPos + glm::vec3(-sx * 0.8f, 1.05f, -sz * 0.8f), glm::vec3(0.22f, 0.12f, 0.22f), 0.0f, 43);
                addBox(scene.triangles, cPos + glm::vec3(-sx * 0.8f, 0.55f, -sz * 0.8f), glm::vec3(0.22f, 0.06f, 0.22f), 0.0f, 45);
            }
        }

        // Overhead traffic signal gantry (Mat 2: Titanium Chrome) across North-South axis
        addCylinder(scene.triangles, glm::vec3(-7.2f, 3.4f, 0.0f), 0.12f, 6.8f, 2, 10);
        addCylinder(scene.triangles, glm::vec3( 7.2f, 3.4f, 0.0f), 0.12f, 6.8f, 2, 10);
        addBox(scene.triangles, glm::vec3(0.0f, 6.7f, 0.0f), glm::vec3(14.6f, 0.25f, 0.25f), 0.0f, 2);

        // Traffic signal heads (Mat 34 Red, Mat 43 Amber, Mat 33 Green)
        for (float tx : {-2.5f, 2.5f}) {
            addBox(scene.triangles, glm::vec3(tx, 6.25f, -0.2f), glm::vec3(0.4f, 0.9f, 0.25f), 0.0f, 20);
            addBox(scene.triangles, glm::vec3(tx, 6.50f, -0.05f), glm::vec3(0.22f, 0.22f, 0.08f), 0.0f, 34); // Red
            addBox(scene.triangles, glm::vec3(tx, 6.25f, -0.05f), glm::vec3(0.22f, 0.22f, 0.08f), 0.0f, 43); // Amber
            addBox(scene.triangles, glm::vec3(tx, 6.00f, -0.05f), glm::vec3(0.22f, 0.22f, 0.08f), 0.0f, 33); // Green
        }

        recordBlasPrototype("Road Intersection", tStart);
    }

    // Prototype 18: Expansive Dark Urban Industrial Apron 32m x 32m (CYBER_BLAS_PERIMETER_GROUND)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Deep sub-base foundation slab (Mat 0) with generous subterranean overlap
        addBox(scene.triangles, glm::vec3(0.0f, -0.45f, 0.0f), glm::vec3(32.4f, 0.9f, 32.4f), 0.0f, 0);

        // Dark industrial asphalt composite surface (Mat 20: Rough Basalt)
        addBox(scene.triangles, glm::vec3(0.0f, 0.015f, 0.0f), glm::vec3(32.0f, 0.03f, 32.0f), 0.0f, 20);

        // Recessed utility drainage / ventilation channel grates (Mat 28: Perforated Acoustic/Ventilation Mesh)
        addBox(scene.triangles, glm::vec3(-10.0f, 0.025f, 0.0f), glm::vec3(1.2f, 0.02f, 30.0f), 0.0f, 28);
        addBox(scene.triangles, glm::vec3( 10.0f, 0.025f, 0.0f), glm::vec3(1.2f, 0.02f, 30.0f), 0.0f, 28);

        // Industrial expansion joints (Mat 35: Electric Gold / Mat 45: Electric Turquoise)
        addBox(scene.triangles, glm::vec3(0.0f, 0.022f, 0.0f), glm::vec3(31.6f, 0.015f, 0.15f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3(0.0f, 0.022f, 0.0f), glm::vec3(0.15f, 0.015f, 31.6f), 0.0f, 45);

        // 4 boundary perimeter beacon pylons (Mat 2 Titanium + Mat 40 Blacklight ring + Mat 34 Ruby Laser)
        for (float bx : {-14.5f, 14.5f}) {
            for (float bz : {-14.5f, 14.5f}) {
                addCylinder(scene.triangles, glm::vec3(bx, 0.6f, bz), 0.12f, 1.2f, 2, 8);
                addCylinder(scene.triangles, glm::vec3(bx, 0.15f, bz), 0.22f, 0.1f, 40, 8);
                addBox(scene.triangles, glm::vec3(bx, 1.25f, bz), glm::vec3(0.28f, 0.18f, 0.28f), 0.0f, 34);
            }
        }

        recordBlasPrototype("Perimeter Ground", tStart);
    }

    // Prototype 19: 3D Voxel Hologram Projector (CYBER_BLAS_HOLO_PROJECTOR)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // 1. Heavy gunmetal base skid (Mat 19: Gunmetal Alloy Armor)
        addBox(scene.triangles, glm::vec3(0.0f, 0.2f, 0.0f), glm::vec3(5.6f, 0.4f, 5.6f), 0.0f, 19);
        // 2. Stepped titanium collar (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(0.0f, 0.45f, 0.0f), glm::vec3(4.6f, 0.15f, 4.6f), 0.0f, 2);
        // 3. Circular receiver plate & glowing turquoise concentric neon border ring (Mat 17 Brushed Platinum, Mat 45 Electric Turquoise)
        addCylinder(scene.triangles, glm::vec3(0.0f, 0.55f, 0.0f), 2.2f, 0.08f, 17, 24);
        addCylinder(scene.triangles, glm::vec3(0.0f, 0.60f, 0.0f), 2.0f, 0.03f, 45, 24);
        addCylinder(scene.triangles, glm::vec3(0.0f, 0.64f, 0.0f), 1.8f, 0.04f, 19, 24);
        // 4. Central optical projector lens / concave dish (Mat 30: Cyan Emitter)
        addCylinder(scene.triangles, glm::vec3(0.0f, 0.70f, 0.0f), 1.68f, 0.04f, 30, 24);

        // 5. Overhead Projector Gantry Rig & Housing (sdProjector from reference shader)
        for (float fx : {-1.85f, 1.85f}) {
            for (float fz : {-1.85f, 1.85f}) {
                // Vertical gunmetal base post
                addBox(scene.triangles, glm::vec3(fx, 0.95f, fz), glm::vec3(0.24f, 0.80f, 0.24f), 0.0f, 19);
                // Glowing turquoise collar ring
                addBox(scene.triangles, glm::vec3(fx, 1.38f, fz), glm::vec3(0.30f, 0.06f, 0.30f), 0.0f, 45);
                // Angled titanium upper truss strut reaching to overhead collar (fx*0.28, 4.25, fz*0.28)
                glm::vec3 pBot(fx, 1.40f, fz);
                glm::vec3 pTop(fx * 0.28f, 4.25f, fz * 0.28f);
                glm::vec3 pMid = (pBot + pTop) * 0.5f;
                addBox(scene.triangles, pMid, glm::vec3(0.12f, 2.90f, 0.12f), (fx * fz > 0.0f ? 28.0f : -28.0f), 2);
                // Collimation guide laser filament connecting overhead lens to corner post
                addBox(scene.triangles, (glm::vec3(0.0f, 4.10f, 0.0f) + pBot) * 0.5f, glm::vec3(0.02f, 2.80f, 0.02f), (fx * fz > 0.0f ? 26.0f : -26.0f), 45);
            }
        }
        // Overhead projector mount gantry ring (Mat 2 Titanium)
        addCylinder(scene.triangles, glm::vec3(0.0f, 4.25f, 0.0f), 0.90f, 0.08f, 2, 24);
        // Projector main housing body (sdProjector body box, Mat 19 Gunmetal)
        addBox(scene.triangles, glm::vec3(0.0f, 4.45f, 0.0f), glm::vec3(0.80f, 0.32f, 0.80f), 0.0f, 19);
        // Projector optical snout (sdProjector snout box, Mat 2 Titanium)
        addBox(scene.triangles, glm::vec3(0.0f, 4.25f, 0.0f), glm::vec3(0.46f, 0.14f, 0.46f), 0.0f, 2);
        // Downward-pointing emitter lens (sdProjector spherical lens, Mat 30 Cyan Emitter)
        addSphere(scene.triangles, glm::vec3(0.0f, 4.10f, 0.0f), 0.24f, 30, 16, 16);
        // Concentric neon emitter focus ring around lens (Mat 45 Electric Turquoise)
        addCylinder(scene.triangles, glm::vec3(0.0f, 4.14f, 0.0f), 0.36f, 0.03f, 45, 24);

        // 6. Upright 3D Voxel Hologram Display: Full 16:9 Widescreen Matrix (32x18 = 576 voxels)
        // Full video coverage (u in [0,1], v in [0,1]) - zero missing content across entire video timeline!
        const uint32_t NX = 32;
        const uint32_t NY = 18;
        const float W = 3.20f;
        const float H = 1.80f;
        const float yBase = 0.95f;
        const float dx = W / static_cast<float>(NX);
        const float dy = H / static_cast<float>(NY);
        const float boxW = dx * 0.88f; // tactile seam gap (12% margin)
        const float boxH = dy * 0.88f; // tactile seam gap (12% margin)
        const float zThick = 0.05f;    // 5cm physical 3D box thickness

        for (uint32_t iy = 0; iy < NY; ++iy) {
            float normY = (static_cast<float>(iy) + 0.5f) / static_cast<float>(NY);
            float cy = yBase + normY * H;
            float v0 = 1.0f - static_cast<float>(iy + 1) / static_cast<float>(NY);
            float v1 = 1.0f - static_cast<float>(iy) / static_cast<float>(NY);

            for (uint32_t ix = 0; ix < NX; ++ix) {
                float normX = (static_cast<float>(ix) + 0.5f) / static_cast<float>(NX);
                float cx = -W * 0.5f + normX * W;
                float u0 = static_cast<float>(ix) / static_cast<float>(NX);
                float u1 = static_cast<float>(ix + 1) / static_cast<float>(NX);

                // Gentle cylindrical concave curvature facing the terrace viewer
                float xRel = cx / (W * 0.5f);
                float cz = -0.15f * (1.0f - xRel * xRel);

                glm::vec3 center(cx, cy, cz);
                glm::vec3 size(boxW, boxH, zThick);
                addVoxelBox(scene.triangles, center, size, 36, glm::vec2(u0, v0), glm::vec2(u1, v1));
            }
        }
        recordBlasPrototype("Hologram Projector", tStart);
    }

    // =========================================================================
    // 2. Hardware TLAS Instancing & Physical Light Sources
    // =========================================================================

    auto addInstance = [&](uint32_t bIdx, const glm::mat4& xform) {
        uint32_t cIdx = static_cast<uint32_t>(scene.instances.size());
        SceneInstance sInst{};
        sInst.blasIndex = bIdx;
        sInst.transform = xform;
        sInst.customIndex = cIdx;
        scene.instances.push_back(sInst);

        InstanceGPU instGPU{};
        instGPU.firstTriangle = scene.blasRanges[bIdx].firstTriangle;
        instGPU.numOpaqueTriangles = scene.blasRanges[bIdx].numOpaqueTriangles;
        instGPU.materialOffset = 0;
        instGPU.flags = 0;
        scene.instanceData.push_back(instGPU);
    };

    auto addBillboardLight = [&](const glm::mat4& xform, glm::vec3 localCorner, glm::vec3 localU, glm::vec3 localV,
                                 glm::vec3 emissionColor, float intensity) {
        glm::vec3 worldCorner = glm::vec3(xform * glm::vec4(localCorner, 1.0f));
        glm::vec3 worldU = glm::vec3(xform * glm::vec4(localU, 0.0f));
        glm::vec3 worldV = glm::vec3(xform * glm::vec4(localV, 0.0f));
        glm::vec3 crossUV = glm::cross(worldU, worldV);
        float area = glm::length(crossUV);
        if (area < 1e-4f) return;
        glm::vec3 worldN = crossUV / area;

        LightGPU light{};
        light.position = glm::vec4(worldCorner, LIGHT_AREA_QUAD);
        light.u = glm::vec4(worldU, 0.0f);
        light.v = glm::vec4(worldV, 0.0f);
        light.normal = glm::vec4(worldN, 0.0f);
        light.emission = glm::vec4(emissionColor * intensity, area);
        scene.lights.push_back(light);
    };

    auto addSpotLight = [&](glm::vec3 pos, glm::vec3 dir, glm::vec3 emissionColor, float intensity,
                            float innerDeg, float outerDeg) {
        LightGPU light{};
        light.position = glm::vec4(pos, LIGHT_SPOT);
        light.normal = glm::vec4(glm::normalize(dir), 0.0f);
        light.emission = glm::vec4(emissionColor * intensity, 1.0f);
        light.u = glm::vec4(0.0f, 0.0f, 0.0f, std::cos(glm::radians(innerDeg)));
        light.v = glm::vec4(0.0f, 0.0f, 0.0f, std::cos(glm::radians(outerDeg)));
        scene.lights.push_back(light);
    };

    // 2a. Expansive Ground Network: Plazas, Boulevards, Intersections & Perimeter Apron
    // 1. Central Plaza District under Megatowers (14x14 grid minus central boulevard corridor)
    for (int gx = -7; gx < 7; ++gx) {
        if (gx == -1 || gx == 0) continue; // Reserved for Central Grand Boulevard corridor at X in [-16, +16]
        for (int gz = -7; gz < 7; ++gz) {
            float px = (static_cast<float>(gx) + 0.5f) * 16.0f;
            float pz = (static_cast<float>(gz) + 0.5f) * 16.0f;
            glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3(px, 0.0f, pz));
            addInstance(CYBER_BLAS_PLAZA_DISTRICT, M);
        }
    }

    // 2. Central Grand Boulevard Corridor (X in [-16, +16], 14 rows along Z in [-112, +112])
    for (int gz = -7; gz < 7; ++gz) {
        float pz = (static_cast<float>(gz) + 0.5f) * 16.0f;

        // Center 16m multi-lane avenue (spanning X in [-8, +8])
        glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, pz));
        addInstance(CYBER_BLAS_ROAD_AVENUE, M);

        // Physical downward sodium streetlights on both sidewalks
        addSpotLight(glm::vec3(-4.9f, 6.2f, pz), glm::vec3(0.15f, -1.0f, 0.0f), glm::vec3(1.0f, 0.75f, 0.2f), 55.0f, 35.0f, 65.0f);
        addSpotLight(glm::vec3( 4.9f, 6.2f, pz), glm::vec3(-0.15f, -1.0f, 0.0f), glm::vec3(1.0f, 0.75f, 0.2f), 55.0f, 35.0f, 65.0f);

        // Left 8m Grand Promenade (spanning X in [-16, -8], centered at px = -12.0m)
        glm::mat4 mLeft = glm::translate(glm::mat4(1.0f), glm::vec3(-12.0f, 0.0f, pz)) *
                          glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f));
        addInstance(CYBER_BLAS_PLAZA_DISTRICT, mLeft);

        // Right 8m Grand Promenade (spanning X in [+8, +16], centered at px = +12.0m)
        glm::mat4 mRight = glm::translate(glm::mat4(1.0f), glm::vec3(12.0f, 0.0f, pz)) *
                           glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f));
        addInstance(CYBER_BLAS_PLAZA_DISTRICT, mRight);
    }

    // 3. Perimeter Arterial Ring Boulevards (North & South at Z = +-120m, East & West at X = +-120m)
    // 3a. North and South Boulevards (East-West axis along Z = +-120m)
    for (float pz : {-120.0f, 120.0f}) {
        // Center intersection at X = 0 (spanning X in [-8, +8])
        glm::mat4 mCenter = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, pz));
        addInstance(CYBER_BLAS_ROAD_INTERSECTION, mCenter);
        addSpotLight(glm::vec3(0.0f, 6.6f, pz), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 0.75f, 0.2f), 45.0f, 30.0f, 60.0f);

        // 8m East-West road connectors flanking center intersection (spanning X in [-16, -8] and [+8, +16])
        for (float px : {-12.0f, 12.0f}) {
            glm::mat4 mConn = glm::translate(glm::mat4(1.0f), glm::vec3(px, 0.0f, pz)) *
                              glm::scale(glm::mat4(1.0f), glm::vec3(0.5f, 1.0f, 1.0f)) *
                              glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            addInstance(CYBER_BLAS_ROAD_AVENUE, mConn);
        }

        // Corner intersections at X = +-120m (spanning X in [+-112, +-128], Z in [+-112, +-128])
        for (float px : {-120.0f, 120.0f}) {
            glm::mat4 mCorner = glm::translate(glm::mat4(1.0f), glm::vec3(px, 0.0f, pz));
            addInstance(CYBER_BLAS_ROAD_INTERSECTION, mCorner);
            addSpotLight(glm::vec3(px, 6.6f, pz), glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(1.0f, 0.75f, 0.2f), 45.0f, 30.0f, 60.0f);
        }

        // East-West road segments between center connectors and corners (X in [-112, -16] and [+16, +112])
        for (int gx = -7; gx < 7; ++gx) {
            if (gx == -1 || gx == 0) continue; // Covered by center intersection & connectors
            float px = (static_cast<float>(gx) + 0.5f) * 16.0f;
            glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3(px, 0.0f, pz)) *
                          glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            addInstance(CYBER_BLAS_ROAD_AVENUE, M);
            addSpotLight(glm::vec3(px, 6.2f, pz - 4.9f), glm::vec3(0.0f, -1.0f, 0.15f), glm::vec3(1.0f, 0.75f, 0.2f), 50.0f, 35.0f, 65.0f);
            addSpotLight(glm::vec3(px, 6.2f, pz + 4.9f), glm::vec3(0.0f, -1.0f, -0.15f), glm::vec3(1.0f, 0.75f, 0.2f), 50.0f, 35.0f, 65.0f);
        }
    }

    // 3b. East and West Boulevards (North-South axis along X = +-120m, all 14 rows Z in [-112, +112])
    for (float px : {-120.0f, 120.0f}) {
        for (int gz = -7; gz < 7; ++gz) {
            float pz = (static_cast<float>(gz) + 0.5f) * 16.0f;
            glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3(px, 0.0f, pz));
            addInstance(CYBER_BLAS_ROAD_AVENUE, M);
            addSpotLight(glm::vec3(px - 4.9f, 6.2f, pz), glm::vec3(0.15f, -1.0f, 0.0f), glm::vec3(1.0f, 0.75f, 0.2f), 50.0f, 35.0f, 65.0f);
            addSpotLight(glm::vec3(px + 4.9f, 6.2f, pz), glm::vec3(-0.15f, -1.0f, 0.0f), glm::vec3(1.0f, 0.75f, 0.2f), 50.0f, 35.0f, 65.0f);
        }
    }

    // 4. Expansive Outer Perimeter Ground Base (32m x 32m tiles spanning +-288m)
    for (int GX = -9; GX < 9; ++GX) {
        float px = (static_cast<float>(GX) + 0.5f) * 32.0f;
        for (int GZ = -9; GZ < 9; ++GZ) {
            float pz = (static_cast<float>(GZ) + 0.5f) * 32.0f;
            // Only instance outer ground outside the central city & ring road core
            if (std::abs(px) >= 128.0f || std::abs(pz) >= 128.0f) {
                glm::mat4 M = glm::translate(glm::mat4(1.0f), glm::vec3(px, 0.0f, pz));
                addInstance(CYBER_BLAS_PERIMETER_GROUND, M);

                // Add perimeter hazard beacons along outer boundary
                if (std::abs(px) >= 256.0f || std::abs(pz) >= 256.0f) {
                    if ((GX + GZ) % 3 == 0) {
                        addSpotLight(glm::vec3(px, 1.5f, pz), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(1.0f, 0.1f, 0.1f), 30.0f, 40.0f, 75.0f);
                    }
                }
            }
        }
    }

    // 2b. Skyscraper Megatowers (12x12 = 144 towers)
    for (uint32_t gx = 0; gx < 12; ++gx) {
        float posX = (gx <= 5) ? (-14.0f - static_cast<float>(5 - gx) * 18.0f)
                               : (+14.0f + static_cast<float>(gx - 6) * 18.0f);
        for (uint32_t gz = 0; gz < 12; ++gz) {
            float posZ = -99.0f + static_cast<float>(gz) * 18.0f;

            uint32_t seed = (gx * 37u + gz * 61u + 17u) % 100u;
            uint32_t numMids;
            if (gx == 5 || gx == 6) {
                numMids = 24u + (seed % 11u); // Super-tall central avenue towers: 24 to 34 mids (152m - 212m)
            } else if (gx == 4 || gx == 7) {
                numMids = 18u + (seed % 9u);  // Inner high-rise towers: 18 to 26 mids (116m - 164m)
            } else if (gx == 3 || gx == 8) {
                numMids = 14u + (seed % 7u);  // Mid-district towers: 14 to 20 mids (92m - 128m)
            } else {
                numMids = 10u + (seed % 7u);  // Outer district towers: 10 to 16 mids (68m - 104m)
            }

            // Tower Base Block
            glm::mat4 mBase = glm::translate(glm::mat4(1.0f), glm::vec3(posX, 0.0f, posZ));
            addInstance(CYBER_BLAS_TOWER_BASE, mBase);

            // Entrance portal lights on all 4 cardinal facades (+X, -X, +Z, -Z outward)
            addBillboardLight(mBase, glm::vec3(6.33f, 0.42f, 1.6f), glm::vec3(0.0f, 0.0f, -3.2f), glm::vec3(0.0f, 3.88f, 0.0f),
                              glm::vec3(2.0f, 28.0f, 36.0f), 1.0f);
            addBillboardLight(mBase, glm::vec3(-6.33f, 0.42f, -1.6f), glm::vec3(0.0f, 0.0f, 3.2f), glm::vec3(0.0f, 3.88f, 0.0f),
                              glm::vec3(2.0f, 28.0f, 36.0f), 1.0f);
            addBillboardLight(mBase, glm::vec3(-1.6f, 0.42f, 6.33f), glm::vec3(3.2f, 0.0f, 0.0f), glm::vec3(0.0f, 3.88f, 0.0f),
                              glm::vec3(2.0f, 28.0f, 36.0f), 1.0f);
            addBillboardLight(mBase, glm::vec3(1.6f, 0.42f, -6.33f), glm::vec3(-3.2f, 0.0f, 0.0f), glm::vec3(0.0f, 3.88f, 0.0f),
                              glm::vec3(2.0f, 28.0f, 36.0f), 1.0f);

            // Stacking Mid Modules (Cycling Mid A, Mid B, Mid C with 90-degree rotations)
            for (uint32_t m = 0; m < numMids; ++m) {
                float y = 8.0f + static_cast<float>(m) * 6.0f;
                uint32_t mType = (m + gx + gz) % 3u;
                uint32_t midProto = (mType == 0) ? CYBER_BLAS_TOWER_MID_A
                                  : (mType == 1) ? CYBER_BLAS_TOWER_MID_B
                                                 : CYBER_BLAS_TOWER_MID_C;
                float rotAngle = static_cast<float>((m * 90u) % 360u);
                glm::mat4 mMid = glm::translate(glm::mat4(1.0f), glm::vec3(posX, y, posZ)) *
                                 glm::rotate(glm::mat4(1.0f), glm::radians(rotAngle), glm::vec3(0.0f, 1.0f, 0.0f));
                addInstance(midProto, mMid);

                // Periodic communication relay gantries mounted on mid setbacks
                if (m % 3 == 2 && ((gx + gz + m) % 3 == 0)) {
                    glm::mat4 mGantry = glm::translate(glm::mat4(1.0f), glm::vec3(posX + 4.8f, y + 1.0f, posZ + 4.8f));
                    addInstance(CYBER_BLAS_COMM_GANTRY, mGantry);
                }

                // Camera-facing terrace holographic projectors on Mid B setbacks
                if (gx == 5 && gz == 7 && m == 3) {
                    glm::mat4 mTerraceHolo = glm::translate(glm::mat4(1.0f), glm::vec3(-6.4f, y + 1.2f, 30.5f)) *
                                             glm::rotate(glm::mat4(1.0f), glm::radians(15.0f), glm::vec3(0.0f, 1.0f, 0.0f));
                    addInstance(CYBER_BLAS_HOLO_PROJECTOR, mTerraceHolo);
                    addBillboardLight(mTerraceHolo, glm::vec3(-1.6f, 1.85f, 0.0f), glm::vec3(3.2f, 0.0f, 0.0f), glm::vec3(0.0f, 1.8f, 0.0f), glm::vec3(0.5f, 1.5f, 3.0f), 0.5f);
                }
                if (gx == 6 && gz == 7 && m == 3) {
                    glm::mat4 mTerraceHolo = glm::translate(glm::mat4(1.0f), glm::vec3(6.4f, y + 1.2f, 30.5f)) *
                                             glm::rotate(glm::mat4(1.0f), glm::radians(-15.0f), glm::vec3(0.0f, 1.0f, 0.0f));
                    addInstance(CYBER_BLAS_HOLO_PROJECTOR, mTerraceHolo);
                    addBillboardLight(mTerraceHolo, glm::vec3(-1.6f, 1.85f, 0.0f), glm::vec3(3.2f, 0.0f, 0.0f), glm::vec3(0.0f, 1.8f, 0.0f), glm::vec3(0.5f, 1.5f, 3.0f), 0.5f);
                }
            }

            // Tower Crown & Spire
            float crownY = 8.0f + static_cast<float>(numMids) * 6.0f;
            glm::mat4 mCrown = glm::translate(glm::mat4(1.0f), glm::vec3(posX, crownY, posZ));
            addInstance(CYBER_BLAS_TOWER_CROWN, mCrown);

            // Spire Warning Beacons: Downward spotlights on cross-arms
            addSpotLight(glm::vec3(posX + 0.60f, crownY + 13.35f, posZ), glm::vec3(0.0f, -1.0f, 0.0f),
                         glm::vec3(50.0f, 3.0f, 3.0f), 1.0f, 40.0f, 70.0f);
            addSpotLight(glm::vec3(posX - 0.60f, crownY + 13.35f, posZ), glm::vec3(0.0f, -1.0f, 0.0f),
                         glm::vec3(50.0f, 3.0f, 3.0f), 1.0f, 40.0f, 70.0f);

            // Rooftop Equipment: Alternate Solar Roof Arrays and Industrial HVAC Pods
            glm::mat4 mRoof = glm::translate(glm::mat4(1.0f), glm::vec3(posX, crownY + 2.0f, posZ));
            if ((gx + gz) % 2 == 0) {
                addInstance(CYBER_BLAS_SOLAR_ROOF, mRoof);
            } else {
                addInstance(CYBER_BLAS_ROOFTOP_HVAC, mRoof);
            }

            // Rooftop Holographic Video Projection Displays (CYBER_BLAS_BILLBOARD_VIOLET - Mat 36)
            if ((gx * 3 + gz * 5) % 2 == 0) {
                glm::mat4 mRoofHolo = glm::translate(glm::mat4(1.0f), glm::vec3(posX, crownY + 3.8f, posZ));
                addInstance(CYBER_BLAS_BILLBOARD_VIOLET, mRoofHolo);
                addBillboardLight(mRoofHolo, glm::vec3(-2.5f, -2.5f, 0.12f), glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(0.0f, 5.0f, 0.0f), glm::vec3(20.0f, 2.0f, 35.0f), 1.2f);
            }

            // Billboards on tower facades
            for (float by = 16.0f; by < crownY - 6.0f; by += 12.0f) {
                uint32_t bbSeed = static_cast<uint32_t>(gx * 19 + gz * 31 + static_cast<int>(by)) % 10u;
                if (bbSeed > 4u) continue;

                // Lower level billboards (by <= 28.0f) play video as flat 2D emission (CYBER_BLAS_BILLBOARD_CYAN - Mat 30)
                uint32_t bbProto = CYBER_BLAS_BILLBOARD_CYAN;
                glm::vec3 bbLocalCorner(-3.8f, -2.05f, 0.12f);
                glm::vec3 bbLocalU(7.6f, 0.0f, 0.0f);
                glm::vec3 bbLocalV(0.0f, 4.1f, 0.0f);
                glm::vec3 bbColor(1.5f, 2.5f, 3.5f);

                if (by > 28.0f) {
                    if (bbSeed % 3 == 0) {
                        bbProto = CYBER_BLAS_BILLBOARD_VIOLET;
                        bbLocalCorner = glm::vec3(-2.5f, -2.5f, 0.12f);
                        bbLocalU = glm::vec3(5.0f, 0.0f, 0.0f);
                        bbLocalV = glm::vec3(0.0f, 5.0f, 0.0f);
                        bbColor = glm::vec3(20.0f, 2.0f, 35.0f);
                    } else if (bbSeed % 3 == 1) {
                        bbProto = CYBER_BLAS_BILLBOARD_MAGENTA;
                        bbLocalCorner = glm::vec3(-2.05f, -3.8f, 0.12f);
                        bbLocalU = glm::vec3(4.1f, 0.0f, 0.0f);
                        bbLocalV = glm::vec3(0.0f, 7.6f, 0.0f);
                        bbColor = glm::vec3(35.0f, 2.0f, 18.0f);
                    } else {
                        bbProto = CYBER_BLAS_BILLBOARD_ORANGE;
                        bbLocalCorner = glm::vec3(-3.3f, -1.45f, 0.12f);
                        bbLocalU = glm::vec3(6.6f, 0.0f, 0.0f);
                        bbLocalV = glm::vec3(0.0f, 2.9f, 0.0f);
                        bbColor = glm::vec3(36.0f, 12.0f, 1.5f);
                    }
                }

                glm::mat4 mBillboard(1.0f);
                if (gx == 5) {
                    mBillboard = glm::translate(glm::mat4(1.0f), glm::vec3(posX + 6.25f, by, posZ)) *
                                 glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0, 1, 0));
                } else if (gx == 6) {
                    mBillboard = glm::translate(glm::mat4(1.0f), glm::vec3(posX - 6.25f, by, posZ)) *
                                 glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(0, 1, 0));
                } else if (bbSeed % 2 == 0) {
                    mBillboard = glm::translate(glm::mat4(1.0f), glm::vec3(posX, by, posZ + 6.25f));
                } else {
                    mBillboard = glm::translate(glm::mat4(1.0f), glm::vec3(posX, by, posZ - 6.25f)) *
                                 glm::rotate(glm::mat4(1.0f), glm::radians(180.0f), glm::vec3(0, 1, 0));
                }

                addInstance(bbProto, mBillboard);
                addBillboardLight(mBillboard, bbLocalCorner, bbLocalU, bbLocalV, bbColor, 1.0f);
            }
        }
    }

    // 2c. Multi-Tier Skybridges (CYBER_BLAS_SKYBRIDGE)
    // Central Avenue Skybridges (Connecting gx = 5 and gx = 6, 28m span) across 6 tiers
    for (uint32_t gz = 0; gz < 12; ++gz) {
        float posZ = -99.0f + static_cast<float>(gz) * 18.0f;
        for (float by : { 24.0f, 44.0f, 64.0f, 84.0f, 104.0f, 124.0f }) {
            glm::mat4 mBridge = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, by, posZ)) *
                                glm::scale(glm::mat4(1.0f), glm::vec3(28.0f / 16.0f, 1.0f, 1.0f));
            addInstance(CYBER_BLAS_SKYBRIDGE, mBridge);
            // Light quad sits beneath bottom face at y = -0.11f
            addBillboardLight(mBridge, glm::vec3(-14.0f / 1.75f, -0.11f, -0.2f), glm::vec3(28.0f / 1.75f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.4f),
                              glm::vec3(2.0f, 28.0f, 36.0f), 1.0f);

            // On tier 1 (by == 24.0f) of the camera foreground skybridge (gz == 7, posZ == 27.0f), mount central holographic projection display
            if (gz == 7 && std::abs(by - 24.0f) < 1.0f) {
                glm::mat4 mHoloBridge = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, by + 3.20f, posZ));
                addInstance(CYBER_BLAS_HOLO_PROJECTOR, mHoloBridge);
                addBillboardLight(mHoloBridge, glm::vec3(-1.6f, 1.85f, 0.0f), glm::vec3(3.2f, 0.0f, 0.0f), glm::vec3(0.0f, 1.8f, 0.0f), glm::vec3(0.5f, 1.5f, 3.0f), 0.5f);
            }
        }
    }

    // Inter-Tower Skybridges along X across multiple vertical elevations
    for (uint32_t gx = 0; gx < 11; ++gx) {
        if (gx == 5) continue; // Avenue gap
        float posX1 = (gx <= 5) ? (-14.0f - static_cast<float>(5 - gx) * 18.0f) : (+14.0f + static_cast<float>(gx - 6) * 18.0f);
        float posX2 = ((gx + 1) <= 5) ? (-14.0f - static_cast<float>(5 - (gx + 1)) * 18.0f) : (+14.0f + static_cast<float>((gx + 1) - 6) * 18.0f);
        float midX = (posX1 + posX2) * 0.5f;

        for (uint32_t gz = 0; gz < 12; ++gz) {
            float posZ = -99.0f + static_cast<float>(gz) * 18.0f;
            if ((gx + gz) % 2 == 0) {
                for (float by : { 28.0f, 60.0f }) {
                    glm::mat4 mBridge = glm::translate(glm::mat4(1.0f), glm::vec3(midX, by, posZ)) *
                                        glm::scale(glm::mat4(1.0f), glm::vec3(18.0f / 16.0f, 1.0f, 1.0f));
                    addInstance(CYBER_BLAS_SKYBRIDGE, mBridge);
                    addBillboardLight(mBridge, glm::vec3(-32.0f / 9.0f, -0.11f, -0.2f), glm::vec3(64.0f / 9.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.4f),
                                      glm::vec3(2.0f, 28.0f, 36.0f), 1.0f);
                }
            }
        }
    }

    // Inter-Tower Skybridges along Z across multiple vertical elevations
    for (uint32_t gx = 0; gx < 12; ++gx) {
        float posX = (gx <= 5) ? (-14.0f - static_cast<float>(5 - gx) * 18.0f) : (+14.0f + static_cast<float>(gx - 6) * 18.0f);
        for (uint32_t gz = 0; gz < 11; ++gz) {
            float posZ1 = -99.0f + static_cast<float>(gz) * 18.0f;
            float posZ2 = -99.0f + static_cast<float>(gz + 1) * 18.0f;
            float midZ = (posZ1 + posZ2) * 0.5f;
            if ((gx * 3 + gz) % 3 == 0) {
                for (float by : { 34.0f, 66.0f }) {
                    glm::mat4 mBridge = glm::translate(glm::mat4(1.0f), glm::vec3(posX, by, midZ)) *
                                        glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)) *
                                        glm::scale(glm::mat4(1.0f), glm::vec3(18.0f / 16.0f, 1.0f, 1.0f));
                    addInstance(CYBER_BLAS_SKYBRIDGE, mBridge);
                    addBillboardLight(mBridge, glm::vec3(-32.0f / 9.0f, -0.11f, -0.2f), glm::vec3(64.0f / 9.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.4f),
                                      glm::vec3(2.0f, 28.0f, 36.0f), 1.0f);
                }
            }
        }
    }

    // 2d. Elevated Maglev Transit Guideways (CYBER_BLAS_TRANSIT_GUIDEWAY)
    // Avenue Guideways North-South (Low, Mid, and Express High)
    for (float hx : { -5.0f, 5.0f }) {
        for (int sz = -6; sz <= 6; ++sz) {
            float pz = static_cast<float>(sz) * 16.0f;
            glm::mat4 mTrackLow = glm::translate(glm::mat4(1.0f), glm::vec3(hx, 10.0f, pz)) *
                                  glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            addInstance(CYBER_BLAS_TRANSIT_GUIDEWAY, mTrackLow);

            glm::mat4 mTrackMid = glm::translate(glm::mat4(1.0f), glm::vec3(hx, 17.0f, pz)) *
                                  glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            addInstance(CYBER_BLAS_TRANSIT_GUIDEWAY, mTrackMid);

            glm::mat4 mTrackHigh = glm::translate(glm::mat4(1.0f), glm::vec3(hx, 24.0f, pz)) *
                                   glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            addInstance(CYBER_BLAS_TRANSIT_GUIDEWAY, mTrackHigh);
        }
    }
    // Cross-Avenue Guideways East-West (Two altitude tiers)
    for (float cz : { -54.0f, -18.0f, 18.0f, 54.0f }) {
        for (int sx = -6; sx <= 6; ++sx) {
            float px = static_cast<float>(sx) * 16.0f;
            glm::mat4 mTrackLow = glm::translate(glm::mat4(1.0f), glm::vec3(px, 13.5f, cz));
            addInstance(CYBER_BLAS_TRANSIT_GUIDEWAY, mTrackLow);

            glm::mat4 mTrackHigh = glm::translate(glm::mat4(1.0f), glm::vec3(px, 20.5f, cz));
            addInstance(CYBER_BLAS_TRANSIT_GUIDEWAY, mTrackHigh);
        }
    }

    // 2e. Autonomous Sky Cabs in 3D Flight Corridors (CYBER_BLAS_SKY_CAB) - 700 vehicles
    // Corridor 1: Avenue North-South Low (175 vehicles)
    for (int i = 0; i < 175; ++i) {
        float f = static_cast<float>(i) / 175.0f;
        float y = 14.0f + std::fmod(f * 48.0f, 44.0f);
        float z = -105.0f + f * 210.0f;
        float x = (i % 2 == 0) ? -2.4f : 2.4f;
        float rot = (i % 2 == 0) ? 90.0f : -90.0f;

        glm::mat4 mVehicle = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z)) *
                             glm::rotate(glm::mat4(1.0f), glm::radians(rot), glm::vec3(0.0f, 1.0f, 0.0f));
        addInstance(CYBER_BLAS_SKY_CAB, mVehicle);
    }
    // Corridor 2: Avenue North-South High (175 vehicles)
    for (int i = 0; i < 175; ++i) {
        float f = static_cast<float>(i) / 175.0f;
        float y = 62.0f + std::fmod(f * 56.0f, 52.0f);
        float z = -105.0f + f * 210.0f;
        float x = (i % 2 == 0) ? -3.2f : 3.2f;
        float rot = (i % 2 == 0) ? 90.0f : -90.0f;

        glm::mat4 mVehicle = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z)) *
                             glm::rotate(glm::mat4(1.0f), glm::radians(rot), glm::vec3(0.0f, 1.0f, 0.0f));
        addInstance(CYBER_BLAS_SKY_CAB, mVehicle);
    }
    // Corridor 3: Cross-City East-West Low (175 vehicles)
    for (int i = 0; i < 175; ++i) {
        float f = static_cast<float>(i) / 175.0f;
        float y = 18.0f + std::fmod(f * 40.0f, 36.0f);
        float x = -105.0f + f * 210.0f;
        float z = (i % 2 == 0) ? -45.0f : 45.0f;
        float rot = (i % 2 == 0) ? 0.0f : 180.0f;

        glm::mat4 mVehicle = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z)) *
                             glm::rotate(glm::mat4(1.0f), glm::radians(rot), glm::vec3(0.0f, 1.0f, 0.0f));
        addInstance(CYBER_BLAS_SKY_CAB, mVehicle);
    }
    // Corridor 4: Cross-City East-West High (175 vehicles)
    for (int i = 0; i < 175; ++i) {
        float f = static_cast<float>(i) / 175.0f;
        float y = 70.0f + std::fmod(f * 48.0f, 44.0f);
        float x = -105.0f + f * 210.0f;
        float z = (i % 2 == 0) ? -18.0f : 18.0f;
        float rot = (i % 2 == 0) ? 0.0f : 180.0f;

        glm::mat4 mVehicle = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z)) *
                             glm::rotate(glm::mat4(1.0f), glm::radians(rot), glm::vec3(0.0f, 1.0f, 0.0f));
        addInstance(CYBER_BLAS_SKY_CAB, mVehicle);
    }

    // =========================================================================
    // 3. Compute Scene Spatial Bounds & Dielectric Bounds
    // =========================================================================
    glm::vec3 bMin(1e30f);
    glm::vec3 bMax(-1e30f);
    glm::vec3 dMin(1e30f);
    glm::vec3 dMax(-1e30f);
    std::vector<glm::vec3> blasMin(scene.blasRanges.size(), glm::vec3(1e30f));
    std::vector<glm::vec3> blasMax(scene.blasRanges.size(), glm::vec3(-1e30f));
    std::vector<glm::vec3> blasDMin(scene.blasRanges.size(), glm::vec3(1e30f));
    std::vector<glm::vec3> blasDMax(scene.blasRanges.size(), glm::vec3(-1e30f));
    std::vector<bool> blasHasDielectric(scene.blasRanges.size(), false);

    for (size_t b = 0; b < scene.blasRanges.size(); ++b) {
        const auto& range = scene.blasRanges[b];
        for (uint32_t t = 0; t < range.triangleCount; ++t) {
            const auto& tri = scene.triangles[range.firstTriangle + t];
            glm::vec3 v0(tri.v0.position);
            glm::vec3 v1(tri.v1.position);
            glm::vec3 v2(tri.v2.position);
            blasMin[b] = glm::min(blasMin[b], glm::min(v0, glm::min(v1, v2)));
            blasMax[b] = glm::max(blasMax[b], glm::max(v0, glm::max(v1, v2)));

            if (tri.materialId < scene.materials.size() &&
                (scene.materials[tri.materialId].type & 0xFFu) == MATERIAL_DIELECTRIC) {
                blasHasDielectric[b] = true;
                blasDMin[b] = glm::min(blasDMin[b], glm::min(v0, glm::min(v1, v2)));
                blasDMax[b] = glm::max(blasDMax[b], glm::max(v0, glm::max(v1, v2)));
            }
        }
    }

    for (const auto& inst : scene.instances) {
        if (inst.blasIndex >= scene.blasRanges.size()) continue;
        const glm::vec3& lMin = blasMin[inst.blasIndex];
        const glm::vec3& lMax = blasMax[inst.blasIndex];
        for (int c = 0; c < 8; ++c) {
            glm::vec3 corner(
                (c & 1) ? lMax.x : lMin.x,
                (c & 2) ? lMax.y : lMin.y,
                (c & 4) ? lMax.z : lMin.z
            );
            glm::vec3 wCorner = glm::vec3(inst.transform * glm::vec4(corner, 1.0f));
            bMin = glm::min(bMin, wCorner);
            bMax = glm::max(bMax, wCorner);
        }

        if (blasHasDielectric[inst.blasIndex]) {
            const glm::vec3& dlMin = blasDMin[inst.blasIndex];
            const glm::vec3& dlMax = blasDMax[inst.blasIndex];
            for (int c = 0; c < 8; ++c) {
                glm::vec3 corner(
                    (c & 1) ? dlMax.x : dlMin.x,
                    (c & 2) ? dlMax.y : dlMin.y,
                    (c & 4) ? dlMax.z : dlMin.z
                );
                glm::vec3 wCorner = glm::vec3(inst.transform * glm::vec4(corner, 1.0f));
                dMin = glm::min(dMin, wCorner);
                dMax = glm::max(dMax, wCorner);
            }
        }
    }

    scene.boundsMin = bMin;
    scene.boundsMax = bMax;
    scene.sceneRadius = glm::length(bMax - bMin) * 0.5f;
    scene.focalBoundsMin = bMin;
    scene.focalBoundsMax = bMax;
    scene.focalRadius = scene.sceneRadius;

    if (dMin.x <= dMax.x) {
        scene.hasDielectrics = true;
        scene.dielectricBoundsMin = dMin;
        scene.dielectricBoundsMax = dMax;
    }

    scene.hasCamera = true;
    scene.cameraPosition = glm::vec3(-7.0f, 28.0f, 42.0f);
    scene.cameraTarget = glm::vec3(6.0f, 22.0f, -30.0f);
    scene.cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    scene.cameraFov = 62.0f;
    scene.focalDistance = glm::length(scene.cameraPosition - scene.cameraTarget);
    scene.centralTarget = scene.cameraTarget;

    return scene;
}

static std::vector<MaterialGPU> createInfinityMirrorMaterials() {
    std::vector<MaterialGPU> materials;

    // 0: Dark Studio Matte Diffuse Floor
    MaterialGPU matFloor{};
    matFloor.albedo = glm::vec4(0.08f, 0.08f, 0.09f, 1.0f);
    matFloor.roughness = 0.85f;
    matFloor.metallic = 0.0f;
    matFloor.type = MATERIAL_DIFFUSE;
    materials.push_back(matFloor);

    // 1: Portal Frame (Dark anodized metallic)
    MaterialGPU matFrame{};
    matFrame.albedo = glm::vec4(0.035f, 0.035f, 0.04f, 1.0f);
    matFrame.roughness = 0.20f;
    matFrame.metallic = 0.85f;
    matFrame.type = MATERIAL_METALLIC;
    materials.push_back(matFrame);

    // 2: Optical Mirror (Opposing end mirrors, 99.5% reflectance, 0.001 roughness)
    MaterialGPU matMirror{};
    matMirror.albedo = glm::vec4(0.995f, 0.995f, 0.995f, 1.0f);
    matMirror.roughness = 0.001f;
    matMirror.metallic = 1.0f;
    matMirror.type = MATERIAL_METALLIC;
    materials.push_back(matMirror);

    // 3: Chrome Mirror Sphere (Off-axis left)
    MaterialGPU matChrome{};
    matChrome.albedo = glm::vec4(0.96f, 0.96f, 0.96f, 1.0f);
    matChrome.roughness = 0.004f;
    matChrome.metallic = 1.0f;
    matChrome.type = MATERIAL_METALLIC;
    materials.push_back(matChrome);

    // 4: Polished Gold (Tilted accent ring on right)
    MaterialGPU matGold{};
    matGold.albedo = glm::vec4(1.0f, 0.82f, 0.32f, 1.0f);
    matGold.roughness = 0.03f;
    matGold.metallic = 1.0f;
    matGold.type = MATERIAL_METALLIC;
    materials.push_back(matGold);

    // 5: Gloss Red (Bauhaus lacquer)
    MaterialGPU matRed{};
    matRed.albedo = glm::vec4(0.88f, 0.04f, 0.04f, 1.0f);
    matRed.roughness = 0.04f;
    matRed.metallic = 0.0f;
    matRed.clearcoat = 1.0f;
    matRed.clearcoatRoughness = 0.02f;
    matRed.type = MATERIAL_DIFFUSE;
    materials.push_back(matRed);

    // 6: Gloss Green (Bauhaus lacquer)
    MaterialGPU matGreen{};
    matGreen.albedo = glm::vec4(0.04f, 0.82f, 0.38f, 1.0f);
    matGreen.roughness = 0.04f;
    matGreen.metallic = 0.0f;
    matGreen.clearcoat = 1.0f;
    matGreen.clearcoatRoughness = 0.02f;
    matGreen.type = MATERIAL_DIFFUSE;
    materials.push_back(matGreen);

    // 7: Gloss Yellow (Bauhaus lacquer)
    MaterialGPU matYellow{};
    matYellow.albedo = glm::vec4(0.98f, 0.78f, 0.02f, 1.0f);
    matYellow.roughness = 0.05f;
    matYellow.metallic = 0.0f;
    matYellow.clearcoat = 1.0f;
    matYellow.clearcoatRoughness = 0.02f;
    matYellow.type = MATERIAL_DIFFUSE;
    materials.push_back(matYellow);

    // 8: Gloss Blue (Bauhaus lacquer)
    MaterialGPU matBlue{};
    matBlue.albedo = glm::vec4(0.02f, 0.22f, 0.92f, 1.0f);
    matBlue.roughness = 0.04f;
    matBlue.metallic = 0.0f;
    matBlue.clearcoat = 1.0f;
    matBlue.clearcoatRoughness = 0.02f;
    matBlue.type = MATERIAL_DIFFUSE;
    materials.push_back(matBlue);

    // 9: Pedestal Matte (Dark pedestal)
    MaterialGPU matPed{};
    matPed.albedo = glm::vec4(0.10f, 0.10f, 0.11f, 1.0f);
    matPed.roughness = 0.80f;
    matPed.metallic = 0.0f;
    matPed.type = MATERIAL_DIFFUSE;
    materials.push_back(matPed);

    // 10: Overhead Linear Light Bar (Warm LED)
    MaterialGPU matLight{};
    matLight.albedo = glm::vec4(1.0f, 0.95f, 0.88f, 1.0f);
    matLight.emissive = glm::vec4(20.0f, 18.8f, 16.0f, 1.0f);
    matLight.type = MATERIAL_EMISSIVE;
    materials.push_back(matLight);

    // 11: Dielectric Crown Glass
    MaterialGPU matGlass{};
    matGlass.albedo = glm::vec4(0.96f, 1.0f, 0.98f, 1.0f);
    matGlass.roughness = 0.005f;
    matGlass.metallic = 0.0f;
    matGlass.ior = 1.52f;
    matGlass.transmission = 1.0f;
    matGlass.thickness = 1.0f;
    matGlass.type = MATERIAL_DIELECTRIC;
    materials.push_back(matGlass);

    return materials;
}

SceneData ProceduralScene::createInfinityMirrorScene() {
    SceneData scene;
    scene.materials = createInfinityMirrorMaterials();

    auto recordRange = [&](const std::string& name, uint32_t startTri) {
        if (scene.triangles.size() <= startTri) return;
        MeshRange mr{};
        mr.name = name;
        mr.firstTriangle = startTri;
        mr.triangleCount = static_cast<uint32_t>(scene.triangles.size() - startTri);
        for (uint32_t i = startTri; i < scene.triangles.size(); ++i) {
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v0.position));
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v1.position));
            mr.minBound = glm::min(mr.minBound, glm::vec3(scene.triangles[i].v2.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v0.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v1.position));
            mr.maxBound = glm::max(mr.maxBound, glm::vec3(scene.triangles[i].v2.position));
        }
        scene.meshRanges.push_back(mr);
    };

    // 1. Studio diffuse floor (X in [-8.0, 8.0], Z in [-16.0, 16.0])
    uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
    addBox(scene.triangles, glm::vec3(0.0f, -0.05f, 0.0f), glm::vec3(16.0f, 0.1f, 32.0f), 0.0f, 0);
    recordRange("Floor", tStart);

    // 2. Opposing Optical Planar Mirrors
    // Front Mirror at Z = -13.0 (facing +Z into corridor)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3(-1.9f, 0.0f, -13.0f),
            glm::vec3( 1.9f, 0.0f, -13.0f),
            glm::vec3( 1.9f, 3.5f, -13.0f),
            glm::vec3(-1.9f, 3.5f, -13.0f),
            glm::vec3(0.0f, 0.0f, 1.0f), 2);
    recordRange("Mirror_Front", tStart);

    // Rear Mirror at Z = +13.0 (facing -Z into corridor)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addQuad(scene.triangles,
            glm::vec3( 1.9f, 0.0f, 13.0f),
            glm::vec3(-1.9f, 0.0f, 13.0f),
            glm::vec3(-1.9f, 3.5f, 13.0f),
            glm::vec3( 1.9f, 3.5f, 13.0f),
            glm::vec3(0.0f, 0.0f, -1.0f), 2);
    recordRange("Mirror_Back", tStart);

    // 3. Portal Frames along Z
    const float frameZPositions[] = {
        -11.0f, -9.0f, -7.0f, -5.0f, -3.0f, -1.0f, 1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f
    };
    const float frameW = 3.6f;
    const float frameH = 3.4f;
    const float barThick = 0.10f;

    for (size_t i = 0; i < sizeof(frameZPositions) / sizeof(frameZPositions[0]); ++i) {
        float z = frameZPositions[i];
        tStart = static_cast<uint32_t>(scene.triangles.size());

        // Left post
        addBox(scene.triangles, glm::vec3(-frameW * 0.5f + barThick * 0.5f, frameH * 0.5f, z),
               glm::vec3(barThick, frameH, barThick), 0.0f, 1);

        // Right post
        addBox(scene.triangles, glm::vec3(frameW * 0.5f - barThick * 0.5f, frameH * 0.5f, z),
               glm::vec3(barThick, frameH, barThick), 0.0f, 1);

        // Top beam
        addBox(scene.triangles, glm::vec3(0.0f, frameH - barThick * 0.5f, z),
               glm::vec3(frameW, barThick, barThick), 0.0f, 1);

        // Suspended LED luminaire troffer (30cm wide diffuser along Z for soft shadows)
        float lightHalfW = frameW * 0.85f * 0.5f;
        float lightDepth = 0.30f; // 30cm total width along Z
        float lightHalfDepth = lightDepth * 0.5f;
        float lightY = frameH - barThick - 0.05f;

        // Slim luminaire housing under the top beam
        addBox(scene.triangles, glm::vec3(0.0f, lightY + 0.025f, z),
               glm::vec3(2.0f * lightHalfW + 0.04f, 0.05f, lightDepth + 0.04f), 0.0f, 1);

        addQuad(scene.triangles,
                glm::vec3(-lightHalfW, lightY, z - lightHalfDepth),
                glm::vec3( lightHalfW, lightY, z - lightHalfDepth),
                glm::vec3( lightHalfW, lightY, z + lightHalfDepth),
                glm::vec3(-lightHalfW, lightY, z + lightHalfDepth),
                glm::vec3(0.0f, -1.0f, 0.0f), 10);

        // Add corresponding analytical LightGPU
        LightGPU light{};
        light.position = glm::vec4(-lightHalfW, lightY - 0.01f, z - lightHalfDepth, LIGHT_AREA_QUAD);
        light.u = glm::vec4(2.0f * lightHalfW, 0.0f, 0.0f, 0.0f);
        light.v = glm::vec4(0.0f, 0.0f, lightDepth, 0.0f);
        light.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
        light.emission = glm::vec4(20.0f, 18.8f, 16.0f, (2.0f * lightHalfW) * lightDepth);
        scene.lights.push_back(light);

        recordRange("Frame_" + std::to_string(i), tStart);
    }

    // 4. Center Runway Sculptures
    const float pedR = 0.40f;
    const float pedH = 0.20f;

    auto addPedestal = [&](float x, float z, const std::string& name) {
        tStart = static_cast<uint32_t>(scene.triangles.size());
        addCylinder(scene.triangles, glm::vec3(x, pedH * 0.5f, z), pedR, pedH, 9, 24);
        recordRange("Pedestal_" + name, tStart);
    };

    // Item 1: Red Sphere (Z = -6.0)
    addPedestal(0.0f, -6.0f, "RedSphere_1");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addSphere(scene.triangles, glm::vec3(0.0f, pedH + 0.42f, -6.0f), 0.42f, 5, 32, 32);
    recordRange("Art_RedSphere_1", tStart);

    // Item 2: Green Torus (Z = -3.5)
    addPedestal(0.0f, -3.5f, "GreenTorus_1");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addTorus(scene.triangles, glm::vec3(0.0f, pedH + 0.45f, -3.5f), 0.40f, 0.13f, 6, 40, 20, glm::vec3(0.0f, 0.0f, 1.0f));
    recordRange("Art_GreenTorus_1", tStart);

    // Item 3: Yellow Cube (Z = -1.0)
    addPedestal(0.0f, -1.0f, "YellowCube_1");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addBox(scene.triangles, glm::vec3(0.0f, pedH + 0.35f, -1.0f), glm::vec3(0.70f, 0.70f, 0.70f), 25.0f, 7);
    recordRange("Art_YellowCube_1", tStart);

    // Item 4: Blue Cylinder (Z = 1.5)
    addPedestal(0.0f, 1.5f, "BlueCyl_1");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addCylinder(scene.triangles, glm::vec3(0.0f, pedH + 0.85f * 0.5f, 1.5f), 0.34f, 0.85f, 8, 28);
    recordRange("Art_BlueCyl_1", tStart);

    // Item 5: Glass Sphere / Crystal (Z = 4.0)
    addPedestal(0.0f, 4.0f, "GlassSphere");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addSphere(scene.triangles, glm::vec3(0.0f, pedH + 0.42f, 4.0f), 0.42f, 11, 32, 32);
    recordRange("Art_GlassSphere", tStart);

    // Item 6: Green Torus (Z = 6.5)
    addPedestal(0.0f, 6.5f, "GreenTorus_2");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addTorus(scene.triangles, glm::vec3(0.0f, pedH + 0.45f, 6.5f), 0.40f, 0.13f, 6, 40, 20, glm::vec3(0.0f, 0.0f, 1.0f));
    recordRange("Art_GreenTorus_2", tStart);

    // Item 7: Yellow Cube (Z = 9.0)
    addPedestal(0.0f, 9.0f, "YellowCube_2");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addBox(scene.triangles, glm::vec3(0.0f, pedH + 0.35f, 9.0f), glm::vec3(0.70f, 0.70f, 0.70f), 25.0f, 7);
    recordRange("Art_YellowCube_2", tStart);

    // 5. Off-Axis Accent Sculptures
    // Left: Prominent Polished Chrome Mirror Sphere (X = -1.3, Z = -5.5)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addCylinder(scene.triangles, glm::vec3(-1.3f, pedH * 0.5f, -5.5f), 0.50f, pedH, 9, 28);
    recordRange("Pedestal_ChromeSphere", tStart);

    tStart = static_cast<uint32_t>(scene.triangles.size());
    addSphere(scene.triangles, glm::vec3(-1.3f, pedH + 0.70f, -5.5f), 0.70f, 3, 48, 48);
    recordRange("Art_ChromeSphere", tStart);

    // Right: Tilted Polished Gold Accent Ring (X = 1.3, Z = -2.5)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addCylinder(scene.triangles, glm::vec3(1.3f, pedH * 0.5f, -2.5f), 0.40f, pedH, 9, 24);
    recordRange("Pedestal_GoldRing", tStart);

    tStart = static_cast<uint32_t>(scene.triangles.size());
    addTorus(scene.triangles, glm::vec3(1.3f, pedH + 0.42f, -2.5f), 0.38f, 0.10f, 4, 36, 18,
             glm::normalize(glm::vec3(0.35f, 0.88f, 0.25f)));
    recordRange("Art_GoldRing", tStart);

    // 6. Camera Setup
    scene.hasCamera = true;
    scene.cameraPosition = glm::vec3(0.75f, 1.65f, -11.5f);
    scene.cameraTarget = glm::vec3(-0.15f, 1.25f, 2.0f);
    scene.cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    scene.cameraFov = 50.0f;

    // Bounds and targets
    scene.boundsMin = glm::vec3(-3.0f, -0.2f, -14.0f);
    scene.boundsMax = glm::vec3(3.0f, 4.0f, 14.0f);
    scene.sceneRadius = 16.0f;
    scene.focalBoundsMin = glm::vec3(-2.0f, 0.0f, -12.0f);
    scene.focalBoundsMax = glm::vec3(2.0f, 3.5f, 12.0f);
    scene.focalRadius = 14.0f;
    scene.focalDistance = glm::length(scene.cameraTarget - scene.cameraPosition);
    scene.centralTarget = glm::vec3(0.0f, 1.2f, 0.0f);

    scene.hasDielectrics = true;
    scene.dielectricBoundsMin = glm::vec3(-0.5f, 0.2f, 3.5f);
    scene.dielectricBoundsMax = glm::vec3(0.5f, 1.1f, 4.5f);

    return scene;
}

static inline bool intersectRayAABB(const glm::vec3& orig, const glm::vec3& invDir,
                                    const glm::vec3& bmin, const glm::vec3& bmax,
                                    float& tmin, float& tmax) {
    glm::vec3 t0 = (bmin - orig) * invDir;
    glm::vec3 t1 = (bmax - orig) * invDir;
    glm::vec3 tsmall = glm::min(t0, t1);
    glm::vec3 tbig = glm::max(t0, t1);

    tmin = std::max(std::max(tsmall.x, tsmall.y), tsmall.z);
    tmax = std::min(std::min(tbig.x, tbig.y), tbig.z);

    return tmax >= std::max(0.0f, tmin);
}

static inline bool intersectRayTriangle(const glm::vec3& orig, const glm::vec3& dir,
                                       const glm::vec3& v0, const glm::vec3& v1, const glm::vec3& v2,
                                       float& t) {
    const float EPSILON = 1e-7f;
    glm::vec3 edge1 = v1 - v0;
    glm::vec3 edge2 = v2 - v0;
    glm::vec3 h = glm::cross(dir, edge2);
    float a = glm::dot(edge1, h);
    if (a > -EPSILON && a < EPSILON) return false;

    float f = 1.0f / a;
    glm::vec3 s = orig - v0;
    float u = f * glm::dot(s, h);
    if (u < 0.0f || u > 1.0f) return false;

    glm::vec3 q = glm::cross(s, edge1);
    float v = f * glm::dot(dir, q);
    if (v < 0.0f || u + v > 1.0f) return false;

    float dist = f * glm::dot(edge2, q);
    if (dist > 0.01f) {
        t = dist;
        return true;
    }
    return false;
}

static inline bool intersectRaySphere(const glm::vec3& orig, const glm::vec3& dir,
                                     const glm::vec3& center, float radius, float& t) {
    glm::vec3 oc = orig - center;
    float b = glm::dot(oc, dir);
    float c = glm::dot(oc, oc) - radius * radius;
    float discriminant = b * b - c;
    if (discriminant < 0.0f) return false;
    float sqrtDisc = std::sqrt(discriminant);
    float t0 = -b - sqrtDisc;
    float t1 = -b + sqrtDisc;
    if (t0 > 0.01f) {
        t = t0;
        return true;
    }
    if (t1 > 0.01f) {
        t = t1;
        return true;
    }
    return false;
}

bool SceneData::raycast(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float maxDist,
                        float& outHitDist, glm::vec3& outHitPoint, std::string* outHitName) const {
    float closestT = maxDist;
    bool hit = false;
    std::string hitName = "";

    // Test GPU spheres
    for (size_t i = 0; i < spheres.size(); ++i) {
        glm::vec3 center = glm::vec3(spheres[i].centerRadius);
        float radius = spheres[i].centerRadius.w;
        float t = 0.0f;
        if (intersectRaySphere(rayOrigin, rayDir, center, radius, t) && t < closestT) {
            closestT = t;
            hit = true;
            hitName = "Sphere_" + std::to_string(i);
        }
    }

    // Safe inverse direction avoiding division by 0
    glm::vec3 safeDir = rayDir;
    for (int k = 0; k < 3; ++k) {
        if (std::abs(safeDir[k]) < 1e-8f) safeDir[k] = (safeDir[k] < 0.0f ? -1e-8f : 1e-8f);
    }
    glm::vec3 invDir = 1.0f / safeDir;

    // Test instances if present (e.g. cyber-city)
    if (!instances.empty()) {
        for (size_t instIdx = 0; instIdx < instances.size(); ++instIdx) {
            const auto& inst = instances[instIdx];
            if (inst.blasIndex >= blasRanges.size()) continue;
            const auto& range = blasRanges[inst.blasIndex];

            glm::mat4 invTransform = glm::inverse(inst.transform);
            glm::vec3 localOrig = glm::vec3(invTransform * glm::vec4(rayOrigin, 1.0f));
            glm::vec3 localDir = glm::vec3(invTransform * glm::vec4(rayDir, 0.0f));

            if (inst.blasIndex < meshRanges.size()) {
                const auto& mr = meshRanges[inst.blasIndex];
                glm::vec3 safeLocalDir = localDir;
                for (int k = 0; k < 3; ++k) {
                    if (std::abs(safeLocalDir[k]) < 1e-8f) safeLocalDir[k] = (safeLocalDir[k] < 0.0f ? -1e-8f : 1e-8f);
                }
                glm::vec3 invLocalDir = 1.0f / safeLocalDir;
                float tmin, tmax;
                if (!intersectRayAABB(localOrig, invLocalDir, mr.minBound, mr.maxBound, tmin, tmax) || tmin >= closestT) {
                    continue;
                }
            }

            uint32_t endTri = std::min<uint32_t>(range.firstTriangle + range.triangleCount, static_cast<uint32_t>(triangles.size()));
            for (uint32_t i = range.firstTriangle; i < endTri; ++i) {
                float t = 0.0f;
                if (intersectRayTriangle(localOrig, localDir,
                                         glm::vec3(triangles[i].v0.position),
                                         glm::vec3(triangles[i].v1.position),
                                         glm::vec3(triangles[i].v2.position), t)) {
                    if (t < closestT) {
                        closestT = t;
                        hit = true;
                        if (inst.blasIndex < meshRanges.size()) {
                            hitName = meshRanges[inst.blasIndex].name + "_" + std::to_string(instIdx);
                        } else {
                            hitName = "Instance_" + std::to_string(instIdx);
                        }
                    }
                }
            }
        }
    } else if (!meshRanges.empty()) {
        for (const auto& mr : meshRanges) {
            float tmin, tmax;
            if (intersectRayAABB(rayOrigin, invDir, mr.minBound, mr.maxBound, tmin, tmax) && tmin < closestT) {
                uint32_t endTri = std::min<uint32_t>(mr.firstTriangle + mr.triangleCount, static_cast<uint32_t>(triangles.size()));
                for (uint32_t i = mr.firstTriangle; i < endTri; ++i) {
                    float t = 0.0f;
                    if (intersectRayTriangle(rayOrigin, rayDir,
                                             glm::vec3(triangles[i].v0.position),
                                             glm::vec3(triangles[i].v1.position),
                                             glm::vec3(triangles[i].v2.position), t)) {
                        if (t < closestT) {
                            closestT = t;
                            hit = true;
                            hitName = mr.name;
                        }
                    }
                }
            }
        }
    } else {
        // Fallback: test all triangles
        for (size_t i = 0; i < triangles.size(); ++i) {
            float t = 0.0f;
            if (intersectRayTriangle(rayOrigin, rayDir,
                                     glm::vec3(triangles[i].v0.position),
                                     glm::vec3(triangles[i].v1.position),
                                     glm::vec3(triangles[i].v2.position), t)) {
                if (t < closestT) {
                    closestT = t;
                    hit = true;
                    hitName = "Triangle_" + std::to_string(i);
                }
            }
        }
    }

    if (hit) {
        outHitDist = closestT;
        outHitPoint = rayOrigin + rayDir * closestT;
        if (outHitName) *outHitName = hitName;
        return true;
    }
    return false;
}

} // namespace pathways
