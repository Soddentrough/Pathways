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

static void addBeam(std::vector<TriangleGPU>& triangles,
                    glm::vec3 pA, glm::vec3 pB, float width, float thickness, uint32_t matId) {
    glm::vec3 dir = pB - pA;
    float len = glm::length(dir);
    if (len < 1e-4f) return;
    dir /= len;

    glm::vec3 up = std::abs(dir.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 u = glm::normalize(glm::cross(dir, up));
    glm::vec3 v = glm::cross(u, dir);

    float hw = width * 0.5f;
    float ht = thickness * 0.5f;

    glm::vec3 p0 = pA - u * hw - v * ht;
    glm::vec3 p1 = pA + u * hw - v * ht;
    glm::vec3 p2 = pA + u * hw + v * ht;
    glm::vec3 p3 = pA - u * hw + v * ht;

    glm::vec3 p4 = pB - u * hw - v * ht;
    glm::vec3 p5 = pB + u * hw - v * ht;
    glm::vec3 p6 = pB + u * hw + v * ht;
    glm::vec3 p7 = pB - u * hw + v * ht;

    // Bottom cap at pA (-dir)
    addQuad(triangles, p0, p3, p2, p1, -dir, matId);
    // Top cap at pB (+dir)
    addQuad(triangles, p4, p5, p6, p7, dir, matId);
    // Side 1 (-u)
    addQuad(triangles, p0, p4, p7, p3, -u, matId);
    // Side 2 (+u)
    addQuad(triangles, p1, p2, p6, p5, u, matId);
    // Side 3 (-v)
    addQuad(triangles, p0, p1, p5, p4, -v, matId);
    // Side 4 (+v)
    addQuad(triangles, p3, p7, p6, p2, v, matId);
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

            // Add visible emissive quad mesh on ceiling (material 3: white emissive 18, 18, 15)
            addQuad(scene.triangles,
                    glm::vec3(cx - halfSize, 1.99f, cz - halfSize),
                    glm::vec3(cx + halfSize, 1.99f, cz - halfSize),
                    glm::vec3(cx + halfSize, 1.99f, cz + halfSize),
                    glm::vec3(cx - halfSize, 1.99f, cz + halfSize),
                    glm::vec3(0.0f, -1.0f, 0.0f), 3);

            // Add corresponding physical area light matching the white ceiling quad emission for unbiased MIS
            LightGPU light{};
            light.position = glm::vec4(cx - halfSize, 1.99f, cz - halfSize, LIGHT_AREA_QUAD);
            light.u = glm::vec4(2.0f * halfSize, 0.0f, 0.0f, 0.0f);
            light.v = glm::vec4(0.0f, 0.0f, 2.0f * halfSize, 0.0f);
            light.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
            light.emission = glm::vec4(18.0f, 18.0f, 15.0f, area);
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
        // 1. Heavy Industrial Base Platform & Chassis (Mat 19 Gunmetal, Mat 1 Carbon Steel, Mat 2 Titanium)
        addBox(scene.triangles, glm::vec3(0.0f, 0.16f, 0.0f), glm::vec3(5.8f, 0.32f, 5.8f), 0.0f, 19);
        // Heavy structural corner footpads / armor shoes
        for (float sx : {-2.35f, 2.35f}) {
            for (float sz : {-2.35f, 2.35f}) {
                addBox(scene.triangles, glm::vec3(sx, 0.18f, sz), glm::vec3(1.10f, 0.36f, 1.10f), 0.0f, 1);
            }
        }
        // Stepped titanium collar plate
        addBox(scene.triangles, glm::vec3(0.0f, 0.38f, 0.0f), glm::vec3(4.9f, 0.14f, 4.9f), 0.0f, 2);

        // Perimeter copper heatsink cooling louvres around chassis
        for (int ang = 0; ang < 360; ang += 45) {
            float rad = glm::radians(static_cast<float>(ang));
            float hx = 2.35f * std::cos(rad);
            float hz = 2.35f * std::sin(rad);
            addBox(scene.triangles, glm::vec3(hx, 0.36f, hz), glm::vec3(0.24f, 0.12f, 0.24f), 0.0f, 10);
        }

        // Telemetry status LED banks on front & rear chassis faces (Mat 45 Turquoise, Mat 35 Gold)
        addBox(scene.triangles, glm::vec3(0.0f, 0.26f,  2.91f), glm::vec3(2.4f, 0.06f, 0.02f), 0.0f, 45);
        addBox(scene.triangles, glm::vec3(0.0f, 0.18f,  2.91f), glm::vec3(1.6f, 0.04f, 0.02f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3(0.0f, 0.26f, -2.91f), glm::vec3(2.4f, 0.06f, 0.02f), 0.0f, 45);
        addBox(scene.triangles, glm::vec3(0.0f, 0.18f, -2.91f), glm::vec3(1.6f, 0.04f, 0.02f), 0.0f, 35);

        // 2. Precision Recessed Optical Emitter Basin (Matte gunmetal basin with segmented neon ring and central lens)
        // Outer titanium containment collar (Mat 2)
        addCylinder(scene.triangles, glm::vec3(0.0f, 0.48f, 0.0f), 2.30f, 0.08f, 2, 24);
        // Recessed gunmetal projector basin well (Mat 19)
        addCylinder(scene.triangles, glm::vec3(0.0f, 0.52f, 0.0f), 2.10f, 0.08f, 19, 24);

        // Hollow segmented concentric neon ring (Mat 45 Electric Turquoise) - crisp circular accent, zero interior fill glare
        for (int i = 0; i < 24; ++i) {
            float deg = static_cast<float>(i) * 15.0f;
            float rad = glm::radians(deg);
            float rx = 1.80f * std::cos(rad);
            float rz = 1.80f * std::sin(rad);
            addBox(scene.triangles, glm::vec3(rx, 0.55f, rz), glm::vec3(0.48f, 0.025f, 0.05f), -deg, 45);
        }

        // Radial titanium aperture vanes around well (Mat 2 Titanium)
        for (int ang = 0; ang < 360; ang += 45) {
            float rad = glm::radians(static_cast<float>(ang));
            float px = 1.45f * std::cos(rad);
            float pz = 1.45f * std::sin(rad);
            addBox(scene.triangles, glm::vec3(px, 0.54f, pz), glm::vec3(0.42f, 0.03f, 0.08f), static_cast<float>(-ang), 2);
        }

        // Central optical projector emitter lens (Mat 30 Cyan Emitter) & laser iris core (Mat 45)
        addCylinder(scene.triangles, glm::vec3(0.0f, 0.56f, 0.0f), 0.65f, 0.06f, 30, 24);
        addCylinder(scene.triangles, glm::vec3(0.0f, 0.59f, 0.0f), 0.28f, 0.03f, 45, 24);

        // 8 Radial magnetic emitter pods / beam focus coils around perimeter well
        for (int ang = 0; ang < 360; ang += 45) {
            float rad = glm::radians(static_cast<float>(ang));
            float px = 2.15f * std::cos(rad);
            float pz = 2.15f * std::sin(rad);
            addBox(scene.triangles, glm::vec3(px, 0.56f, pz), glm::vec3(0.18f, 0.08f, 0.18f), 0.0f, 2);
            addBox(scene.triangles, glm::vec3(px * 0.94f, 0.58f, pz * 0.94f), glm::vec3(0.08f, 0.06f, 0.08f), 0.0f, 45);
        }

        // 3. Cantilevered Overhead Suspension Gantry (Unobstructed Frontal Viewing!)
        const float gimbalY = 3.55f;
        const float overheadLensY = 3.32f;

        // Rear Structural Gantry Masts at fz = -2.35f
        for (float fx : {-2.35f, 2.35f}) {
            // Heavy gunmetal pylon base footing on top of corner armor footpad
            addBox(scene.triangles, glm::vec3(fx, 0.55f, -2.35f), glm::vec3(0.50f, 0.28f, 0.50f), 0.0f, 19);
            // Main vertical channeled pylon column
            addBox(scene.triangles, glm::vec3(fx, 1.40f, -2.35f), glm::vec3(0.34f, 1.45f, 0.34f), 0.0f, 19);
            // Waveguide neon accent strip
            addBox(scene.triangles, glm::vec3(fx * 0.95f, 1.40f, -2.35f), glm::vec3(0.08f, 1.40f, 0.08f), 0.0f, 45);
            // Articulated titanium gantry collar node
            addBox(scene.triangles, glm::vec3(fx, 2.20f, -2.35f), glm::vec3(0.42f, 0.18f, 0.42f), 0.0f, 2);
            addBox(scene.triangles, glm::vec3(fx, 2.20f, -2.35f), glm::vec3(0.46f, 0.06f, 0.46f), 0.0f, 45);

            // Cantilever boom truss reaching from rear mast node up and forward to overhead gimbal
            glm::vec3 pBot(fx, 2.20f, -2.35f);
            glm::vec3 pTop(fx * 0.28f, gimbalY, -0.25f);
            addBeam(scene.triangles, pBot, pTop, 0.15f, 0.15f, 2);

            // Secondary cantilever gusset truss
            glm::vec3 pGussetBot(fx, 1.65f, -2.35f);
            glm::vec3 pGussetTop(fx * 0.58f, 2.85f, -1.25f);
            addBeam(scene.triangles, pGussetBot, pGussetTop, 0.10f, 0.10f, 19);
        }

        // Heavy rear cross-member connecting rear masts (Mat 1 Carbon Steel)
        addBeam(scene.triangles, glm::vec3(-2.35f, 2.20f, -2.35f), glm::vec3(2.35f, 2.20f, -2.35f), 0.14f, 0.14f, 1);
        // Central K-truss strut from cross-member to overhead gimbal
        addBeam(scene.triangles, glm::vec3(0.0f, 2.20f, -2.35f), glm::vec3(0.0f, gimbalY, -0.45f), 0.12f, 0.12f, 2);

        // Front Field-Containment & Telemetry Pedestals at fz = +2.35f (Low profile, wide clearance, zero screen obstruction)
        for (float fx : {-2.35f, 2.35f}) {
            addBox(scene.triangles, glm::vec3(fx, 0.55f, 2.35f), glm::vec3(0.46f, 0.28f, 0.46f), 0.0f, 19);
            addBox(scene.triangles, glm::vec3(fx, 0.95f, 2.35f), glm::vec3(0.28f, 0.55f, 0.28f), 0.0f, 19);
            addBox(scene.triangles, glm::vec3(fx, 1.25f, 2.35f), glm::vec3(0.36f, 0.10f, 0.36f), 0.0f, 2);
            addBox(scene.triangles, glm::vec3(fx, 1.32f, 2.35f), glm::vec3(0.16f, 0.06f, 0.16f), 0.0f, 45);

            // Lateral perimeter side handrails (along x = +/-2.35f, completely outside display frustum)
            addBeam(scene.triangles, glm::vec3(fx, 1.25f, -2.35f), glm::vec3(fx, 1.25f, 2.35f), 0.07f, 0.07f, 2);
            // Lateral diagonal stabilizing brace along side plane
            addBeam(scene.triangles, glm::vec3(fx, 1.25f, 1.20f), glm::vec3(fx, 2.20f, -1.20f), 0.08f, 0.08f, 19);
        }

        // Overhead projector suspension gimbal (Mat 2 Titanium)
        addCylinder(scene.triangles, glm::vec3(0.0f, gimbalY, 0.0f), 1.05f, 0.08f, 2, 24);
        // Inner magnetic containment ring (Mat 45 Electric Turquoise)
        addCylinder(scene.triangles, glm::vec3(0.0f, gimbalY - 0.02f, 0.0f), 0.88f, 0.04f, 45, 24);
        // Projector housing body (Mat 19 Gunmetal)
        addBox(scene.triangles, glm::vec3(0.0f, gimbalY + 0.18f, 0.0f), glm::vec3(0.85f, 0.28f, 0.85f), 0.0f, 19);
        // Heatsink cooling ribs on top of projector housing (Mat 27 Steel)
        addBox(scene.triangles, glm::vec3(0.0f, gimbalY + 0.35f, 0.0f), glm::vec3(0.70f, 0.06f, 0.70f), 0.0f, 27);
        // Multi-stage optical snout (Mat 2 Titanium)
        addCylinder(scene.triangles, glm::vec3(0.0f, gimbalY - 0.10f, 0.0f), 0.48f, 0.14f, 2, 24);
        // Downward compound emitter lens (Mat 30 Cyan Emitter)
        addSphere(scene.triangles, glm::vec3(0.0f, overheadLensY, 0.0f), 0.25f, 30, 16, 16);
        // Concentric neon focus ring around lens (Mat 45 Electric Turquoise)
        addCylinder(scene.triangles, glm::vec3(0.0f, overheadLensY + 0.03f, 0.0f), 0.38f, 0.03f, 45, 24);

        // 4. Upright 3D Curved Hologram Display: Full 16:9 Widescreen Continuous Parametric Mesh (32x18 quads = 1152 triangles)
        // Watertight parametric curved quad mesh with shared vertex positions and continuous (u, v) coordinates.
        const uint32_t NX = 32;
        const uint32_t NY = 18;
        const float W = 3.20f;
        const float H = 1.80f;
        const float yBase = 0.95f;

        struct DisplayVertex {
            glm::vec3 pos;
            glm::vec3 normal;
            glm::vec4 tangent;
            glm::vec2 uv;
        };

        std::vector<std::vector<DisplayVertex>> grid(NY + 1, std::vector<DisplayVertex>(NX + 1));
        for (uint32_t j = 0; j <= NY; ++j) {
            float normY = static_cast<float>(j) / static_cast<float>(NY);
            float y = yBase + normY * H;
            float v = 1.0f - normY;

            for (uint32_t i = 0; i <= NX; ++i) {
                float normX = static_cast<float>(i) / static_cast<float>(NX);
                float x = -W * 0.5f + normX * W;
                float u = normX;

                // Gentle cylindrical concave curvature facing the terrace viewer
                float xRel = x / (W * 0.5f);
                float z = -0.16f * (1.0f - xRel * xRel);

                // Tangent along +X / +U, normal facing +Z
                // dz/dx = 0.20 * xRel
                glm::vec3 tanDir = glm::normalize(glm::vec3(1.0f, 0.0f, 0.20f * xRel));
                glm::vec3 normal = glm::normalize(glm::vec3(-0.20f * xRel, 0.0f, 1.0f));

                grid[j][i] = {
                    glm::vec3(x, y, z),
                    normal,
                    glm::vec4(tanDir, 1.0f),
                    glm::vec2(u, v)
                };
            }
        }

        for (uint32_t iy = 0; iy < NY; ++iy) {
            for (uint32_t ix = 0; ix < NX; ++ix) {
                const auto& v00 = grid[iy][ix];         // bottom-left
                const auto& v10 = grid[iy][ix + 1];     // bottom-right
                const auto& v11 = grid[iy + 1][ix + 1]; // top-right
                const auto& v01 = grid[iy + 1][ix];     // top-left

                TriangleGPU t1{};
                t1.v0.position = glm::vec4(v00.pos, v00.uv.x);
                t1.v0.normal   = glm::vec4(v00.normal, v00.uv.y);
                t1.v0.tangent  = v00.tangent;
                t1.v1.position = glm::vec4(v10.pos, v10.uv.x);
                t1.v1.normal   = glm::vec4(v10.normal, v10.uv.y);
                t1.v1.tangent  = v10.tangent;
                t1.v2.position = glm::vec4(v11.pos, v11.uv.x);
                t1.v2.normal   = glm::vec4(v11.normal, v11.uv.y);
                t1.v2.tangent  = v11.tangent;
                t1.materialId  = 36;

                TriangleGPU t2{};
                t2.v0.position = glm::vec4(v00.pos, v00.uv.x);
                t2.v0.normal   = glm::vec4(v00.normal, v00.uv.y);
                t2.v0.tangent  = v00.tangent;
                t2.v1.position = glm::vec4(v11.pos, v11.uv.x);
                t2.v1.normal   = glm::vec4(v11.normal, v11.uv.y);
                t2.v1.tangent  = v11.tangent;
                t2.v2.position = glm::vec4(v01.pos, v01.uv.x);
                t2.v2.normal   = glm::vec4(v01.normal, v01.uv.y);
                t2.v2.tangent  = v01.tangent;
                t2.materialId  = 36;

                scene.triangles.push_back(t1);
                scene.triangles.push_back(t2);
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

    // 12: Gold Pedestal Area Light (Warm amber/golden LED luminaire)
    MaterialGPU matGoldLight{};
    matGoldLight.albedo = glm::vec4(1.0f, 0.85f, 0.55f, 1.0f);
    matGoldLight.emissive = glm::vec4(28.0f, 22.0f, 12.0f, 1.0f);
    matGoldLight.type = MATERIAL_EMISSIVE;
    materials.push_back(matGoldLight);

    // 13: Perimeter Frame Neon Strip (Ice Blue Luminescent Conduit)
    MaterialGPU matNeon{};
    matNeon.albedo = glm::vec4(0.70f, 0.90f, 1.0f, 1.0f);
    matNeon.emissive = glm::vec4(16.0f, 20.0f, 26.0f, 1.0f);
    matNeon.type = MATERIAL_EMISSIVE;
    materials.push_back(matNeon);

    // 14: Focused Caustic Spotlight Lens / Emitter (Cool White Beam)
    MaterialGPU matSpotLens{};
    matSpotLens.albedo = glm::vec4(0.95f, 0.98f, 1.0f, 1.0f);
    matSpotLens.emissive = glm::vec4(120.0f, 125.0f, 140.0f, 1.0f);
    matSpotLens.type = MATERIAL_EMISSIVE;
    materials.push_back(matSpotLens);

    // 15: Gyroscope Outer Ring (Polished Platinum / Titanium)
    MaterialGPU matPlatinum{};
    matPlatinum.albedo = glm::vec4(0.92f, 0.93f, 0.96f, 1.0f);
    matPlatinum.roughness = 0.008f;
    matPlatinum.metallic = 1.0f;
    matPlatinum.type = MATERIAL_METALLIC;
    materials.push_back(matPlatinum);

    // 16: Gyroscope Middle Ring (Polished Rose Gold / Copper)
    MaterialGPU matCopper{};
    matCopper.albedo = glm::vec4(0.97f, 0.72f, 0.58f, 1.0f);
    matCopper.roughness = 0.012f;
    matCopper.metallic = 1.0f;
    matCopper.type = MATERIAL_METALLIC;
    materials.push_back(matCopper);

    // 17: Gyroscope Inner Ring (Dark Mirror Chrome)
    MaterialGPU matDarkChrome{};
    matDarkChrome.albedo = glm::vec4(0.82f, 0.84f, 0.88f, 1.0f);
    matDarkChrome.roughness = 0.005f;
    matDarkChrome.metallic = 1.0f;
    matDarkChrome.type = MATERIAL_METALLIC;
    materials.push_back(matDarkChrome);

    return materials;
}

SceneData ProceduralScene::createInfinityMirrorScene() {
    SceneData scene;
    scene.materials = createInfinityMirrorMaterials();

    auto recordRange = [&](const std::string& /*name*/, uint32_t /*startTri*/) {
        // Individual sub-ranges are not added to meshRanges because meshRanges in an instanced scene
        // indexes BLAS prototypes 1:1 with blasRanges. BLAS prototypes are recorded via recordBlasPrototype.
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

        const float quadY = lightY - 0.001f;
        addQuad(scene.triangles,
                glm::vec3(-lightHalfW, quadY, z - lightHalfDepth),
                glm::vec3( lightHalfW, quadY, z - lightHalfDepth),
                glm::vec3( lightHalfW, quadY, z + lightHalfDepth),
                glm::vec3(-lightHalfW, quadY, z + lightHalfDepth),
                glm::vec3(0.0f, -1.0f, 0.0f), 10);

        // Add corresponding analytical LightGPU
        LightGPU light{};
        light.position = glm::vec4(-lightHalfW, quadY, z - lightHalfDepth, LIGHT_AREA_QUAD);
        light.u = glm::vec4(2.0f * lightHalfW, 0.0f, 0.0f, 0.0f);
        light.v = glm::vec4(0.0f, 0.0f, lightDepth, 0.0f);
        light.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
        light.emission = glm::vec4(20.0f, 18.8f, 16.0f, (2.0f * lightHalfW) * lightDepth);
        scene.lights.push_back(light);

        // Emissive Perimeter Frame Neon Conduit (Luminescent trim tracing the rectangular portal)
        const float neonW = 0.035f;
        const float neonD = 0.008f;
        const float zFront = z + barThick * 0.5f + neonD * 0.5f;
        const float zBack = z - barThick * 0.5f - neonD * 0.5f;

        // Front Face Neon Trim (Facing +Z into corridor/front mirror)
        addBox(scene.triangles, glm::vec3(-frameW * 0.5f + barThick * 0.5f, frameH * 0.5f, zFront),
               glm::vec3(neonW, frameH, neonD), 0.0f, 13 /* matNeon */);
        addBox(scene.triangles, glm::vec3(frameW * 0.5f - barThick * 0.5f, frameH * 0.5f, zFront),
               glm::vec3(neonW, frameH, neonD), 0.0f, 13 /* matNeon */);
        addBox(scene.triangles, glm::vec3(0.0f, frameH - barThick * 0.5f, zFront),
               glm::vec3(frameW, neonW, neonD), 0.0f, 13 /* matNeon */);

        // Rear Face Neon Trim (Facing -Z into corridor/rear mirror)
        addBox(scene.triangles, glm::vec3(-frameW * 0.5f + barThick * 0.5f, frameH * 0.5f, zBack),
               glm::vec3(neonW, frameH, neonD), 0.0f, 13 /* matNeon */);
        addBox(scene.triangles, glm::vec3(frameW * 0.5f - barThick * 0.5f, frameH * 0.5f, zBack),
               glm::vec3(neonW, frameH, neonD), 0.0f, 13 /* matNeon */);
        addBox(scene.triangles, glm::vec3(0.0f, frameH - barThick * 0.5f, zBack),
               glm::vec3(frameW, neonW, neonD), 0.0f, 13 /* matNeon */);

        // Inner Reveal Strips (Facing inward toward corridor center)
        addBox(scene.triangles, glm::vec3(-frameW * 0.5f + barThick + 0.002f, (frameH - barThick) * 0.5f, z),
               glm::vec3(0.006f, frameH - barThick, barThick * 0.8f), 0.0f, 13 /* matNeon */);
        addBox(scene.triangles, glm::vec3(frameW * 0.5f - barThick - 0.002f, (frameH - barThick) * 0.5f, z),
               glm::vec3(0.006f, frameH - barThick, barThick * 0.8f), 0.0f, 13 /* matNeon */);

        recordRange("Frame_" + std::to_string(i), tStart);
    }

    const float pedR = 0.40f;
    const float pedH = 0.20f;

    // 3b. Freestanding Architectural Standing Spotlight Luminaires
    // Authentic museum-grade floor-standing spotlight fixtures with heavy cast base,
    // slender vertical stanchion mast, articulated gold knuckle, and aimed lamp cowl.
    auto addStandingSpotlight = [&](const glm::vec3& basePos, const glm::vec3& targetPos,
                                   const glm::vec4& emission, const std::string& name) {
        tStart = static_cast<uint32_t>(scene.triangles.size());
        // Heavy circular weighted floor base
        addCylinder(scene.triangles, glm::vec3(basePos.x, 0.015f, basePos.z), 0.18f, 0.03f, 1 /* matFrame */, 24);
        // Beveled gold collar accent
        addCylinder(scene.triangles, glm::vec3(basePos.x, 0.035f, basePos.z), 0.06f, 0.02f, 4 /* matGold */, 16);

        // Vertical slender stanchion mast
        const float mastHeight = 2.10f;
        addBeam(scene.triangles, glm::vec3(basePos.x, 0.04f, basePos.z),
                glm::vec3(basePos.x, mastHeight, basePos.z), 0.030f, 0.030f, 1 /* matFrame */);

        // Mid-height adjustment screw / knob
        addSphere(scene.triangles, glm::vec3(basePos.x + 0.025f, 1.20f, basePos.z), 0.022f, 4 /* matGold */, 12, 12);

        // Articulated swivel hinge knuckle at mast top
        addSphere(scene.triangles, glm::vec3(basePos.x, mastHeight, basePos.z), 0.038f, 4 /* matGold */, 16, 16);

        // Cantilever boom arm extending toward the target
        glm::vec3 boomDir = glm::normalize(glm::vec3(targetPos.x - basePos.x, 0.0f, targetPos.z - basePos.z));
        glm::vec3 headPos = glm::vec3(basePos.x, mastHeight, basePos.z) + boomDir * 0.30f + glm::vec3(0.0f, 0.12f, 0.0f);
        addBeam(scene.triangles, glm::vec3(basePos.x, mastHeight, basePos.z), headPos, 0.026f, 0.026f, 1 /* matFrame */);

        // Spotlight cowl / head barrel aimed at targetPos
        glm::vec3 spotDir = glm::normalize(targetPos - headPos);
        glm::vec3 lensPos = headPos + spotDir * 0.14f;
        addBeam(scene.triangles, headPos - spotDir * 0.06f, lensPos, 0.12f, 0.12f, 1 /* matFrame */);

        // Emissive lens quad
        glm::vec3 upRef(0.0f, 1.0f, 0.0f);
        glm::vec3 spotRight = glm::normalize(glm::cross(spotDir, upRef));
        glm::vec3 spotUp = glm::cross(spotRight, spotDir);
        float lensR = 0.060f;
        addQuad(scene.triangles,
                lensPos - spotRight * lensR - spotUp * lensR,
                lensPos + spotRight * lensR - spotUp * lensR,
                lensPos + spotRight * lensR + spotUp * lensR,
                lensPos - spotRight * lensR + spotUp * lensR,
                spotDir, 14 /* matSpotLens */);

        // Analytical spotlight emitter
        LightGPU spotLight{};
        spotLight.position = glm::vec4(lensPos, LIGHT_SPOT);
        spotLight.normal = glm::vec4(spotDir, 0.0f);
        spotLight.u = glm::vec4(0.0f, 0.0f, 0.0f, std::cos(glm::radians(15.0f)));
        spotLight.v = glm::vec4(0.0f, 0.0f, 0.0f, std::cos(glm::radians(24.0f)));
        spotLight.emission = emission;
        scene.lights.push_back(spotLight);
        recordRange("StandingSpot_" + name, tStart);
    };

    // 4. Center Runway Sculptures (Spacious 3.5m rhythmic spacing from Z = -7.0 to +10.0)

    auto addPedestal = [&](float x, float z, const std::string& name) {
        tStart = static_cast<uint32_t>(scene.triangles.size());
        addCylinder(scene.triangles, glm::vec3(x, pedH * 0.5f, z), pedR, pedH, 9, 24);
        recordRange("Pedestal_" + name, tStart);
    };

    // Item 1: Red Sphere (Z = -7.0)
    addPedestal(0.0f, -7.0f, "RedSphere_1");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addSphere(scene.triangles, glm::vec3(0.0f, pedH + 0.42f, -7.0f), 0.42f, 5, 32, 32);
    recordRange("Art_RedSphere_1", tStart);

    // Item 2: Pedestal for Rotating Reflective Torus (Z = -3.5)
    addPedestal(0.0f, -3.5f, "ReflectiveTorus_1");

    // Item 3: Yellow Cube (Z = 0.0)
    addPedestal(0.0f, 0.0f, "YellowCube_1");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addBox(scene.triangles, glm::vec3(0.0f, pedH + 0.35f, 0.0f), glm::vec3(0.70f, 0.70f, 0.70f), 25.0f, 7);
    recordRange("Art_YellowCube_1", tStart);

    // Item 4: Glass Sphere / Crystal (Z = 3.5)
    addPedestal(0.0f, 3.5f, "GlassSphere");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addSphere(scene.triangles, glm::vec3(0.0f, pedH + 0.42f, 3.5f), 0.42f, 11, 32, 32);
    recordRange("Art_GlassSphere", tStart);

    // Item 5: Blue Cylinder (Z = 7.0)
    addPedestal(0.0f, 7.0f, "BlueCyl_1");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addCylinder(scene.triangles, glm::vec3(0.0f, pedH + 0.85f * 0.5f, 7.0f), 0.34f, 0.85f, 8, 28);
    recordRange("Art_BlueCyl_1", tStart);

    // Item 6: Green Torus (Z = 10.0)
    addPedestal(0.0f, 10.0f, "GreenTorus_2");
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addTorus(scene.triangles, glm::vec3(0.0f, pedH + 0.45f, 10.0f), 0.40f, 0.13f, 6, 40, 20, glm::vec3(0.0f, 0.0f, 1.0f));
    recordRange("Art_GreenTorus_2", tStart);

    // 5. Off-Axis Accent Sculptures & Standing Spotlight Luminaires
    // Right (screen-right): Prominent Polished Chrome Mirror Sphere (X = -0.92, Z = -5.0)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    addCylinder(scene.triangles, glm::vec3(-0.92f, pedH * 0.5f, -5.0f), 0.50f, pedH, 9, 28);
    recordRange("Pedestal_ChromeSphere", tStart);

    tStart = static_cast<uint32_t>(scene.triangles.size());
    addSphere(scene.triangles, glm::vec3(-0.92f, pedH + 0.70f, -5.0f), 0.70f, 3, 48, 48);
    recordRange("Art_ChromeSphere", tStart);

    // Left Mid-Foreground: Pedestal & Precision Gimbal Fork for Multi-Ring Gyroscopic Sculpture (X = 1.3, Z = -1.5)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    // Glowing base plinth at floor level (Ice blue neon trim)
    addCylinder(scene.triangles, glm::vec3(1.3f, 0.015f, -1.5f), 0.48f, 0.03f, 13 /* matNeon */, 28);
    // Dark pedestal column
    addCylinder(scene.triangles, glm::vec3(1.3f, pedH * 0.5f, -1.5f), 0.44f, pedH, 9 /* matPed */, 24);
    // Metallic bezel platform
    addCylinder(scene.triangles, glm::vec3(1.3f, pedH + 0.005f, -1.5f), 0.40f, 0.01f, 1 /* matFrame */, 24);

    // Support cradle fork arms elevating the gimbal
    const glm::vec3 gyroPos(1.3f, pedH + 0.52f, -1.5f);
    const float forkSpan = 0.48f;
    addBeam(scene.triangles, glm::vec3(1.3f - forkSpan, pedH + 0.01f, -1.5f),
            glm::vec3(1.3f - forkSpan, gyroPos.y, -1.5f), 0.03f, 0.045f, 1 /* matFrame */);
    addBeam(scene.triangles, glm::vec3(1.3f + forkSpan, pedH + 0.01f, -1.5f),
            glm::vec3(1.3f + forkSpan, gyroPos.y, -1.5f), 0.03f, 0.045f, 1 /* matFrame */);
    addBeam(scene.triangles, glm::vec3(1.3f - forkSpan, pedH + 0.02f, -1.5f),
            glm::vec3(1.3f + forkSpan, pedH + 0.02f, -1.5f), 0.04f, 0.035f, 1 /* matFrame */);

    // Gold gimbal pivot bearing nodes
    addSphere(scene.triangles, glm::vec3(1.3f - forkSpan, gyroPos.y, -1.5f), 0.035f, 4 /* matGold */, 16, 16);
    addSphere(scene.triangles, glm::vec3(1.3f + forkSpan, gyroPos.y, -1.5f), 0.035f, 4 /* matGold */, 16, 16);
    recordRange("Pedestal_Gyroscope", tStart);

    // Architectural Freestanding Spotlight illuminating the Multi-Ring Gyroscope
    addStandingSpotlight(glm::vec3(1.60f, 0.0f, -2.10f), gyroPos,
                         glm::vec4(175.0f, 180.0f, 205.0f, 1.0f), "Gyroscope");

    // Right Midground: Pedestal & Emissive Area Light for Gold Accent Ring (X = -1.3, Z = 1.75)
    tStart = static_cast<uint32_t>(scene.triangles.size());
    // Glowing base plinth at floor level
    addCylinder(scene.triangles, glm::vec3(-1.3f, 0.015f, 1.75f), 0.44f, 0.03f, 12 /* matGoldLight */, 28);
    // Dark pedestal column
    addCylinder(scene.triangles, glm::vec3(-1.3f, pedH * 0.5f, 1.75f), 0.40f, pedH, 9, 24);
    // Luminaire fixture bezel atop platform
    addCylinder(scene.triangles, glm::vec3(-1.3f, pedH + 0.005f, 1.75f), 0.34f, 0.01f, 1 /* matFrame */, 24);
    // Emissive area light pad directly underneath the gold torus (facing +Y upwards)
    const float goldLightPadW = 0.22f; // 44cm x 44cm square area emitter
    addQuad(scene.triangles,
            glm::vec3(-1.3f - goldLightPadW, pedH + 0.011f, 1.75f - goldLightPadW),
            glm::vec3(-1.3f + goldLightPadW, pedH + 0.011f, 1.75f - goldLightPadW),
            glm::vec3(-1.3f + goldLightPadW, pedH + 0.011f, 1.75f + goldLightPadW),
            glm::vec3(-1.3f - goldLightPadW, pedH + 0.011f, 1.75f + goldLightPadW),
            glm::vec3(0.0f, 1.0f, 0.0f), 12 /* matGoldLight */);

    // Analytical LightGPU: Upward-facing warm amber/golden area light quad illuminating the gold torus
    LightGPU goldLight{};
    goldLight.position = glm::vec4(-1.3f - goldLightPadW, pedH + 0.011f, 1.75f - goldLightPadW, LIGHT_AREA_QUAD);
    goldLight.u = glm::vec4(2.0f * goldLightPadW, 0.0f, 0.0f, 0.0f);
    goldLight.v = glm::vec4(0.0f, 0.0f, 2.0f * goldLightPadW, 0.0f);
    goldLight.normal = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
    goldLight.emission = glm::vec4(28.0f, 22.0f, 12.0f, (2.0f * goldLightPadW) * (2.0f * goldLightPadW));
    scene.lights.push_back(goldLight);
    recordRange("Pedestal_GoldRing", tStart);

    // Architectural Freestanding Spotlight illuminating the Glass Crystal Sphere for Caustics
    addStandingSpotlight(glm::vec3(-1.60f, 0.0f, 2.80f), glm::vec3(0.0f, pedH + 0.42f, 3.5f),
                         glm::vec4(140.0f, 145.0f, 165.0f, 1.0f), "GlassSphere");

    // =========================================================================
    // 6. BLAS Geometry Prototypes & Hardware TLAS Instancing
    // =========================================================================

    auto recordBlasPrototype = [&](const std::string& name, uint32_t startTri) {
        uint32_t triCount = static_cast<uint32_t>(scene.triangles.size() - startTri);
        BlasGeometryRange range{};
        range.firstTriangle = startTri;
        range.triangleCount = triCount;
        range.numOpaqueTriangles = 0; // populated during Engine::partitionSceneGeometry
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

    // Record BLAS 0: Static Room Environment
    recordBlasPrototype("StaticRoom", 0);

    // Record BLAS 1: Dynamic Reflective Torus Prototype (Centered at origin in local space)
    uint32_t torusStart = static_cast<uint32_t>(scene.triangles.size());
    addTorus(scene.triangles, glm::vec3(0.0f, 0.0f, 0.0f), 0.40f, 0.13f, 3 /* matChrome */, 40, 20, glm::vec3(0.0f, 0.0f, 1.0f));
    recordBlasPrototype("Prototype_ReflectiveTorus", torusStart);

    // Record BLAS 2: Dynamic Gold Torus Prototype (Centered at origin in local space)
    uint32_t goldTorusStart = static_cast<uint32_t>(scene.triangles.size());
    addTorus(scene.triangles, glm::vec3(0.0f, 0.0f, 0.0f), 0.38f, 0.10f, 4 /* matGold */, 36, 18,
             glm::normalize(glm::vec3(0.35f, 0.88f, 0.25f)));
    recordBlasPrototype("Prototype_GoldTorus", goldTorusStart);

    // Record BLAS 3: Dynamic Multi-Ring Gyroscope - Outer Ring (Platinum/Titanium)
    uint32_t gyroOuterStart = static_cast<uint32_t>(scene.triangles.size());
    addTorus(scene.triangles, glm::vec3(0.0f), 0.44f, 0.024f, 15 /* matPlatinum */, 36, 18, glm::vec3(0.0f, 0.0f, 1.0f));
    // Outer ring bearing pins along X axis
    addBeam(scene.triangles, glm::vec3(-0.48f, 0.0f, 0.0f), glm::vec3(-0.42f, 0.0f, 0.0f), 0.016f, 0.016f, 4 /* matGold */);
    addBeam(scene.triangles, glm::vec3( 0.42f, 0.0f, 0.0f), glm::vec3( 0.48f, 0.0f, 0.0f), 0.016f, 0.016f, 4 /* matGold */);
    recordBlasPrototype("Prototype_GyroOuter", gyroOuterStart);

    // Record BLAS 4: Dynamic Multi-Ring Gyroscope - Middle Ring (Rose Gold / Copper)
    uint32_t gyroMidStart = static_cast<uint32_t>(scene.triangles.size());
    addTorus(scene.triangles, glm::vec3(0.0f), 0.34f, 0.020f, 16 /* matCopper */, 32, 16, glm::vec3(1.0f, 0.0f, 0.0f));
    // Middle ring bearing pins along Y axis
    addBeam(scene.triangles, glm::vec3(0.0f, -0.38f, 0.0f), glm::vec3(0.0f, -0.32f, 0.0f), 0.014f, 0.014f, 4 /* matGold */);
    addBeam(scene.triangles, glm::vec3(0.0f,  0.32f, 0.0f), glm::vec3(0.0f,  0.38f, 0.0f), 0.014f, 0.014f, 4 /* matGold */);
    recordBlasPrototype("Prototype_GyroMiddle", gyroMidStart);

    // Record BLAS 5: Dynamic Multi-Ring Gyroscope - Inner Ring & Core (Dark Chrome & Gold)
    uint32_t gyroInnerStart = static_cast<uint32_t>(scene.triangles.size());
    addTorus(scene.triangles, glm::vec3(0.0f), 0.24f, 0.016f, 17 /* matDarkChrome */, 28, 14, glm::vec3(0.0f, 1.0f, 0.0f));
    // Inner ring bearing pins along Z axis
    addBeam(scene.triangles, glm::vec3(0.0f, 0.0f, -0.28f), glm::vec3(0.0f, 0.0f, -0.22f), 0.012f, 0.012f, 4 /* matGold */);
    addBeam(scene.triangles, glm::vec3(0.0f, 0.0f,  0.22f), glm::vec3(0.0f, 0.0f,  0.28f), 0.012f, 0.012f, 4 /* matGold */);
    // Central kinetic jewel core
    addSphere(scene.triangles, glm::vec3(0.0f), 0.09f, 4 /* matGold */, 20, 20);
    recordBlasPrototype("Prototype_GyroInner", gyroInnerStart);

    // Hardware TLAS Instances
    // Instance 0: Static Room
    SceneInstance instStatic{};
    instStatic.blasIndex = 0;
    instStatic.transform = glm::mat4(1.0f);
    instStatic.customIndex = 0;
    scene.instances.push_back(instStatic);

    InstanceGPU instStaticGPU{};
    instStaticGPU.firstTriangle = scene.blasRanges[0].firstTriangle;
    instStaticGPU.numOpaqueTriangles = scene.blasRanges[0].numOpaqueTriangles;
    instStaticGPU.materialOffset = 0;
    instStaticGPU.flags = 0;
    scene.instanceData.push_back(instStaticGPU);

    // Instance 1: Reflective Torus atop Pedestal (Z = -3.5) with 20-degree tilt
    glm::vec3 torusPos(0.0f, pedH + 0.45f, -3.5f);
    glm::mat4 baseTilt = glm::rotate(glm::mat4(1.0f), glm::radians(20.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    glm::mat4 initialTransform = glm::translate(glm::mat4(1.0f), torusPos) * baseTilt;

    SceneInstance instTorus{};
    instTorus.blasIndex = 1;
    instTorus.transform = initialTransform;
    instTorus.customIndex = 1;
    scene.instances.push_back(instTorus);

    InstanceGPU instTorusGPU{};
    instTorusGPU.firstTriangle = scene.blasRanges[1].firstTriangle;
    instTorusGPU.numOpaqueTriangles = scene.blasRanges[1].numOpaqueTriangles;
    instTorusGPU.materialOffset = 0;
    instTorusGPU.flags = 0;
    scene.instanceData.push_back(instTorusGPU);

    // Register Dynamic Kinematic Animator for Instance 1
    AnimatedInstance animTorus{};
    animTorus.instanceIndex = 1;
    animTorus.basePosition = torusPos;
    animTorus.rotationAxis = glm::vec3(0.0f, 1.0f, 0.0f); // Spin about vertical Y axis
    animTorus.rotationSpeed = 0.8f;                         // ~45 deg/second (smooth, majestic spin)
    animTorus.baseTransform = baseTilt;
    scene.animatedInstances.push_back(animTorus);

    // Instance 2: Gold Torus atop Pedestal (X = -1.3, Z = 1.75)
    glm::vec3 goldTorusPos(-1.3f, pedH + 0.42f, 1.75f);
    glm::mat4 goldInitialTransform = glm::translate(glm::mat4(1.0f), goldTorusPos);

    SceneInstance instGoldTorus{};
    instGoldTorus.blasIndex = 2;
    instGoldTorus.transform = goldInitialTransform;
    instGoldTorus.customIndex = 2;
    scene.instances.push_back(instGoldTorus);

    InstanceGPU instGoldTorusGPU{};
    instGoldTorusGPU.firstTriangle = scene.blasRanges[2].firstTriangle;
    instGoldTorusGPU.numOpaqueTriangles = scene.blasRanges[2].numOpaqueTriangles;
    instGoldTorusGPU.materialOffset = 0;
    instGoldTorusGPU.flags = 0;
    scene.instanceData.push_back(instGoldTorusGPU);

    // Register Dynamic Kinematic Animator for Instance 2
    AnimatedInstance animGoldTorus{};
    animGoldTorus.instanceIndex = 2;
    animGoldTorus.basePosition = goldTorusPos;
    animGoldTorus.rotationAxis = glm::vec3(0.0f, 1.0f, 0.0f); // Spin about vertical Y axis
    animGoldTorus.rotationSpeed = -0.6f;                       // Counter-rotation to chrome torus (~-34.4 deg/s)
    animGoldTorus.baseTransform = glm::mat4(1.0f);
    scene.animatedInstances.push_back(animGoldTorus);

    // Instance 3: Gyroscope Outer Ring (X = 1.3, Z = -1.5)
    SceneInstance instGyroOuter{};
    instGyroOuter.blasIndex = 3;
    instGyroOuter.transform = glm::translate(glm::mat4(1.0f), gyroPos);
    instGyroOuter.customIndex = 3;
    scene.instances.push_back(instGyroOuter);

    InstanceGPU instGyroOuterGPU{};
    instGyroOuterGPU.firstTriangle = scene.blasRanges[3].firstTriangle;
    instGyroOuterGPU.numOpaqueTriangles = scene.blasRanges[3].numOpaqueTriangles;
    instGyroOuterGPU.materialOffset = 0;
    instGyroOuterGPU.flags = 0;
    scene.instanceData.push_back(instGyroOuterGPU);

    AnimatedInstance animGyroOuter{};
    animGyroOuter.instanceIndex = 3;
    animGyroOuter.basePosition = gyroPos;
    animGyroOuter.rotationAxis = glm::vec3(0.0f, 1.0f, 0.0f); // Smooth yaw rotation
    animGyroOuter.rotationSpeed = 0.70f;                      // ~40.1 deg/s
    animGyroOuter.baseTransform = glm::mat4(1.0f);
    scene.animatedInstances.push_back(animGyroOuter);

    // Instance 4: Gyroscope Middle Ring (X = 1.3, Z = -1.5)
    glm::mat4 midInitialTilt = glm::rotate(glm::mat4(1.0f), glm::radians(35.0f), glm::vec3(0.0f, 0.0f, 1.0f));
    SceneInstance instGyroMid{};
    instGyroMid.blasIndex = 4;
    instGyroMid.transform = glm::translate(glm::mat4(1.0f), gyroPos) * midInitialTilt;
    instGyroMid.customIndex = 4;
    scene.instances.push_back(instGyroMid);

    InstanceGPU instGyroMidGPU{};
    instGyroMidGPU.firstTriangle = scene.blasRanges[4].firstTriangle;
    instGyroMidGPU.numOpaqueTriangles = scene.blasRanges[4].numOpaqueTriangles;
    instGyroMidGPU.materialOffset = 0;
    instGyroMidGPU.flags = 0;
    scene.instanceData.push_back(instGyroMidGPU);

    AnimatedInstance animGyroMid{};
    animGyroMid.instanceIndex = 4;
    animGyroMid.basePosition = gyroPos;
    animGyroMid.rotationAxis = glm::vec3(1.0f, 0.0f, 0.0f); // Pitch rotation
    animGyroMid.rotationSpeed = -1.15f;                     // ~-65.9 deg/s counter-rotation
    animGyroMid.baseTransform = midInitialTilt;
    scene.animatedInstances.push_back(animGyroMid);

    // Instance 5: Gyroscope Inner Ring & Core (X = 1.3, Z = -1.5)
    glm::vec3 innerAxis = glm::normalize(glm::vec3(0.0f, 0.7071f, 0.7071f));
    glm::mat4 innerInitialTilt = glm::rotate(glm::mat4(1.0f), glm::radians(-25.0f), glm::vec3(1.0f, 0.0f, 0.0f));
    SceneInstance instGyroInner{};
    instGyroInner.blasIndex = 5;
    instGyroInner.transform = glm::translate(glm::mat4(1.0f), gyroPos) * innerInitialTilt;
    instGyroInner.customIndex = 5;
    scene.instances.push_back(instGyroInner);

    InstanceGPU instGyroInnerGPU{};
    instGyroInnerGPU.firstTriangle = scene.blasRanges[5].firstTriangle;
    instGyroInnerGPU.numOpaqueTriangles = scene.blasRanges[5].numOpaqueTriangles;
    instGyroInnerGPU.materialOffset = 0;
    instGyroInnerGPU.flags = 0;
    scene.instanceData.push_back(instGyroInnerGPU);

    AnimatedInstance animGyroInner{};
    animGyroInner.instanceIndex = 5;
    animGyroInner.basePosition = gyroPos;
    animGyroInner.rotationAxis = innerAxis;                 // Compound oblique roll/spin
    animGyroInner.rotationSpeed = 1.75f;                    // ~100.3 deg/s
    animGyroInner.baseTransform = innerInitialTilt;
    scene.animatedInstances.push_back(animGyroInner);

    // 7. Camera Setup
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
    scene.dielectricBoundsMin = glm::vec3(-0.5f, 0.2f, 3.0f);
    scene.dielectricBoundsMax = glm::vec3(0.5f, 1.1f, 4.0f);

    return scene;
}

// ============================================================================
// MINESHAFT COASTER MODELS & PROCEDURAL SCENE
// ============================================================================

static void addOrientedCylinder(std::vector<TriangleGPU>& triangles,
                                glm::vec3 pA, glm::vec3 pB,
                                float radiusA, float radiusB,
                                uint32_t matId, int segments = 12, bool capEnds = true) {
    glm::vec3 axis = pB - pA;
    float len = glm::length(axis);
    if (len < 1e-5f) return;
    glm::vec3 dir = axis / len;

    glm::vec3 up = std::abs(dir.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 u = glm::normalize(glm::cross(dir, up));
    glm::vec3 v = glm::normalize(glm::cross(u, dir));
    const float PI = 3.14159265358979323846f;

    for (int i = 0; i < segments; ++i) {
        float a0 = (2.0f * PI * static_cast<float>(i)) / static_cast<float>(segments);
        float a1 = (2.0f * PI * static_cast<float>(i + 1)) / static_cast<float>(segments);

        glm::vec3 rDir0 = u * std::cos(a0) + v * std::sin(a0);
        glm::vec3 rDir1 = u * std::cos(a1) + v * std::sin(a1);
        glm::vec3 nSide = glm::normalize(rDir0 + rDir1);

        glm::vec3 p0 = pA + rDir0 * radiusA;
        glm::vec3 p1 = pA + rDir1 * radiusA;
        glm::vec3 p2 = pB + rDir1 * radiusB;
        glm::vec3 p3 = pB + rDir0 * radiusB;

        addQuad(triangles, p0, p1, p2, p3, nSide, matId);

        if (capEnds) {
            if (radiusB > 1e-4f) {
                TriangleGPU topTri{};
                topTri.v0.position = glm::vec4(pB, 0.5f);
                topTri.v0.normal = glm::vec4(dir, 0.5f);
                topTri.v0.tangent = glm::vec4(u, 1.0f);
                topTri.v1.position = glm::vec4(p3, 0.0f);
                topTri.v1.normal = glm::vec4(dir, 0.0f);
                topTri.v1.tangent = glm::vec4(u, 1.0f);
                topTri.v2.position = glm::vec4(p2, 1.0f);
                topTri.v2.normal = glm::vec4(dir, 1.0f);
                topTri.v2.tangent = glm::vec4(u, 1.0f);
                topTri.materialId = matId;
                triangles.push_back(topTri);
            }
            if (radiusA > 1e-4f) {
                TriangleGPU botTri{};
                botTri.v0.position = glm::vec4(pA, 0.5f);
                botTri.v0.normal = glm::vec4(-dir, 0.5f);
                botTri.v0.tangent = glm::vec4(u, 1.0f);
                botTri.v1.position = glm::vec4(p0, 0.0f);
                botTri.v1.normal = glm::vec4(-dir, 0.0f);
                botTri.v1.tangent = glm::vec4(u, 1.0f);
                botTri.v2.position = glm::vec4(p1, 1.0f);
                botTri.v2.normal = glm::vec4(-dir, 1.0f);
                botTri.v2.tangent = glm::vec4(u, 1.0f);
                botTri.materialId = matId;
                triangles.push_back(botTri);
            }
        }
    }
}

static void addHexCrystalPrism(std::vector<TriangleGPU>& triangles,
                               glm::vec3 basePos, glm::vec3 axisDir,
                               float radius, float shaftLen, float tipLen,
                               uint32_t outerMatId, int coreMatId = -1) {
    glm::vec3 dir = glm::normalize(axisDir);
    glm::vec3 up = std::abs(dir.y) < 0.99f ? glm::vec3(0.0f, 1.0f, 0.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 u = glm::normalize(glm::cross(dir, up));
    glm::vec3 v = glm::normalize(glm::cross(u, dir));
    const float PI = 3.14159265358979323846f;

    glm::vec3 shoulderCenter = basePos + dir * shaftLen;
    glm::vec3 apexPos = shoulderCenter + dir * tipLen;

    for (int i = 0; i < 6; ++i) {
        float a0 = (2.0f * PI * static_cast<float>(i)) / 6.0f;
        float a1 = (2.0f * PI * static_cast<float>(i + 1)) / 6.0f;

        glm::vec3 r0 = u * std::cos(a0) + v * std::sin(a0);
        glm::vec3 r1 = u * std::cos(a1) + v * std::sin(a1);
        glm::vec3 nSide = glm::normalize(r0 + r1);

        glm::vec3 b0 = basePos + r0 * radius;
        glm::vec3 b1 = basePos + r1 * radius;
        glm::vec3 s1 = shoulderCenter + r1 * radius * 0.92f;
        glm::vec3 s0 = shoulderCenter + r0 * radius * 0.92f;

        // Flat faceted hexagonal prism side
        addQuad(triangles, b0, b1, s1, s0, nSide, outerMatId);

        // Pyramidal termination facet
        glm::vec3 tipNorm = glm::normalize(glm::cross(s1 - s0, apexPos - s0));
        TriangleGPU tipTri{};
        tipTri.v0.position = glm::vec4(s0, 0.0f);
        tipTri.v0.normal = glm::vec4(tipNorm, 0.0f);
        tipTri.v0.tangent = glm::vec4(glm::normalize(s1 - s0), 1.0f);
        tipTri.v1.position = glm::vec4(s1, 1.0f);
        tipTri.v1.normal = glm::vec4(tipNorm, 0.0f);
        tipTri.v1.tangent = glm::vec4(glm::normalize(s1 - s0), 1.0f);
        tipTri.v2.position = glm::vec4(apexPos, 0.5f);
        tipTri.v2.normal = glm::vec4(tipNorm, 1.0f);
        tipTri.v2.tangent = glm::vec4(glm::normalize(s1 - s0), 1.0f);
        tipTri.materialId = outerMatId;
        triangles.push_back(tipTri);
    }

    // Optional inner glowing emissive filament core
    if (coreMatId >= 0) {
        addOrientedCylinder(triangles,
                            basePos + dir * (shaftLen * 0.12f),
                            basePos + dir * (shaftLen * 0.85f),
                            radius * 0.32f, radius * 0.15f,
                            static_cast<uint32_t>(coreMatId), 6, true);
    }
}

struct MineshaftMaterials {
    uint32_t rock = 0;
    uint32_t wetFloor = 1;
    uint32_t timber = 2;
    uint32_t railSteel = 3;
    uint32_t castIron = 4;
    uint32_t brass = 5;
    uint32_t reflector = 6;
    uint32_t glass = 7;
    uint32_t headlampEmissive = 8;
    uint32_t lanternFlame = 9;
    uint32_t crystalViolet = 10;
    uint32_t crystalCyan = 11;
    uint32_t crystalCoreViolet = 12;
    uint32_t crystalCoreCyan = 13;
    uint32_t goldOre = 14;
    uint32_t waterSurface = 15;
};

static MineshaftMaterials initMineshaftMaterials(SceneData& scene) {
    MineshaftMaterials m{};
    scene.materials.clear();

    // 0: Cave Rock Wall (Procedural 3D terrain bump + Voronoi crack detail)
    MaterialGPU matRock{};
    matRock.albedo = glm::vec4(0.24f, 0.21f, 0.19f, 1.0f);
    matRock.roughness = 0.82f;
    matRock.metallic = 0.04f;
    matRock.ior = 1.5f;
    matRock.type = MATERIAL_DIFFUSE | (1u << 9); // MATERIAL_FLAG_PROCEDURAL_TERRAIN
    scene.materials.push_back(matRock);

    // 1: Wet Cave Floor / Ballast (Procedural wet puddles)
    MaterialGPU matWetFloor{};
    matWetFloor.albedo = glm::vec4(0.18f, 0.17f, 0.16f, 1.0f);
    matWetFloor.roughness = 0.45f;
    matWetFloor.metallic = 0.05f;
    matWetFloor.ior = 1.5f;
    matWetFloor.type = MATERIAL_DIFFUSE | (1u << 12); // MATERIAL_FLAG_PROCEDURAL_PUDDLE
    scene.materials.push_back(matWetFloor);

    // 2: Rough-Hewn Oak Timber
    MaterialGPU matTimber{};
    matTimber.albedo = glm::vec4(0.32f, 0.19f, 0.10f, 1.0f);
    matTimber.roughness = 0.76f;
    matTimber.metallic = 0.02f;
    matTimber.ior = 1.5f;
    matTimber.type = MATERIAL_DIFFUSE;
    scene.materials.push_back(matTimber);

    // 3: Polished Steel Rail Head
    MaterialGPU matRailSteel{};
    matRailSteel.albedo = glm::vec4(0.82f, 0.85f, 0.89f, 1.0f);
    matRailSteel.roughness = 0.12f;
    matRailSteel.metallic = 0.96f;
    matRailSteel.ior = 2.5f;
    matRailSteel.type = MATERIAL_METALLIC;
    scene.materials.push_back(matRailSteel);

    // 4: Weathered Cast Iron (Minecart tub, wheels, tie-plates, lantern cage)
    MaterialGPU matCastIron{};
    matCastIron.albedo = glm::vec4(0.24f, 0.26f, 0.29f, 1.0f);
    matCastIron.roughness = 0.34f;
    matCastIron.metallic = 0.88f;
    matCastIron.ior = 2.4f;
    matCastIron.type = MATERIAL_METALLIC;
    scene.materials.push_back(matCastIron);

    // 5: Tarnished Brass / Copper (Headlamp housing, lantern reservoir, rivets)
    MaterialGPU matBrass{};
    matBrass.albedo = glm::vec4(0.90f, 0.66f, 0.26f, 1.0f);
    matBrass.roughness = 0.18f;
    matBrass.metallic = 0.94f;
    matBrass.ior = 1.8f;
    matBrass.type = MATERIAL_METALLIC;
    scene.materials.push_back(matBrass);

    // 6: Parabolic Chrome Reflector
    MaterialGPU matReflector{};
    matReflector.albedo = glm::vec4(0.96f, 0.97f, 0.99f, 1.0f);
    matReflector.roughness = 0.03f;
    matReflector.metallic = 1.0f;
    matReflector.ior = 2.5f;
    matReflector.type = MATERIAL_METALLIC;
    scene.materials.push_back(matReflector);

    // 7: Smooth Clear Dielectric Glass (Lantern globe & headlamp lens)
    MaterialGPU matGlass{};
    matGlass.albedo = glm::vec4(0.98f, 0.99f, 1.0f, 1.0f);
    matGlass.roughness = 0.01f;
    matGlass.metallic = 0.0f;
    matGlass.ior = 1.52f;
    matGlass.transmission = 1.0f;
    matGlass.thickness = 0.02f;
    matGlass.attenuationColor = glm::vec4(0.95f, 0.98f, 1.0f, 1.5f);
    matGlass.type = MATERIAL_DIELECTRIC;
    scene.materials.push_back(matGlass);

    // 8: Carbide Headlamp Filament Emitter
    MaterialGPU matHeadlampEmissive{};
    matHeadlampEmissive.albedo = glm::vec4(1.0f, 0.95f, 0.85f, 1.0f);
    matHeadlampEmissive.emissive = glm::vec4(42.0f, 34.0f, 20.0f, 1.0f);
    matHeadlampEmissive.roughness = 0.1f;
    matHeadlampEmissive.type = MATERIAL_EMISSIVE;
    scene.materials.push_back(matHeadlampEmissive);

    // 9: Lantern Flame Mantle Emitter
    MaterialGPU matLanternFlame{};
    matLanternFlame.albedo = glm::vec4(1.0f, 0.7f, 0.3f, 1.0f);
    matLanternFlame.emissive = glm::vec4(30.0f, 14.0f, 3.2f, 1.0f);
    matLanternFlame.roughness = 0.1f;
    matLanternFlame.type = MATERIAL_EMISSIVE;
    scene.materials.push_back(matLanternFlame);

    // 10: Refractive Amethyst / Violet Quartz Crystal
    MaterialGPU matCrystalViolet{};
    matCrystalViolet.albedo = glm::vec4(0.92f, 0.72f, 1.0f, 1.0f);
    matCrystalViolet.roughness = 0.02f;
    matCrystalViolet.metallic = 0.0f;
    matCrystalViolet.ior = 1.55f;
    matCrystalViolet.transmission = 1.0f;
    matCrystalViolet.thickness = 0.18f;
    matCrystalViolet.dispersion = 0.045f;
    matCrystalViolet.attenuationColor = glm::vec4(0.68f, 0.22f, 0.96f, 0.45f);
    matCrystalViolet.type = MATERIAL_DIELECTRIC;
    scene.materials.push_back(matCrystalViolet);

    // 11: Refractive Cyan Fluorite Crystal
    MaterialGPU matCrystalCyan{};
    matCrystalCyan.albedo = glm::vec4(0.72f, 0.96f, 1.0f, 1.0f);
    matCrystalCyan.roughness = 0.02f;
    matCrystalCyan.metallic = 0.0f;
    matCrystalCyan.ior = 1.54f;
    matCrystalCyan.transmission = 1.0f;
    matCrystalCyan.thickness = 0.18f;
    matCrystalCyan.dispersion = 0.045f;
    matCrystalCyan.attenuationColor = glm::vec4(0.18f, 0.85f, 0.96f, 0.45f);
    matCrystalCyan.type = MATERIAL_DIELECTRIC;
    scene.materials.push_back(matCrystalCyan);

    // 12: Glowing Violet Crystal Core
    MaterialGPU matCrystalCoreViolet{};
    matCrystalCoreViolet.albedo = glm::vec4(0.9f, 0.5f, 1.0f, 1.0f);
    matCrystalCoreViolet.emissive = glm::vec4(11.0f, 3.0f, 18.0f, 1.0f);
    matCrystalCoreViolet.roughness = 0.2f;
    matCrystalCoreViolet.type = MATERIAL_EMISSIVE;
    scene.materials.push_back(matCrystalCoreViolet);

    // 13: Glowing Cyan Crystal Core
    MaterialGPU matCrystalCoreCyan{};
    matCrystalCoreCyan.albedo = glm::vec4(0.5f, 0.95f, 1.0f, 1.0f);
    matCrystalCoreCyan.emissive = glm::vec4(2.5f, 14.0f, 18.0f, 1.0f);
    matCrystalCoreCyan.roughness = 0.2f;
    matCrystalCoreCyan.type = MATERIAL_EMISSIVE;
    scene.materials.push_back(matCrystalCoreCyan);

    // 14: Glinting Gold Ore Vein
    MaterialGPU matGoldOre{};
    matGoldOre.albedo = glm::vec4(1.0f, 0.78f, 0.28f, 1.0f);
    matGoldOre.roughness = 0.15f;
    matGoldOre.metallic = 0.96f;
    matGoldOre.ior = 1.5f;
    matGoldOre.type = MATERIAL_METALLIC;
    scene.materials.push_back(matGoldOre);

    // 15: Subterranean Flooded Sluiceway & Cavern Lake Water Surface
    MaterialGPU matWaterSurface{};
    matWaterSurface.albedo = glm::vec4(0.05f, 0.14f, 0.17f, 1.0f);
    matWaterSurface.roughness = 0.03f;
    matWaterSurface.metallic = 0.38f;
    matWaterSurface.ior = 1.333f;
    matWaterSurface.type = MATERIAL_DIFFUSE | (1u << 12); // MATERIAL_FLAG_PROCEDURAL_PUDDLE
    scene.materials.push_back(matWaterSurface);

    return m;
}

// Model 1: Riveted Ore Minecart with Brass & Glass Carbide Headlamp
static void buildMinecartModel(std::vector<TriangleGPU>& tris, const glm::vec3& origin, const MineshaftMaterials& m) {
    // 1. Timber & Iron Chassis Frame (sits on rails at Y = +0.16m wheel radius)
    const float wheelR = 0.16f;
    const float axleY = wheelR;
    const float halfGauge = 0.46f;
    const float axleZ[2] = { -0.42f, 0.42f };

    // Longitudinal oak chassis beams
    addBox(tris, origin + glm::vec3(-0.34f, axleY + 0.07f, 0.0f), glm::vec3(0.09f, 0.09f, 1.52f), 0.0f, m.timber);
    addBox(tris, origin + glm::vec3( 0.34f, axleY + 0.07f, 0.0f), glm::vec3(0.09f, 0.09f, 1.52f), 0.0f, m.timber);
    // Front & rear heavy iron buffer beams + couplers
    addBox(tris, origin + glm::vec3(0.0f, axleY + 0.07f,  0.76f), glm::vec3(0.86f, 0.10f, 0.09f), 0.0f, m.castIron);
    addBox(tris, origin + glm::vec3(0.0f, axleY + 0.07f, -0.76f), glm::vec3(0.86f, 0.10f, 0.09f), 0.0f, m.castIron);
    addTorus(tris, origin + glm::vec3(0.0f, axleY + 0.07f,  0.84f), 0.05f, 0.015f, m.brass, 16, 10, glm::vec3(0, 1, 0));
    addTorus(tris, origin + glm::vec3(0.0f, axleY + 0.07f, -0.84f), 0.05f, 0.015f, m.brass, 16, 10, glm::vec3(0, 1, 0));

    // 2. Four Flanged Cast-Iron Wheels + Steel Axles
    for (int a = 0; a < 2; ++a) {
        float z = axleZ[a];
        // Solid steel axle rod
        addOrientedCylinder(tris,
                            origin + glm::vec3(-halfGauge - 0.08f, axleY, z),
                            origin + glm::vec3( halfGauge + 0.08f, axleY, z),
                            0.032f, 0.032f, m.railSteel, 12, true);
        // Axle journal bearing boxes
        addBox(tris, origin + glm::vec3(-0.34f, axleY + 0.02f, z), glm::vec3(0.12f, 0.10f, 0.12f), 0.0f, m.castIron);
        addBox(tris, origin + glm::vec3( 0.34f, axleY + 0.02f, z), glm::vec3(0.12f, 0.10f, 0.12f), 0.0f, m.castIron);

        for (int s = -1; s <= 1; s += 2) {
            float x = static_cast<float>(s) * halfGauge;
            float innerX = x - static_cast<float>(s) * 0.035f;
            float outerX = x + static_cast<float>(s) * 0.035f;

            // Polished steel outer running tread
            addOrientedCylinder(tris,
                                origin + glm::vec3(innerX, axleY, z),
                                origin + glm::vec3(outerX, axleY, z),
                                wheelR, wheelR, m.railSteel, 20, false);
            // Inner wheel flange (larger diameter on inside edge)
            addOrientedCylinder(tris,
                                origin + glm::vec3(innerX - static_cast<float>(s) * 0.018f, axleY, z),
                                origin + glm::vec3(innerX, axleY, z),
                                wheelR + 0.038f, wheelR + 0.030f, m.railSteel, 20, true);
            // Cast-iron recessed wheel web disc
            addOrientedCylinder(tris,
                                origin + glm::vec3(innerX, axleY, z),
                                origin + glm::vec3(outerX, axleY, z),
                                wheelR * 0.82f, wheelR * 0.82f, m.castIron, 16, true);
            // 6 Radial Wheel Spokes + Brass Hubcap
            for (int sp = 0; sp < 6; ++sp) {
                float ang = (3.14159265f * static_cast<float>(sp)) / 3.0f;
                glm::vec3 spokeDir(0.0f, std::sin(ang), std::cos(ang));
                addBeam(tris,
                        origin + glm::vec3(outerX, axleY, z) - spokeDir * (wheelR * 0.78f),
                        origin + glm::vec3(outerX, axleY, z) + spokeDir * (wheelR * 0.78f),
                        0.024f, 0.024f, m.brass);
            }
            addOrientedCylinder(tris,
                                origin + glm::vec3(outerX, axleY, z),
                                origin + glm::vec3(outerX + static_cast<float>(s) * 0.03f, axleY, z),
                                0.052f, 0.045f, m.brass, 12, true);
        }
    }

    // 3. Riveted Steel Hopper Tub (Flared Trapezoidal Body)
    const float tubBottomY = axleY + 0.12f;
    const float tubHeight = 0.68f;
    const float tubTopY = tubBottomY + tubHeight;
    addFrustum(tris, origin + glm::vec3(0.0f, tubBottomY + tubHeight * 0.5f, 0.0f),
               glm::vec2(0.78f, 1.28f), glm::vec2(1.02f, 1.54f), tubHeight, m.castIron);

    // Rolled tubular steel top rim around the hopper lip
    const float rimHalfW = 0.51f;
    const float rimHalfL = 0.77f;
    addBeam(tris, origin + glm::vec3(-rimHalfW, tubTopY, -rimHalfL), origin + glm::vec3(-rimHalfW, tubTopY,  rimHalfL), 0.055f, 0.055f, m.railSteel);
    addBeam(tris, origin + glm::vec3( rimHalfW, tubTopY, -rimHalfL), origin + glm::vec3( rimHalfW, tubTopY,  rimHalfL), 0.055f, 0.055f, m.railSteel);
    addBeam(tris, origin + glm::vec3(-rimHalfW, tubTopY,  rimHalfL), origin + glm::vec3( rimHalfW, tubTopY,  rimHalfL), 0.055f, 0.055f, m.railSteel);
    addBeam(tris, origin + glm::vec3(-rimHalfW, tubTopY, -rimHalfL), origin + glm::vec3( rimHalfW, tubTopY, -rimHalfL), 0.055f, 0.055f, m.railSteel);

    // Side vertical reinforcing straps & brass rivets
    for (int side = -1; side <= 1; side += 2) {
        float sx = static_cast<float>(side);
        for (int r = -1; r <= 1; ++r) {
            float rz = static_cast<float>(r) * 0.42f;
            glm::vec3 pBot = origin + glm::vec3(sx * 0.39f, tubBottomY + 0.03f, rz);
            glm::vec3 pTop = origin + glm::vec3(sx * 0.51f, tubTopY - 0.02f, rz);
            addBeam(tris, pBot, pTop, 0.045f, 0.035f, m.railSteel);
            for (int rv = 0; rv < 4; ++rv) {
                float t = (static_cast<float>(rv) + 0.5f) / 4.0f;
                glm::vec3 rPos = glm::mix(pBot, pTop, t) + glm::vec3(sx * 0.022f, 0.0f, 0.0f);
                addSphere(tris, rPos, 0.015f, m.brass, 8, 8);
            }
        }
    }

    // 4. Mound of Mined Rock, Raw Gold Nuggets & Glowing Crystal Shards Inside Tub
    addFrustum(tris, origin + glm::vec3(0.0f, tubTopY - 0.04f, 0.0f),
               glm::vec2(0.92f, 1.42f), glm::vec2(0.56f, 0.96f), 0.22f, m.rock);
    const glm::vec3 oreOffsets[7] = {
        glm::vec3(-0.22f, tubTopY + 0.06f,  0.30f),
        glm::vec3( 0.18f, tubTopY + 0.08f,  0.15f),
        glm::vec3(-0.10f, tubTopY + 0.10f, -0.20f),
        glm::vec3( 0.25f, tubTopY + 0.05f, -0.35f),
        glm::vec3( 0.02f, tubTopY + 0.11f,  0.42f),
        glm::vec3(-0.28f, tubTopY + 0.05f, -0.05f),
        glm::vec3( 0.14f, tubTopY + 0.09f, -0.05f)
    };
    for (int i = 0; i < 7; ++i) {
        uint32_t oreMat = (i % 2 == 0) ? m.goldOre : m.rock;
        addBox(tris, origin + oreOffsets[i], glm::vec3(0.18f, 0.14f, 0.20f), static_cast<float>(i * 37), oreMat);
    }
    // Crystal shards jutting out of the ore load
    addHexCrystalPrism(tris, origin + glm::vec3(-0.12f, tubTopY + 0.04f, 0.22f),
                       glm::normalize(glm::vec3(-0.25f, 0.92f, 0.20f)), 0.055f, 0.22f, 0.09f, m.crystalViolet, static_cast<int>(m.crystalCoreViolet));
    addHexCrystalPrism(tris, origin + glm::vec3(0.16f, tubTopY + 0.04f, -0.12f),
                       glm::normalize(glm::vec3(0.30f, 0.88f, -0.15f)), 0.050f, 0.20f, 0.08f, m.crystalCyan, static_cast<int>(m.crystalCoreCyan));

    // 5. Front Brass & Glass Carbide Headlamp Assembly (at Z = +0.84m)
    glm::vec3 lampCenter = origin + glm::vec3(0.0f, tubTopY - 0.12f, 0.85f);
    // Iron mounting bracket
    addBeam(tris, origin + glm::vec3(0.0f, tubTopY - 0.26f, 0.72f), lampCenter + glm::vec3(0.0f, -0.10f, 0.0f), 0.06f, 0.04f, m.castIron);
    // Brass cylindrical lamp body (open front) + rear cap + top carbide water tank
    addOrientedCylinder(tris, lampCenter + glm::vec3(0.0f, 0.0f, -0.08f), lampCenter + glm::vec3(0.0f, 0.0f, 0.08f), 0.14f, 0.16f, m.brass, 20, false);
    addOrientedCylinder(tris, lampCenter + glm::vec3(0.0f, 0.0f, -0.09f), lampCenter + glm::vec3(0.0f, 0.0f, -0.08f), 0.13f, 0.14f, m.brass, 20, true);
    addCylinder(tris, lampCenter + glm::vec3(0.0f, 0.17f, -0.01f), 0.075f, 0.10f, m.brass, 16);
    // Mirror-polished parabolic reflector dish lining the entire interior of the housing
    addOrientedCylinder(tris, lampCenter + glm::vec3(0.0f, 0.0f, -0.05f), lampCenter + glm::vec3(0.0f, 0.0f, 0.078f), 0.045f, 0.154f, m.reflector, 20, false);
    addOrientedCylinder(tris, lampCenter + glm::vec3(0.0f, 0.0f, -0.055f), lampCenter + glm::vec3(0.0f, 0.0f, -0.048f), 0.01f, 0.048f, m.reflector, 20, true);
    // Glowing carbide filament bulb (prominent warm emitter at focal point of parabolic dish)
    addSphere(tris, lampCenter + glm::vec3(0.0f, 0.0f, 0.032f), 0.064f, m.headlampEmissive, 14, 14);
    // Smooth dielectric front glass lens + brass bezel ring
    addOrientedCylinder(tris, lampCenter + glm::vec3(0.0f, 0.0f, 0.080f), lampCenter + glm::vec3(0.0f, 0.0f, 0.094f), 0.152f, 0.148f, m.glass, 20, true);
    addTorus(tris, lampCenter + glm::vec3(0.0f, 0.0f, 0.085f), 0.158f, 0.020f, m.brass, 20, 12, glm::vec3(0, 0, 1));
}

// Model 2: Hanging Glass & Brass Mine Lantern (pivot at top hook, extends down along -Y)
static void buildMineLanternModel(std::vector<TriangleGPU>& tris, const glm::vec3& pivotOrigin, const MineshaftMaterials& m) {
    // Suspension hook & chain link at pivotOrigin (Y = 0 to -0.14)
    addTorus(tris, pivotOrigin + glm::vec3(0.0f, -0.03f, 0.0f), 0.032f, 0.009f, m.castIron, 14, 8, glm::vec3(0, 0, 1));
    addTorus(tris, pivotOrigin + glm::vec3(0.0f, -0.08f, 0.0f), 0.032f, 0.009f, m.castIron, 14, 8, glm::vec3(1, 0, 0));

    // Lantern body center at Y = -0.34m below hook
    glm::vec3 c = pivotOrigin + glm::vec3(0.0f, -0.34f, 0.0f);

    // Top fluted brass chimney & heat shield dome
    addCylinder(tris, c + glm::vec3(0.0f, 0.18f, 0.0f), 0.065f, 0.07f, m.brass, 16);
    addOrientedCylinder(tris, c + glm::vec3(0.0f, 0.11f, 0.0f), c + glm::vec3(0.0f, 0.16f, 0.0f), 0.125f, 0.068f, m.brass, 16, true);

    // Bottom brass oil reservoir fount
    addCylinder(tris, c + glm::vec3(0.0f, -0.16f, 0.0f), 0.115f, 0.08f, m.brass, 16);
    addOrientedCylinder(tris, c + glm::vec3(0.0f, -0.12f, 0.0f), c + glm::vec3(0.0f, -0.09f, 0.0f), 0.115f, 0.095f, m.brass, 16, true);

    // Inner brass burner cone & glowing warm flame mantle
    addCylinder(tris, c + glm::vec3(0.0f, -0.07f, 0.0f), 0.038f, 0.04f, m.brass, 12);
    addSphere(tris, c + glm::vec3(0.0f, -0.01f, 0.0f), 0.042f, m.lanternFlame, 14, 14);
    addOrientedCylinder(tris, c + glm::vec3(0.0f, -0.04f, 0.0f), c + glm::vec3(0.0f, 0.04f, 0.0f), 0.032f, 0.018f, m.lanternFlame, 12, true);

    // Hollow smooth dielectric glass globe (outer wall + inner wall for realistic refraction)
    addOrientedCylinder(tris, c + glm::vec3(0.0f, -0.09f, 0.0f), c + glm::vec3(0.0f, 0.01f, 0.0f), 0.092f, 0.112f, m.glass, 18, false);
    addOrientedCylinder(tris, c + glm::vec3(0.0f,  0.01f, 0.0f), c + glm::vec3(0.0f, 0.11f, 0.0f), 0.112f, 0.088f, m.glass, 18, false);
    addOrientedCylinder(tris, c + glm::vec3(0.0f, -0.09f, 0.0f), c + glm::vec3(0.0f, 0.01f, 0.0f), 0.084f, 0.104f, m.glass, 18, false);
    addOrientedCylinder(tris, c + glm::vec3(0.0f,  0.01f, 0.0f), c + glm::vec3(0.0f, 0.11f, 0.0f), 0.104f, 0.080f, m.glass, 18, false);

    // Protective wrought-iron cage guard bars (6 vertical struts + equatorial ring)
    addTorus(tris, c + glm::vec3(0.0f, 0.01f, 0.0f), 0.126f, 0.008f, m.castIron, 18, 8, glm::vec3(0, 1, 0));
    const float PI = 3.14159265358979323846f;
    for (int i = 0; i < 6; ++i) {
        float ang = (2.0f * PI * static_cast<float>(i)) / 6.0f;
        glm::vec3 dir(std::cos(ang), 0.0f, std::sin(ang));
        addBeam(tris,
                c + dir * 0.110f + glm::vec3(0.0f, -0.14f, 0.0f),
                c + dir * 0.125f + glm::vec3(0.0f,  0.01f, 0.0f),
                0.012f, 0.012f, m.castIron);
        addBeam(tris,
                c + dir * 0.125f + glm::vec3(0.0f,  0.01f, 0.0f),
                c + dir * 0.105f + glm::vec3(0.0f,  0.13f, 0.0f),
                0.012f, 0.012f, m.castIron);
    }
    // Top bail arch connecting chimney to suspension hook
    addBeam(tris, c + glm::vec3(-0.09f, 0.14f, 0.0f), pivotOrigin + glm::vec3(-0.02f, -0.10f, 0.0f), 0.014f, 0.014f, m.castIron);
    addBeam(tris, c + glm::vec3( 0.09f, 0.14f, 0.0f), pivotOrigin + glm::vec3( 0.02f, -0.10f, 0.0f), 0.014f, 0.014f, m.castIron);
}

// Model 3: Heavy Timber Tunnel Support Arch & Overhead Lagging
static void buildTimberSupportModel(std::vector<TriangleGPU>& tris, const glm::vec3& baseOrigin, const MineshaftMaterials& m) {
    const float halfSpanBottom = 1.65f;
    const float halfSpanTop = 1.42f;
    const float postHeight = 2.85f;
    const float postThick = 0.26f;

    // Left and right battered oak posts (tilted inward toward tunnel crown)
    addBeam(tris,
            baseOrigin + glm::vec3(-halfSpanBottom, 0.0f, 0.0f),
            baseOrigin + glm::vec3(-halfSpanTop, postHeight, 0.0f),
            postThick, postThick, m.timber);
    addBeam(tris,
            baseOrigin + glm::vec3( halfSpanBottom, 0.0f, 0.0f),
            baseOrigin + glm::vec3( halfSpanTop, postHeight, 0.0f),
            postThick, postThick, m.timber);

    // Heavy horizontal collar / cap beam across the top
    addBox(tris, baseOrigin + glm::vec3(0.0f, postHeight + 0.12f, 0.0f),
           glm::vec3(halfSpanTop * 2.0f + 0.52f, 0.28f, 0.28f), 0.0f, m.timber);

    // 45-degree mortised knee braces
    addBeam(tris,
            baseOrigin + glm::vec3(-halfSpanTop - 0.06f, postHeight - 0.78f, 0.0f),
            baseOrigin + glm::vec3(-halfSpanTop + 0.72f, postHeight + 0.04f, 0.0f),
            0.20f, 0.20f, m.timber);
    addBeam(tris,
            baseOrigin + glm::vec3( halfSpanTop + 0.06f, postHeight - 0.78f, 0.0f),
            baseOrigin + glm::vec3( halfSpanTop - 0.72f, postHeight + 0.04f, 0.0f),
            0.20f, 0.20f, m.timber);

    // Cast-iron structural joint gusset plates & bolt heads
    for (int side = -1; side <= 1; side += 2) {
        float sx = static_cast<float>(side);
        glm::vec3 jointPos = baseOrigin + glm::vec3(sx * halfSpanTop, postHeight - 0.02f, 0.0f);
        addBox(tris, jointPos, glm::vec3(0.32f, 0.32f, 0.31f), 0.0f, m.castIron);
        addSphere(tris, jointPos + glm::vec3(0.0f, 0.06f,  0.17f), 0.025f, m.brass, 8, 8);
        addSphere(tris, jointPos + glm::vec3(0.0f, -0.06f, 0.17f), 0.025f, m.brass, 8, 8);
        addSphere(tris, jointPos + glm::vec3(0.0f, 0.06f, -0.17f), 0.025f, m.brass, 8, 8);
        addSphere(tris, jointPos + glm::vec3(0.0f, -0.06f,-0.17f), 0.025f, m.brass, 8, 8);
    }

    // Overhead & upper-wall rough-sawn lagging planks
    for (int p = -3; p <= 3; ++p) {
        float px = static_cast<float>(p) * 0.42f;
        addBox(tris, baseOrigin + glm::vec3(px, postHeight + 0.30f, 0.0f),
               glm::vec3(0.34f, 0.06f, 1.40f), static_cast<float>(p) * 1.5f, m.timber);
    }
    // Wrought-iron lantern hook projecting inward from left knee brace
    addBeam(tris,
            baseOrigin + glm::vec3(-halfSpanTop + 0.35f, postHeight - 0.22f, 0.0f),
            baseOrigin + glm::vec3(-halfSpanTop + 0.55f, postHeight - 0.22f, 0.0f),
            0.03f, 0.03f, m.castIron);
}

// Model 4: Steel Rollercoaster Track Segment & Wooden Sleepers (3.0m modular length along Z)
static void buildRailTrackSegmentModel(std::vector<TriangleGPU>& tris, const glm::vec3& origin, const MineshaftMaterials& m) {
    const float segLen = 3.0f;
    const float halfGauge = 0.46f;

    // 1. Crushed Rock Ballast Bed
    addFrustum(tris, origin + glm::vec3(0.0f, -0.08f, 0.0f),
               glm::vec2(1.95f, segLen), glm::vec2(1.45f, segLen), 0.12f, m.wetFloor);

    // 2. Five Rough-Hewn Wooden Sleepers (Cross-Ties) with Iron Tie-Plates & Spikes
    for (int i = -2; i <= 2; ++i) {
        float z = static_cast<float>(i) * 0.60f;
        float yawJitter = (i % 2 == 0) ? 0.8f : -0.6f;
        addBox(tris, origin + glm::vec3(0.0f, 0.02f, z), glm::vec3(1.52f, 0.11f, 0.22f), yawJitter, m.timber);

        for (int s = -1; s <= 1; s += 2) {
            float rx = static_cast<float>(s) * halfGauge;
            // Ribbed cast-iron tie-plate
            addBox(tris, origin + glm::vec3(rx, 0.085f, z), glm::vec3(0.18f, 0.022f, 0.20f), 0.0f, m.castIron);
            // 4 Rail dog-spikes
            addBox(tris, origin + glm::vec3(rx - 0.06f, 0.11f, z - 0.06f), glm::vec3(0.024f, 0.045f, 0.024f), 0.0f, m.castIron);
            addBox(tris, origin + glm::vec3(rx + 0.06f, 0.11f, z + 0.06f), glm::vec3(0.024f, 0.045f, 0.024f), 0.0f, m.castIron);
        }
    }

    // 3. Profiled I-Beam Steel Rails (Base Flange + Web + Polished Running Head)
    for (int s = -1; s <= 1; s += 2) {
        float rx = static_cast<float>(s) * halfGauge;
        // Wide bottom rail foot flange
        addBox(tris, origin + glm::vec3(rx, 0.105f, 0.0f), glm::vec3(0.11f, 0.022f, segLen), 0.0f, m.castIron);
        // Narrow vertical rail web
        addBox(tris, origin + glm::vec3(rx, 0.145f, 0.0f), glm::vec3(0.032f, 0.065f, segLen), 0.0f, m.castIron);
        // Mirror-brushed steel running head (catches bright specular highlights)
        addBox(tris, origin + glm::vec3(rx, 0.190f, 0.0f), glm::vec3(0.068f, 0.036f, segLen), 0.0f, m.railSteel);
        // Bolted fishplate joint bar at Z = +1.35m
        addBox(tris, origin + glm::vec3(rx, 0.145f, 1.35f), glm::vec3(0.056f, 0.050f, 0.28f), 0.0f, m.brass);
    }
}

// Model 5: Subterranean Glowing Crystal Cluster (Hexagonal Terminated Quartz/Fluorite Prisms)
static void buildCrystalClusterModel(std::vector<TriangleGPU>& tris, const glm::vec3& origin, const MineshaftMaterials& m, bool cyanVariant) {
    uint32_t outerMat = cyanVariant ? m.crystalCyan : m.crystalViolet;
    int coreMat = static_cast<int>(cyanVariant ? m.crystalCoreCyan : m.crystalCoreViolet);
    uint32_t accentMat = cyanVariant ? m.crystalViolet : m.crystalCyan;
    int accentCore = static_cast<int>(cyanVariant ? m.crystalCoreViolet : m.crystalCoreCyan);

    // Fractured host-rock matrix base with gold ore veins
    addFrustum(tris, origin + glm::vec3(0.0f, 0.10f, 0.0f),
               glm::vec2(0.85f, 0.85f), glm::vec2(0.55f, 0.55f), 0.22f, m.rock);
    addBox(tris, origin + glm::vec3(-0.22f, 0.14f, 0.16f), glm::vec3(0.32f, 0.22f, 0.28f), 28.0f, m.rock);
    addBox(tris, origin + glm::vec3( 0.24f, 0.12f,-0.14f), glm::vec3(0.28f, 0.20f, 0.30f), -35.0f, m.rock);
    addBox(tris, origin + glm::vec3( 0.08f, 0.20f, 0.22f), glm::vec3(0.16f, 0.12f, 0.16f), 15.0f, m.goldOre);

    // 9 Radiating Hexagonal Crystal Shards
    struct ShardSpec {
        glm::vec3 offset;
        glm::vec3 dir;
        float radius;
        float shaftLen;
        float tipLen;
        bool useAccent;
    };
    const ShardSpec shards[9] = {
        { glm::vec3( 0.00f, 0.16f,  0.00f), glm::vec3( 0.04f, 0.99f,  0.05f), 0.13f, 0.78f, 0.26f, false },
        { glm::vec3(-0.14f, 0.15f,  0.08f), glm::vec3(-0.34f, 0.90f,  0.22f), 0.10f, 0.62f, 0.21f, false },
        { glm::vec3( 0.15f, 0.15f,  0.06f), glm::vec3( 0.38f, 0.88f,  0.18f), 0.11f, 0.66f, 0.22f, false },
        { glm::vec3( 0.04f, 0.15f, -0.15f), glm::vec3( 0.12f, 0.86f, -0.42f), 0.095f,0.56f, 0.19f, true  },
        { glm::vec3(-0.16f, 0.14f, -0.12f), glm::vec3(-0.45f, 0.82f, -0.32f), 0.085f,0.48f, 0.17f, false },
        { glm::vec3( 0.22f, 0.12f, -0.10f), glm::vec3( 0.58f, 0.74f, -0.28f), 0.080f,0.44f, 0.15f, true  },
        { glm::vec3(-0.06f, 0.14f,  0.18f), glm::vec3(-0.18f, 0.80f,  0.54f), 0.090f,0.50f, 0.18f, true  },
        { glm::vec3(-0.26f, 0.11f,  0.02f), glm::vec3(-0.68f, 0.68f,  0.12f), 0.072f,0.38f, 0.14f, false },
        { glm::vec3( 0.18f, 0.11f,  0.20f), glm::vec3( 0.48f, 0.72f,  0.48f), 0.075f,0.40f, 0.14f, false }
    };

    for (const auto& s : shards) {
        addHexCrystalPrism(tris, origin + s.offset, s.dir, s.radius, s.shaftLen, s.tipLen,
                           s.useAccent ? accentMat : outerMat,
                           s.useAccent ? accentCore : coreMat);
    }
}

// Model 6: Heavy Timber Trestle Bridge Bent & Viaduct Deck (Supports rails over subterranean caverns)
static void buildTrestleSupportModel(std::vector<TriangleGPU>& tris, const glm::vec3& origin, const MineshaftMaterials& m, float trestleDepth = 5.4f) {
    // origin is at the rail-bed level (Y = 0). The trestle structure extends downward to Y = -trestleDepth
    // and upward to Y = +0.92m for the side wooden safety guardrails.
    const float topY = -0.06f;
    const float botY = -trestleDepth;

    // 1. Four Battered Heavy Oak Pilings (2 inner load-bearing posts + 2 outer batter posts)
    const float topX[4] = { -1.16f, -0.54f, 0.54f, 1.16f };
    const float botX[4] = { -1.16f - trestleDepth * 0.20f, -0.54f - trestleDepth * 0.08f,
                             0.54f + trestleDepth * 0.08f,  1.16f + trestleDepth * 0.20f };

    for (int i = 0; i < 4; ++i) {
        glm::vec3 pTop = origin + glm::vec3(topX[i], topY, 0.0f);
        glm::vec3 pBot = origin + glm::vec3(botX[i], botY + 0.22f, 0.0f);
        addBeam(tris, pBot, pTop, 0.20f, 0.22f, m.timber);

        // Stepped stone masonry pier footing at the cavern floor + cast-iron anchor collar
        addFrustum(tris, origin + glm::vec3(botX[i], botY + 0.11f, 0.0f),
                   glm::vec2(0.58f, 0.58f), glm::vec2(0.38f, 0.38f), 0.22f, m.rock);
        addBox(tris, origin + glm::vec3(botX[i], botY + 0.25f, 0.0f),
               glm::vec3(0.28f, 0.10f, 0.28f), 0.0f, m.castIron);
    }

    // 2. Heavy Transverse Top Cap Beam + Corbel Brackets
    addBox(tris, origin + glm::vec3(0.0f, -0.14f, 0.0f), glm::vec3(2.68f, 0.22f, 0.28f), 0.0f, m.timber);
    for (int s = -1; s <= 1; s += 2) {
        float sx = static_cast<float>(s);
        // Diagonal knee braces under the cap beam ends
        addBeam(tris,
                origin + glm::vec3(sx * 1.12f, -0.78f, 0.0f),
                origin + glm::vec3(sx * 1.28f, -0.18f, 0.0f),
                0.13f, 0.15f, m.timber);
        // Cast-iron cap strap plates & brass bolt heads
        addBox(tris, origin + glm::vec3(sx * 0.54f, -0.14f, 0.0f), glm::vec3(0.26f, 0.25f, 0.31f), 0.0f, m.castIron);
        addBox(tris, origin + glm::vec3(sx * 1.16f, -0.14f, 0.0f), glm::vec3(0.26f, 0.25f, 0.31f), 0.0f, m.castIron);
        addBox(tris, origin + glm::vec3(sx * 0.54f, -0.14f, 0.17f), glm::vec3(0.05f, 0.05f, 0.04f), 0.0f, m.brass);
        addBox(tris, origin + glm::vec3(sx * 1.16f, -0.14f, 0.17f), glm::vec3(0.05f, 0.05f, 0.04f), 0.0f, m.brass);
    }

    // 3. Longitudinal Track Stringer Girders & Diagonal Bay Bracing (3.0m modular span along Z)
    const float spanLen = 3.04f;
    for (int s = -1; s <= 1; s += 2) {
        float rx = static_cast<float>(s) * 0.48f;
        // Main longitudinal stringer beam carrying the sleepers across the abyss
        addBox(tris, origin + glm::vec3(rx, -0.26f, 0.0f), glm::vec3(0.20f, 0.24f, spanLen), 0.0f, m.timber);
        // Under-stringer longitudinal knee struts pitching down to the piling
        addBeam(tris,
                origin + glm::vec3(rx * 1.10f, -1.25f, 0.0f),
                origin + glm::vec3(rx, -0.32f,  1.25f),
                0.12f, 0.12f, m.timber);
        addBeam(tris,
                origin + glm::vec3(rx * 1.10f, -1.25f, 0.0f),
                origin + glm::vec3(rx, -0.32f, -1.25f),
                0.12f, 0.12f, m.timber);
    }

    // 4. Multi-Tier Horizontal Sway Ledgers & Crisscross X-Bracing
    const int numTiers = 2;
    float tierFracs[3] = { 0.04f, 0.44f, 0.84f };
    for (int t = 0; t < numTiers; ++t) {
        float f0 = tierFracs[t];
        float f1 = tierFracs[t + 1];
        float y0 = glm::mix(topY, botY, f0);
        float y1 = glm::mix(topY, botY, f1);
        float spanW1 = glm::mix(2.50f, 2.50f + trestleDepth * 0.42f, f1);

        // Horizontal double sway-ledger beams clamping the front & back of the pilings at y1
        addBox(tris, origin + glm::vec3(0.0f, y1,  0.13f), glm::vec3(spanW1, 0.14f, 0.08f), 0.0f, m.timber);
        addBox(tris, origin + glm::vec3(0.0f, y1, -0.13f), glm::vec3(spanW1, 0.14f, 0.08f), 0.0f, m.timber);

        // Iron gusset tie-plates at each piling intersection on the ledger
        for (int i = 0; i < 4; ++i) {
            float x1 = glm::mix(topX[i], botX[i], f1);
            addBox(tris, origin + glm::vec3(x1, y1, 0.0f), glm::vec3(0.24f, 0.18f, 0.36f), 0.0f, m.castIron);
            addBox(tris, origin + glm::vec3(x1, y1, 0.19f), glm::vec3(0.045f, 0.045f, 0.04f), 0.0f, m.brass);
        }

        // Center bay diagonal X-brace struts (between inner posts 1 and 2)
        glm::vec3 cTL = origin + glm::vec3(glm::mix(topX[1], botX[1], f0), y0, 0.0f);
        glm::vec3 cTR = origin + glm::vec3(glm::mix(topX[2], botX[2], f0), y0, 0.0f);
        glm::vec3 cBL = origin + glm::vec3(glm::mix(topX[1], botX[1], f1), y1, 0.0f);
        glm::vec3 cBR = origin + glm::vec3(glm::mix(topX[2], botX[2], f1), y1, 0.0f);
        addBeam(tris, cBL + glm::vec3(0, 0,  0.04f), cTR + glm::vec3(0, 0,  0.04f), 0.11f, 0.09f, m.timber);
        addBeam(tris, cBR + glm::vec3(0, 0, -0.04f), cTL + glm::vec3(0, 0, -0.04f), 0.11f, 0.09f, m.timber);
        // Center X-joint iron clamp plate
        addBox(tris, 0.25f * (cTL + cTR + cBL + cBR), glm::vec3(0.18f, 0.18f, 0.18f), 0.0f, m.castIron);

        // Left & right outer bay diagonal struts
        glm::vec3 lTL = origin + glm::vec3(glm::mix(topX[0], botX[0], f0), y0, 0.0f);
        glm::vec3 lBL = origin + glm::vec3(glm::mix(topX[0], botX[0], f1), y1, 0.0f);
        addBeam(tris, lBL, cTL, 0.10f, 0.10f, m.timber);
        addBeam(tris, cBL, lTL, 0.10f, 0.10f, m.timber);

        glm::vec3 rTR = origin + glm::vec3(glm::mix(topX[3], botX[3], f0), y0, 0.0f);
        glm::vec3 rBR = origin + glm::vec3(glm::mix(topX[3], botX[3], f1), y1, 0.0f);
        addBeam(tris, rBR, cTR, 0.10f, 0.10f, m.timber);
        addBeam(tris, cBR, rTR, 0.10f, 0.10f, m.timber);
    }

    // 5. Side Catwalk Planks & Timber Safety Guardrails (Framing the track at Y = 0..+0.90m)
    for (int s = -1; s <= 1; s += 2) {
        float sx = static_cast<float>(s);
        // Weathered wooden catwalk board outside the rails
        addBox(tris, origin + glm::vec3(sx * 0.96f, -0.01f, 0.0f), glm::vec3(0.28f, 0.045f, spanLen), 0.0f, m.timber);
        // Vertical guardrail posts at Z = -0.95m and Z = +0.95m
        for (int zp = -1; zp <= 1; zp += 2) {
            float zPost = static_cast<float>(zp) * 0.95f;
            addBeam(tris,
                    origin + glm::vec3(sx * 1.18f, -0.16f, zPost),
                    origin + glm::vec3(sx * 1.24f,  0.88f, zPost),
                    0.09f, 0.09f, m.timber);
            addBox(tris, origin + glm::vec3(sx * 1.18f, 0.02f, zPost), glm::vec3(0.13f, 0.10f, 0.13f), 0.0f, m.castIron);
        }
        // Top and mid longitudinal safety handrail beams
        addBox(tris, origin + glm::vec3(sx * 1.23f, 0.82f, 0.0f), glm::vec3(0.07f, 0.08f, spanLen), 0.0f, m.timber);
        addBox(tris, origin + glm::vec3(sx * 1.21f, 0.44f, 0.0f), glm::vec3(0.06f, 0.07f, spanLen), 0.0f, m.timber);
    }
}

SceneData ProceduralScene::createMineshaftModelShowcase(const std::string& modelName) {
    SceneData scene;
    MineshaftMaterials m = initMineshaftMaterials(scene);

    // Dark subterranean studio stage with wet rock floor & warm rim/key area lights
    addBox(scene.triangles, glm::vec3(0.0f, -0.10f, 0.0f), glm::vec3(28.0f, 0.20f, 20.0f), 0.0f, m.wetFloor);
    addBox(scene.triangles, glm::vec3(0.0f,  3.20f, -3.2f), glm::vec3(28.0f, 8.0f, 0.30f), 0.0f, m.rock);

    auto addAreaLight = [&](glm::vec3 pos, glm::vec3 uVec, glm::vec3 vVec, glm::vec3 norm, glm::vec3 emission) {
        LightGPU l{};
        l.position = glm::vec4(pos, 0.0f);
        l.u = glm::vec4(uVec, 0.0f);
        l.v = glm::vec4(vVec, 0.0f);
        float area = glm::length(glm::cross(uVec, vVec));
        l.emission = glm::vec4(emission, area);
        l.normal = glm::vec4(glm::normalize(norm), 0.0f);
        scene.lights.push_back(l);
    };

    // Warm key light (left-front) + cool cave fill light (right) + top rim light
    addAreaLight(glm::vec3(-2.2f, 2.6f, 2.2f), glm::vec3(0.8f, 0.0f, 0.4f), glm::vec3(0.0f, 0.8f, 0.0f),
                 glm::vec3(0.6f, -0.5f, -0.6f), glm::vec3(16.0f, 12.0f, 7.5f));
    addAreaLight(glm::vec3( 2.4f, 1.8f, 1.6f), glm::vec3(-0.4f, 0.0f, 0.8f), glm::vec3(0.0f, 0.8f, 0.0f),
                 glm::vec3(-0.7f, -0.3f, -0.6f), glm::vec3(5.0f, 9.0f, 14.0f));
    addAreaLight(glm::vec3(-0.6f, 3.6f, -0.4f), glm::vec3(1.2f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.8f),
                 glm::vec3(0.0f, -1.0f, 0.0f), glm::vec3(10.0f, 9.5f, 8.5f));

    scene.hasCamera = true;
    scene.cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    scene.cameraFov = 42.0f;

    if (modelName == "cart" || modelName == "minecart") {
        // Place a short section of rail track under the cart
        buildRailTrackSegmentModel(scene.triangles, glm::vec3(0.0f, 0.0f, 0.0f), m);
        buildMinecartModel(scene.triangles, glm::vec3(0.0f, 0.19f, 0.0f), m);
        scene.cameraPosition = glm::vec3(1.85f, 1.35f, 2.15f);
        scene.cameraTarget = glm::vec3(0.0f, 0.68f, 0.10f);
    } else if (modelName == "lantern") {
        // Wooden gallows post holding the hanging lantern
        addBeam(scene.triangles, glm::vec3(-0.45f, 0.0f, 0.0f), glm::vec3(-0.45f, 1.35f, 0.0f), 0.12f, 0.12f, m.timber);
        addBeam(scene.triangles, glm::vec3(-0.50f, 1.30f, 0.0f), glm::vec3( 0.05f, 1.30f, 0.0f), 0.10f, 0.10f, m.timber);
        buildMineLanternModel(scene.triangles, glm::vec3(0.0f, 1.25f, 0.0f), m);
        addAreaLight(glm::vec3(-0.04f, 0.90f, 0.04f), glm::vec3(0.08f, 0.0f, 0.0f), glm::vec3(0.0f, 0.08f, 0.0f),
                     glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(28.0f, 14.0f, 4.0f));
        scene.cameraPosition = glm::vec3(0.58f, 0.98f, 0.82f);
        scene.cameraTarget = glm::vec3(-0.04f, 0.90f, 0.0f);
    } else if (modelName == "timber" || modelName == "arch") {
        buildTimberSupportModel(scene.triangles, glm::vec3(0.0f, 0.0f, 0.0f), m);
        buildMineLanternModel(scene.triangles, glm::vec3(-0.92f, 2.60f, 0.0f), m);
        scene.cameraPosition = glm::vec3(1.65f, 1.75f, 4.30f);
        scene.cameraTarget = glm::vec3(0.0f, 1.55f, 0.0f);
    } else if (modelName == "trestle" || modelName == "bridge") {
        // Showcase the wooden trestle bridge bent with steel rails on top and a lantern on the guardrail
        const float showcaseHeight = 2.65f;
        buildTrestleSupportModel(scene.triangles, glm::vec3(0.0f, showcaseHeight, 0.0f), m, showcaseHeight);
        buildRailTrackSegmentModel(scene.triangles, glm::vec3(0.0f, showcaseHeight, 0.0f), m);
        buildMineLanternModel(scene.triangles, glm::vec3(-1.24f, showcaseHeight + 0.85f, 0.95f), m);
        addAreaLight(glm::vec3(-1.10f, showcaseHeight + 0.55f, 1.05f), glm::vec3(0.10f, 0.0f, 0.0f), glm::vec3(0.0f, 0.10f, 0.0f),
                     glm::vec3(0.5f, -0.3f, 0.8f), glm::vec3(24.0f, 13.0f, 4.0f));
        scene.cameraPosition = glm::vec3(2.55f, 2.05f, 4.45f);
        scene.cameraTarget = glm::vec3(0.0f, 1.55f, 0.0f);
    } else if (modelName == "track" || modelName == "rails") {
        buildRailTrackSegmentModel(scene.triangles, glm::vec3(0.0f, 0.0f, -1.5f), m);
        buildRailTrackSegmentModel(scene.triangles, glm::vec3(0.0f, 0.0f,  1.5f), m);
        scene.cameraPosition = glm::vec3(1.35f, 0.95f, 2.65f);
        scene.cameraTarget = glm::vec3(0.0f, 0.15f, -0.30f);
    } else {
        // Crystal cluster showcase
        buildCrystalClusterModel(scene.triangles, glm::vec3(-0.42f, 0.0f, 0.0f), m, false);
        buildCrystalClusterModel(scene.triangles, glm::vec3( 0.48f, 0.0f, -0.15f), m, true);
        scene.cameraPosition = glm::vec3(0.15f, 0.85f, 1.95f);
        scene.cameraTarget = glm::vec3(0.02f, 0.45f, 0.0f);
    }

    scene.boundsMin = glm::vec3(-14.0f, -0.2f, -10.0f);
    scene.boundsMax = glm::vec3( 14.0f,  7.5f,  10.0f);
    scene.sceneRadius = 15.0f;
    scene.focalRadius = 2.5f;
    scene.focalDistance = glm::length(scene.cameraTarget - scene.cameraPosition);
    scene.centralTarget = scene.cameraTarget;
    scene.hasDielectrics = true;
    scene.dielectricBoundsMin = glm::vec3(-1.5f, 0.0f, -1.5f);
    scene.dielectricBoundsMax = glm::vec3( 1.5f, 3.5f,  1.5f);
    return scene;
}

// =========================================================================================
// 500-Meter, 30-Second 4-Zone Mineshaft Coaster Spline & Kinematic Ride Evaluator
// =========================================================================================
// Parameter s in [0, 500] meters represents distance along the primary Z progression (Z = -s).
//   Zone 1 (  0..115m): Old Timber Shaft & First Steep Plunge (-18m drop)
//   Zone 2 (115..235m): Flooded Sluiceway & High-Speed Banked S-Curves
//   Zone 3 (235..375m): Grand Crystal Geode Cavern & Wooden Trestle Viaduct over Abyss
//   Zone 4 (375..500m): Gold-Vein Corkscrew Slalom & Deep Geothermal Finale
static glm::vec3 evalMineshaftTrackCenter(float s) {
    const float PI = 3.14159265358979323846f;
    s = glm::clamp(s, -5.0f, 510.0f);

    // Smooth zone envelopes
    float wZ1 = 1.0f - glm::smoothstep(100.0f, 130.0f, s);
    float wZ2 = glm::smoothstep(100.0f, 130.0f, s) * (1.0f - glm::smoothstep(220.0f, 250.0f, s));
    float wZ3 = glm::smoothstep(220.0f, 250.0f, s) * (1.0f - glm::smoothstep(360.0f, 390.0f, s));
    float wZ4 = glm::smoothstep(360.0f, 390.0f, s);

    // Lateral X(s):
    // Zone 1: Sweeping right-then-left plunge curve
    float x1 = 6.5f * std::sin(s * (2.0f * PI / 110.0f)) - 1.8f * std::cos(s * (2.0f * PI / 55.0f)) + 1.8f;
    // Zone 2: Rapid banked S-curve chicanes through the flooded sluiceway
    float x2 = 10.5f * std::sin((s - 115.0f) * (2.0f * PI / 68.0f));
    // Zone 3: Majestic wide sweeping viaduct curve across the Grand Crystal Cavern
    float x3 = 14.0f * std::sin((s - 235.0f) * (PI / 140.0f)) - 4.5f * std::sin((s - 235.0f) * (2.0f * PI / 70.0f));
    // Zone 4: Tight high-G corkscrew slalom into the gold-vein forge brake run
    float brakeTaper = 1.0f - glm::smoothstep(455.0f, 492.0f, s);
    float x4 = brakeTaper * (9.2f * std::sin((s - 375.0f) * (2.0f * PI / 52.0f)));

    float x = wZ1 * x1 + wZ2 * x2 + wZ3 * x3 + wZ4 * x4;

    // Vertical Y(s): Rollercoaster elevation profile (drops from Y = 0m down to Y = -58m)
    // Base descent slope + First Big Plunge (s = 18..78m) + Sluiceway Camelback + Cavern Viaduct Arch + Finale Drop
    float firstPlunge = -14.5f * glm::smoothstep(14.0f, 72.0f, s);
    float camelback   =  2.8f  * std::sin(glm::clamp((s - 72.0f) / 48.0f, 0.0f, 1.0f) * PI);
    float sluiceDrop  = -11.0f * glm::smoothstep(115.0f, 235.0f, s)
                      +  1.8f  * std::sin(glm::clamp((s - 120.0f) / 115.0f, 0.0f, 1.0f) * 3.0f * PI);
    float cavernArch  = -8.5f  * glm::smoothstep(235.0f, 375.0f, s)
                      +  3.6f  * std::sin(glm::clamp((s - 235.0f) / 140.0f, 0.0f, 1.0f) * PI);
    float goldPlunge  = -16.0f * glm::smoothstep(370.0f, 475.0f, s)
                      +  2.2f  * std::sin(glm::clamp((s - 375.0f) / 90.0f, 0.0f, 1.0f) * 2.0f * PI);

    float y = firstPlunge + camelback + sluiceDrop + cavernArch + goldPlunge;
    float z = 12.0f - s; // Starts at Z = +12m (s=0) and runs to Z = -488m (s=500)
    return glm::vec3(x, y, z);
}

static void evalMineshaftTrackFrame(float s, glm::vec3& outPos, glm::vec3& outRight, glm::vec3& outUp, glm::vec3& outForward) {
    const float ds = 0.6f;
    outPos = evalMineshaftTrackCenter(s);
    glm::vec3 nextPos = evalMineshaftTrackCenter(s + ds);
    glm::vec3 prevPos = evalMineshaftTrackCenter(s - ds);
    outForward = glm::normalize(nextPos - prevPos);

    glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
    glm::vec3 flatRight = glm::cross(worldUp, outForward);
    float flatLen = glm::length(flatRight);
    flatRight = (flatLen > 1e-5f) ? (flatRight / flatLen) : glm::vec3(-1.0f, 0.0f, 0.0f);

    // Centripetal lateral curvature banking: rolls the cart into every left/right turn
    glm::vec3 accel = (nextPos - 2.0f * outPos + prevPos) / (ds * ds);
    float latCurv = glm::dot(accel, flatRight);
    float bankAngle = glm::clamp(latCurv * 6.5f, -0.44f, 0.44f); // up to ~25 deg roll banking

    glm::vec3 bankedUp = glm::normalize(worldUp + flatRight * std::tan(bankAngle));
    outRight = glm::normalize(glm::cross(bankedUp, outForward));
    outUp    = glm::normalize(glm::cross(outForward, outRight));
}

static glm::mat4 makeCoasterFrameMatrix(const glm::vec3& pos, const glm::vec3& right, const glm::vec3& up, const glm::vec3& forward) {
    glm::mat4 m(1.0f);
    m[0] = glm::vec4(right, 0.0f);
    m[1] = glm::vec4(up, 0.0f);
    m[2] = glm::vec4(forward, 0.0f);
    m[3] = glm::vec4(pos, 1.0f);
    return m;
}

// Maps elapsed ride time (0..30s loop) to track parameter s in [6.0, 476.0] meters
// with physically motivated gravity acceleration on plunges and slight slowdowns on crests.
static float mapCoasterTimeToDistance(float simTimeSec, float* outSpeedMps = nullptr) {
    static constexpr float RIDE_DURATION = 30.0f;
    static constexpr float S_START = 6.0f;
    static constexpr float S_END   = 476.0f;
    static constexpr int   STEPS   = 300;

    // Build static pre-integrated time->distance table once
    struct CoasterLut {
        float sTable[STEPS + 1];
        float vTable[STEPS + 1];
        CoasterLut() {
            float rawTime[STEPS + 1];
            rawTime[0] = 0.0f;
            float ds = (S_END - S_START) / static_cast<float>(STEPS);
            for (int i = 0; i <= STEPS; ++i) {
                float s = S_START + static_cast<float>(i) * ds;
                glm::vec3 p0 = evalMineshaftTrackCenter(s - 0.5f);
                glm::vec3 p1 = evalMineshaftTrackCenter(s + 0.5f);
                float slope = (p1.y - p0.y); // negative on downhill drops

                // Downhill drops boost speed; cavern bridge crest savors the view; finale stays fast
                float zoneCavernSlow = 1.0f - 0.14f * glm::smoothstep(240.0f, 275.0f, s) * (1.0f - glm::smoothstep(335.0f, 365.0f, s));
                float vRel = std::clamp(1.0f - 1.35f * slope, 0.68f, 1.42f) * zoneCavernSlow;
                vTable[i] = vRel;
                if (i > 0) {
                    float avgV = 0.5f * (vTable[i - 1] + vTable[i]);
                    rawTime[i] = rawTime[i - 1] + (ds / avgV);
                }
            }
            float totalRawTime = rawTime[STEPS];
            float speedScale = totalRawTime / RIDE_DURATION;
            for (int i = 0; i <= STEPS; ++i) {
                vTable[i] *= speedScale;
            }
            // Resample uniformly in normalized time u in [0, 1]
            sTable[0] = S_START;
            sTable[STEPS] = S_END;
            int k = 0;
            for (int i = 1; i < STEPS; ++i) {
                float targetT = (static_cast<float>(i) / static_cast<float>(STEPS)) * totalRawTime;
                while (k + 1 < STEPS && rawTime[k + 1] < targetT) {
                    ++k;
                }
                float segDt = std::max(1e-6f, rawTime[k + 1] - rawTime[k]);
                float alpha = (targetT - rawTime[k]) / segDt;
                sTable[i] = S_START + (static_cast<float>(k) + alpha) * ds;
            }
        }
    };
    static const CoasterLut lut{};

    float wrappedT = std::fmod(simTimeSec, RIDE_DURATION);
    if (wrappedT < 0.0f) wrappedT += RIDE_DURATION;

    float fIdx = (wrappedT / RIDE_DURATION) * static_cast<float>(STEPS);
    int idx0 = std::clamp(static_cast<int>(fIdx), 0, STEPS - 1);
    float frac = fIdx - static_cast<float>(idx0);
    float s = glm::mix(lut.sTable[idx0], lut.sTable[idx0 + 1], frac);
    if (outSpeedMps) {
        *outSpeedMps = (lut.sTable[idx0 + 1] - lut.sTable[idx0]) * (static_cast<float>(STEPS) / RIDE_DURATION);
    }
    return s;
}

void ProceduralScene::updateMineshaftCoaster(
    SceneData& scene, float simTimeSec,
    glm::vec3& outCamPos, glm::vec3& outCamTarget, glm::vec3& outCamUp, float& outCamFov,
    glm::mat4& outRiderCartMat, glm::mat4& outLeadCart1Mat, glm::mat4& outLeadCart2Mat)
{
    float speedMps = 15.6f;
    float sRider = mapCoasterTimeToDistance(simTimeSec, &speedMps);

    // Lead Cart #1 and #2 stay ahead on the rails with subtle accordion spacing on drops/climbs
    float sLead1 = sRider + 6.6f + 1.35f * std::sin(simTimeSec * 1.65f);
    float sLead2 = sRider + 14.8f + 1.90f * std::cos(simTimeSec * 1.25f);

    glm::vec3 rPos, rRight, rUp, rFwd;
    evalMineshaftTrackFrame(sRider, rPos, rRight, rUp, rFwd);
    outRiderCartMat = makeCoasterFrameMatrix(rPos, rRight, rUp, rFwd);

    glm::vec3 c1Pos, c1Right, c1Up, c1Fwd;
    evalMineshaftTrackFrame(sLead1, c1Pos, c1Right, c1Up, c1Fwd);
    outLeadCart1Mat = makeCoasterFrameMatrix(c1Pos, c1Right, c1Up, c1Fwd);

    glm::vec3 c2Pos, c2Right, c2Up, c2Fwd;
    evalMineshaftTrackFrame(sLead2, c2Pos, c2Right, c2Up, c2Fwd);
    outLeadCart2Mat = makeCoasterFrameMatrix(c2Pos, c2Right, c2Up, c2Fwd);

    // First-person Rider Camera mounted at the front prow of the Rider's Minecart:
    // Positioned ahead and above the internal ore load so the front riveted rim & brass carbide headlamp
    // sit cleanly at the bottom edge of the frame while looking ahead at Lead Cart #1 & #2!
    glm::vec3 lookPos, lRight, lUp, lFwd;
    evalMineshaftTrackFrame(sRider + 6.5f, lookPos, lRight, lUp, lFwd);

    // Subtle high-frequency rail clatter vibration proportional to speed
    float vib = 0.008f * glm::clamp(speedMps / 16.0f, 0.5f, 1.5f);
    glm::vec3 railShake = rUp * (vib * std::sin(simTimeSec * 34.0f)) + rRight * (vib * 0.7f * std::cos(simTimeSec * 27.0f));

    outCamPos    = rPos + rUp * 1.56f + rFwd * 0.52f + railShake;
    outCamTarget = lookPos + lUp * 0.92f;
    outCamUp     = glm::normalize(glm::mix(glm::vec3(0.0f, 1.0f, 0.0f), rUp, 0.78f));
    outCamFov    = glm::clamp(55.0f + (speedMps - 14.0f) * 0.9f, 52.0f, 62.0f);

    // Update Active Sliding-Window Light Buffer (Fixed 32 Lights = 6 Dynamic Cart Lights + 26 Nearest Static Lights)
    auto makePointAreaLight = [](glm::vec3 pos, glm::vec3 forwardDir, glm::vec3 emission, float size) -> LightGPU {
        glm::vec3 f = glm::normalize(forwardDir);
        glm::vec3 up = std::abs(f.y) < 0.95f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        glm::vec3 r = glm::normalize(glm::cross(f, up));
        glm::vec3 u = glm::normalize(glm::cross(r, f));

        LightGPU l{};
        l.position = glm::vec4(pos - r * (size * 0.5f) - u * (size * 0.5f), 0.0f);
        l.u = glm::vec4(r * size, 0.0f);
        l.v = glm::vec4(u * size, 0.0f);
        l.emission = glm::vec4(emission, size * size);
        l.normal = glm::vec4(f, 0.0f);
        return l;
    };

    const size_t kMaxActiveLights = 32;
    scene.lights.clear();
    scene.lights.reserve(kMaxActiveLights);

    // 1. Six Dynamic Cart Lights (2 per Minecart: Forward Carbide Headlamp Beam + Warm Rim/Tail Fill)
    // Rider Cart headlamp beam + cockpit rim fill
    glm::vec3 riderLampPos = rPos + rUp * 1.04f + rFwd * 1.04f;
    scene.lights.push_back(makePointAreaLight(riderLampPos, glm::normalize(rFwd - rUp * 0.12f), glm::vec3(95.0f, 74.0f, 40.0f), 0.22f));
    scene.lights.push_back(makePointAreaLight(rPos + rUp * 1.38f + rFwd * 0.35f, glm::normalize(rFwd - rUp * 0.45f), glm::vec3(36.0f, 26.0f, 14.0f), 0.16f));

    // Lead Cart #1 headlamp beam + rear amber tail-light illuminating Cart #1's back & rails
    glm::vec3 c1LampPos = c1Pos + c1Up * 1.04f + c1Fwd * 1.04f;
    scene.lights.push_back(makePointAreaLight(c1LampPos, glm::normalize(c1Fwd - c1Up * 0.14f), glm::vec3(120.0f, 92.0f, 48.0f), 0.24f));
    scene.lights.push_back(makePointAreaLight(c1Pos + c1Up * 0.85f - c1Fwd * 0.95f, glm::normalize(-c1Fwd - c1Up * 0.25f), glm::vec3(45.0f, 20.0f, 6.0f), 0.18f));

    // Lead Cart #2 headlamp beam + local fill
    glm::vec3 c2LampPos = c2Pos + c2Up * 1.04f + c2Fwd * 1.04f;
    scene.lights.push_back(makePointAreaLight(c2LampPos, glm::normalize(c2Fwd - c2Up * 0.14f), glm::vec3(110.0f, 84.0f, 44.0f), 0.24f));
    scene.lights.push_back(makePointAreaLight(c2Pos + c2Up * 0.90f - c2Fwd * 0.95f, glm::normalize(-c2Fwd - c2Up * 0.20f), glm::vec3(38.0f, 18.0f, 5.0f), 0.18f));

    // 2. Select the 26 Nearest Static Tunnel/Cavern Lights around (sRider + 15m)
    if (!scene.coasterStaticLights.empty()) {
        const float focusS = sRider + 15.0f;
        std::vector<size_t> order(scene.coasterStaticLights.size());
        for (size_t i = 0; i < order.size(); ++i) order[i] = i;

        size_t takeCount = std::min(kMaxActiveLights - scene.lights.size(), order.size());
        std::partial_sort(order.begin(), order.begin() + takeCount, order.end(),
            [&](size_t a, size_t b) {
                return std::abs(scene.coasterStaticLights[a].trackDistance - focusS) <
                       std::abs(scene.coasterStaticLights[b].trackDistance - focusS);
            });

        for (size_t i = 0; i < takeCount; ++i) {
            scene.lights.push_back(scene.coasterStaticLights[order[i]].light);
        }
    }
}

SceneData ProceduralScene::createMineshaftScene() {
    SceneData scene;
    scene.isMineshaftCoaster = true;
    MineshaftMaterials m = initMineshaftMaterials(scene);

    const float PI = 3.14159265358979323846f;
    const float totalTrackLength = 500.0f;
    const int   numChunks        = 25;   // 25 spatial BLAS chunks x 20.0m = 500.0m
    const int   ringsPerChunk    = 10;   // 2.0m per rock ring -> 250 rings total
    const int   totalRings       = numChunks * ringsPerChunk;
    const int   ringSectors      = 10;

    auto recordBlasPrototype = [&](const std::string& name, uint32_t startTri) -> uint32_t {
        uint32_t bIdx = static_cast<uint32_t>(scene.blasRanges.size());
        uint32_t triCount = static_cast<uint32_t>(scene.triangles.size() - startTri);
        BlasGeometryRange range{};
        range.firstTriangle = startTri;
        range.triangleCount = triCount;
        range.numOpaqueTriangles = 0;
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
        return bIdx;
    };

    // Helper for tunnel/cavern cross-section geometry at ring index r (0..totalRings)
    auto getTunnelRingVertex = [&](int ringIdx, int secIdx) -> std::pair<glm::vec3, glm::vec3> {
        float s = (static_cast<float>(ringIdx) / static_cast<float>(totalRings)) * totalTrackLength;
        glm::vec3 cPos, rAxis, uAxis, fAxis;
        evalMineshaftTrackFrame(s, cPos, rAxis, uAxis, fAxis);

        // Smoothly expand tunnel in Zone 2 (Sluiceway: s=115..235m) and dramatically in Zone 3 (Grand Cavern: s=235..375m)
        float sluiceEnv = glm::smoothstep(105.0f, 130.0f, s) * (1.0f - glm::smoothstep(225.0f, 245.0f, s));
        float cavernEnv = glm::smoothstep(228.0f, 256.0f, s) * (1.0f - glm::smoothstep(352.0f, 382.0f, s));

        float tunnelW   = 2.10f + 0.75f * sluiceEnv + 7.40f * cavernEnv;
        float tunnelH   = 3.20f + 0.45f * sluiceEnv + 6.80f * cavernEnv;
        float floorDrop = -0.08f - 5.25f * cavernEnv; // In Zone 3 cavern, floor drops 5.3m below the rails for the trestle bridge!

        float u = static_cast<float>(secIdx) / static_cast<float>(ringSectors);
        float ang = glm::mix(-0.14f * PI, 1.14f * PI, u);

        // Deterministic multi-frequency rock displacement
        float hash = std::sin(static_cast<float>(ringIdx * 17 + secIdx * 31) * 1.37f) * (0.18f + 0.45f * cavernEnv) +
                     std::cos(static_cast<float>(ringIdx * 7 - secIdx * 13) * 0.83f) * (0.12f + 0.30f * cavernEnv);
        float rx = std::cos(ang) * (tunnelW + hash);
        float ry = std::max(floorDrop, std::sin(ang) * (tunnelH + hash) + 1.05f);

        // At floor corners (secIdx == 0 or ringSectors), pin ry to floorDrop so the floor ribbon seals cleanly
        if (secIdx == 0 || secIdx == ringSectors) {
            ry = floorDrop;
        }

        glm::vec3 worldPos = cPos + rAxis * rx + uAxis * ry;
        glm::vec3 inwardNorm = glm::normalize(-rAxis * std::cos(ang) - uAxis * std::sin(ang));
        return { worldPos, inwardNorm };
    };

    // 1. Build 25 Spatial 20-Meter Tunnel/Cavern BLAS Chunks (Disjoint Z-Intervals for O(1) BVH Traversal)
    std::vector<uint32_t> chunkBlasIndices;
    chunkBlasIndices.reserve(numChunks);

    for (int chunk = 0; chunk < numChunks; ++chunk) {
        uint32_t startTri = static_cast<uint32_t>(scene.triangles.size());
        int rStart = chunk * ringsPerChunk;
        int rEnd   = rStart + ringsPerChunk;

        for (int r = rStart; r < rEnd; ++r) {
            float sMid = ((static_cast<float>(r) + 0.5f) / static_cast<float>(totalRings)) * totalTrackLength;
            bool inFloodedZone = (sMid >= 115.0f && sMid <= 378.0f);
            uint32_t floorMat  = inFloodedZone ? m.waterSurface : m.wetFloor;

            for (int sec = 0; sec < ringSectors; ++sec) {
                auto [p00, n00] = getTunnelRingVertex(r, sec);
                auto [p10, n10] = getTunnelRingVertex(r + 1, sec);
                auto [p11, n11] = getTunnelRingVertex(r + 1, sec + 1);
                auto [p01, n01] = getTunnelRingVertex(r, sec + 1);
                glm::vec3 faceNorm = glm::normalize(n00 + n10 + n11 + n01);
                addQuad(scene.triangles, p00, p10, p11, p01, faceNorm, m.rock);
            }

            // Cave/Sluiceway/Cavern Chasm Floor Ribbon
            auto [left0, nl0]   = getTunnelRingVertex(r, 0);
            auto [right0, nr0]  = getTunnelRingVertex(r, ringSectors);
            auto [left1, nl1]   = getTunnelRingVertex(r + 1, 0);
            auto [right1, nr1]  = getTunnelRingVertex(r + 1, ringSectors);
            addQuad(scene.triangles, left0, right0, right1, left1, glm::vec3(0.0f, 1.0f, 0.0f), floorMat);

            // Zone 2 (s = 115..235m): Rusty Iron & Brass Sluiceway Drainage Pipes along the lower tunnel walls
            if (sMid >= 115.0f && sMid <= 232.0f) {
                glm::vec3 c0, r0, u0, f0;
                glm::vec3 c1, r1, u1, f1;
                float s0 = (static_cast<float>(r) / static_cast<float>(totalRings)) * totalTrackLength;
                float s1 = (static_cast<float>(r + 1) / static_cast<float>(totalRings)) * totalTrackLength;
                evalMineshaftTrackFrame(s0, c0, r0, u0, f0);
                evalMineshaftTrackFrame(s1, c1, r1, u1, f1);
                for (int side = -1; side <= 1; side += 2) {
                    float sx = static_cast<float>(side) * 2.15f;
                    glm::vec3 pipeA = c0 + r0 * sx + u0 * 0.55f;
                    glm::vec3 pipeB = c1 + r1 * sx + u1 * 0.55f;
                    addOrientedCylinder(scene.triangles, pipeA, pipeB, 0.14f, 0.14f, m.castIron, 10, false);
                    if (r % 2 == 0) {
                        addOrientedCylinder(scene.triangles, pipeA - f0 * 0.06f, pipeA + f0 * 0.06f, 0.19f, 0.19f, m.brass, 10, true);
                    }
                }
            }

            // Zone 3 (s = 240..370m): Hanging Cavern Stalactites overhead in the Grand Crystal Geode Cavern
            if (sMid >= 242.0f && sMid <= 368.0f && (r % 2 == 0)) {
                glm::vec3 cPos, rAxis, uAxis, fAxis;
                evalMineshaftTrackFrame(sMid, cPos, rAxis, uAxis, fAxis);
                for (int st = -1; st <= 1; st += 2) {
                    float offX = static_cast<float>(st) * (2.6f + 1.8f * std::sin(static_cast<float>(r * 13 + st)));
                    glm::vec3 ceilBase = cPos + rAxis * offX + uAxis * 9.2f;
                    glm::vec3 tipPos   = ceilBase - uAxis * (2.4f + 0.9f * std::cos(static_cast<float>(r * 7)));
                    addOrientedCylinder(scene.triangles, ceilBase, tipPos, 0.55f, 0.04f, m.rock, 8, true);
                }
            }

            // Zone 4 (s = 375..495m): Glittering Metallic Gold Ore Veins protruding from the deep rock walls
            if (sMid >= 375.0f && sMid <= 495.0f) {
                glm::vec3 cPos, rAxis, uAxis, fAxis;
                evalMineshaftTrackFrame(sMid, cPos, rAxis, uAxis, fAxis);
                float side = (r % 2 == 0) ? -1.0f : 1.0f;
                glm::vec3 veinStart = cPos + rAxis * (side * 2.05f) + uAxis * 0.45f - fAxis * 0.8f;
                glm::vec3 veinEnd   = cPos + rAxis * (side * 1.95f) + uAxis * 2.15f + fAxis * 0.8f;
                addBeam(scene.triangles, veinStart, veinEnd, 0.28f, 0.24f, m.goldOre);
            }
        }

        // Seal far cavern end wall on the final chunk
        if (chunk == numChunks - 1) {
            glm::vec3 endPos, rAxis, uAxis, fAxis;
            evalMineshaftTrackFrame(totalTrackLength, endPos, rAxis, uAxis, fAxis);
            addBox(scene.triangles, endPos + uAxis * 2.0f, glm::vec3(10.0f, 8.0f, 0.8f), 0.0f, m.rock);
        }

        chunkBlasIndices.push_back(recordBlasPrototype("TunnelChunk_" + std::to_string(chunk), startTri));
    }

    // 2. Build Reusable Instanced Prototype BLASes (Models 1..6)
    uint32_t startCart = static_cast<uint32_t>(scene.triangles.size());
    buildMinecartModel(scene.triangles, glm::vec3(0.0f, 0.19f, 0.0f), m);
    uint32_t blasCart = recordBlasPrototype("Prototype_Minecart", startCart);

    uint32_t startLantern = static_cast<uint32_t>(scene.triangles.size());
    buildMineLanternModel(scene.triangles, glm::vec3(0.0f, 0.0f, 0.0f), m);
    uint32_t blasLantern = recordBlasPrototype("Prototype_MineLantern", startLantern);

    uint32_t startTimber = static_cast<uint32_t>(scene.triangles.size());
    buildTimberSupportModel(scene.triangles, glm::vec3(0.0f, 0.0f, 0.0f), m);
    uint32_t blasTimber = recordBlasPrototype("Prototype_TimberSupport", startTimber);

    uint32_t startTrestle = static_cast<uint32_t>(scene.triangles.size());
    buildTrestleSupportModel(scene.triangles, glm::vec3(0.0f, 0.0f, 0.0f), m, 5.35f);
    uint32_t blasTrestle = recordBlasPrototype("Prototype_TrestleSupport", startTrestle);

    uint32_t startTrack = static_cast<uint32_t>(scene.triangles.size());
    buildRailTrackSegmentModel(scene.triangles, glm::vec3(0.0f, 0.0f, 0.0f), m);
    uint32_t blasTrack = recordBlasPrototype("Prototype_RailTrack", startTrack);

    uint32_t startCrystalV = static_cast<uint32_t>(scene.triangles.size());
    buildCrystalClusterModel(scene.triangles, glm::vec3(0.0f, 0.0f, 0.0f), m, false);
    uint32_t blasCrystalV = recordBlasPrototype("Prototype_CrystalViolet", startCrystalV);

    uint32_t startCrystalC = static_cast<uint32_t>(scene.triangles.size());
    buildCrystalClusterModel(scene.triangles, glm::vec3(0.0f, 0.0f, 0.0f), m, true);
    uint32_t blasCrystalC = recordBlasPrototype("Prototype_CrystalCyan", startCrystalC);

    auto addInstance = [&](uint32_t blasIdx, const glm::mat4& xform, bool animated = false,
                           glm::vec3 pivotPos = glm::vec3(0.0f), glm::vec3 rotAxis = glm::vec3(0, 1, 0), float rotSpeed = 0.0f) -> uint32_t {
        uint32_t instIdx = static_cast<uint32_t>(scene.instances.size());
        SceneInstance si{};
        si.blasIndex = blasIdx;
        si.transform = xform;
        si.customIndex = instIdx;
        scene.instances.push_back(si);

        InstanceGPU ig{};
        ig.firstTriangle = scene.blasRanges[blasIdx].firstTriangle;
        ig.numOpaqueTriangles = scene.blasRanges[blasIdx].numOpaqueTriangles;
        ig.materialOffset = 0;
        ig.flags = 0;
        ig.transform = xform;
        ig.invTransform = glm::inverse(xform);
        scene.instanceData.push_back(ig);

        if (animated) {
            AnimatedInstance anim{};
            anim.instanceIndex = instIdx;
            anim.basePosition = pivotPos;
            anim.rotationAxis = rotAxis;
            anim.rotationSpeed = rotSpeed;
            anim.baseTransform = glm::translate(glm::mat4(1.0f), -pivotPos) * xform;
            scene.animatedInstances.push_back(anim);
        }
        return instIdx;
    };

    auto addStaticTrackLight = [&](float sTrack, glm::vec3 pos, glm::vec3 forwardDir, glm::vec3 emission, float size = 0.18f) {
        glm::vec3 f = glm::normalize(forwardDir);
        glm::vec3 up = std::abs(f.y) < 0.95f ? glm::vec3(0, 1, 0) : glm::vec3(1, 0, 0);
        glm::vec3 r = glm::normalize(glm::cross(f, up));
        glm::vec3 u = glm::normalize(glm::cross(r, f));

        LightGPU l{};
        l.position = glm::vec4(pos - r * (size * 0.5f) - u * (size * 0.5f), 0.0f);
        l.u = glm::vec4(r * size, 0.0f);
        l.v = glm::vec4(u * size, 0.0f);
        l.emission = glm::vec4(emission, size * size);
        l.normal = glm::vec4(f, 0.0f);
        scene.coasterStaticLights.push_back({ sTrack, l });
    };

    // Instantiate all 25 Static Rock Tunnel / Cavern Chunks
    for (uint32_t bChunk : chunkBlasIndices) {
        addInstance(bChunk, glm::mat4(1.0f));
    }

    // Continuous Steel Rail Track Segments along the entire 500m coaster spline
    for (float s = 0.0f; s <= totalTrackLength - 1.5f; s += 2.90f) {
        glm::vec3 pos, right, up, forward;
        evalMineshaftTrackFrame(s, pos, right, up, forward);
        addInstance(blasTrack, makeCoasterFrameMatrix(pos, right, up, forward));
    }

    // Timber Support Arches (Zones 1, 2, 4) vs Wooden Trestle Bridge Bents (Zone 3 Grand Cavern: s = 236..374m)
    int supportIdx = 0;
    for (float s = 2.0f; s <= totalTrackLength - 6.0f; s += 3.0f, ++supportIdx) {
        glm::vec3 pos, right, up, forward;
        evalMineshaftTrackFrame(s, pos, right, up, forward);
        glm::mat4 frameMat = makeCoasterFrameMatrix(pos, right, up, forward);

        bool inGrandCavern = (s >= 236.0f && s <= 374.0f);
        if (inGrandCavern) {
            // Wooden Trestle Bridge Viaduct Bent spanning the 5.35m cavern abyss!
            addInstance(blasTrestle, frameMat);

            // Every second trestle bent has a swinging lantern mounted on its outer guardrail post
            if (supportIdx % 2 == 0) {
                float side = ((supportIdx / 2) % 2 == 0) ? -1.22f : 1.22f;
                glm::vec3 hookWorld = pos + right * side + up * 1.22f;
                glm::mat4 lanternMat = glm::translate(glm::mat4(1.0f), hookWorld);
                addInstance(blasLantern, lanternMat, true, hookWorld, glm::vec3(0.0f, 1.0f, 0.0f),
                            (side < 0.0f) ? 1.1f : -1.1f);

                glm::vec3 flamePos = hookWorld - up * 0.34f - right * (side * 0.18f);
                addStaticTrackLight(s, flamePos, glm::normalize(forward - up * 0.30f - right * (side * 0.35f)),
                                    glm::vec3(46.0f, 24.0f, 7.0f), 0.22f);
            }
        } else {
            // Place Timber Support Arches every 4.5m in Zones 1, 2, and 4
            if (supportIdx % 3 != 1) {
                addInstance(blasTimber, frameMat);

                float side = (supportIdx % 2 == 0) ? -0.92f : 0.92f;
                glm::vec3 hookWorld = pos + right * side + up * 2.58f;
                float tiltDeg = (supportIdx % 2 == 0) ? 9.0f : -9.0f;
                glm::mat4 lanternMat = glm::translate(glm::mat4(1.0f), hookWorld) *
                                       glm::rotate(glm::mat4(1.0f), glm::radians(tiltDeg), forward);
                addInstance(blasLantern, lanternMat, true, hookWorld, glm::vec3(0.0f, 1.0f, 0.0f),
                            (supportIdx % 2 == 0) ? 1.25f : -1.25f);

                // Warm amber/forge lantern emitter in Zone 4, classic oil-lantern glow in Zones 1 & 2
                glm::vec3 lanternColor = (s >= 375.0f) ? glm::vec3(62.0f, 34.0f, 8.0f) : glm::vec3(44.0f, 22.0f, 6.0f);
                glm::vec3 flamePos = hookWorld - up * 0.34f - right * (side * 0.24f);
                addStaticTrackLight(s, flamePos, glm::normalize(forward * 0.5f - up * 0.5f - right * (side * 0.35f)),
                                    lanternColor, 0.20f);
            }
        }
    }

    // Subterranean Bioluminescent Crystal Clusters along the 500m ride (extra-dense & colossal in Zone 3 Grand Cavern!)
    for (int c = 0; c < 110; ++c) {
        float u = static_cast<float>(c) / 109.0f;
        // Concentrate crystal clusters in Zone 2, Zone 3 (Grand Cavern), and Zone 4
        float s = glm::mix(14.0f, 490.0f, u);
        glm::vec3 pos, right, up, forward;
        evalMineshaftTrackFrame(s, pos, right, up, forward);

        bool inCavern = (s >= 232.0f && s <= 378.0f);
        float side = (c % 2 == 0) ? -1.0f : 1.0f;
        bool isCyan = ((c % 3) != 0);

        float wallDist = inCavern ? (3.2f + 4.2f * std::abs(std::sin(static_cast<float>(c) * 1.9f)))
                                  : (1.78f + 0.25f * std::abs(std::sin(static_cast<float>(c) * 2.3f)));
        float heightOff = inCavern ? (-3.8f + 5.6f * std::sin(static_cast<float>(c) * 1.37f))
                                   : (0.12f + 0.45f * std::abs(std::sin(static_cast<float>(c) * 1.7f)));
        float scale = inCavern ? (1.65f + 1.15f * std::abs(std::cos(static_cast<float>(c) * 1.1f)))
                               : (0.75f + 0.25f * std::abs(std::sin(static_cast<float>(c))));

        glm::vec3 clusterPos = pos + right * (side * wallDist) + up * heightOff;
        glm::vec3 clusterUp = glm::normalize(up * 0.74f - right * (side * 0.66f));
        glm::vec3 clusterRight = glm::normalize(glm::cross(forward, clusterUp));
        glm::vec3 clusterFwd = glm::normalize(glm::cross(clusterRight, clusterUp));

        glm::mat4 cMat = makeCoasterFrameMatrix(clusterPos, clusterRight * scale, clusterUp * scale, clusterFwd * scale);
        addInstance(isCyan ? blasCrystalC : blasCrystalV, cMat);

        if (inCavern || (c % 3 == 0)) {
            glm::vec3 cLightColor = isCyan ? glm::vec3(7.0f, 34.0f, 44.0f) : glm::vec3(30.0f, 8.0f, 44.0f);
            if (inCavern) cLightColor *= 1.65f;
            addStaticTrackLight(s, clusterPos + clusterUp * (1.16f * scale), clusterUp, cLightColor, 0.26f);
        }
    }

    // 3. Instantiate the 3 Animated Coaster Minecarts (Rider Cart + Lead Cart #1 + Lead Cart #2)
    glm::mat4 riderMat(1.0f), lead1Mat(1.0f), lead2Mat(1.0f);
    scene.hasCamera = true;
    updateMineshaftCoaster(scene, 0.0f,
                           scene.cameraPosition, scene.cameraTarget, scene.cameraUp, scene.cameraFov,
                           riderMat, lead1Mat, lead2Mat);

    scene.coasterRiderCartInst = addInstance(blasCart, riderMat);
    scene.coasterLeadCart1Inst = addInstance(blasCart, lead1Mat);
    scene.coasterLeadCart2Inst = addInstance(blasCart, lead2Mat);

    // Ensure animatedInstances is non-empty so Engine::renderFrame enables isAnimating
    scene.boundsMin = glm::vec3(-35.0f, -68.0f, -500.0f);
    scene.boundsMax = glm::vec3( 35.0f,   8.0f,   20.0f);
    scene.sceneRadius = 260.0f;
    scene.focalBoundsMin = glm::vec3(-12.0f, -15.0f, -45.0f);
    scene.focalBoundsMax = glm::vec3( 12.0f,   6.0f,  15.0f);
    scene.focalRadius = 18.0f;
    scene.focalDistance = glm::length(scene.cameraTarget - scene.cameraPosition);
    scene.centralTarget = scene.cameraTarget;

    scene.hasDielectrics = true;
    scene.dielectricBoundsMin = glm::vec3(-25.0f, -65.0f, -490.0f);
    scene.dielectricBoundsMax = glm::vec3( 25.0f,   6.0f,   16.0f);

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
