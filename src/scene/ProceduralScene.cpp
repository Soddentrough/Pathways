#include "scene/ProceduralScene.hpp"
#include <cmath>
#include <algorithm>
#include <glm/gtc/matrix_transform.hpp>

namespace pathways {

static void addQuad(std::vector<TriangleGPU>& triangles,
                    glm::vec3 p0, glm::vec3 p1, glm::vec3 p2, glm::vec3 p3,
                    glm::vec3 normal, uint32_t matId) {
    glm::vec3 up = std::abs(normal.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
    glm::vec3 tanDir = glm::normalize(glm::cross(up, normal));
    glm::vec4 tangentVec = glm::vec4(tanDir, 1.0f);

    TriangleGPU t1{};
    t1.v0.position = glm::vec4(p0, 0.0f);
    t1.v0.normal = glm::vec4(normal, 0.0f);
    t1.v0.tangent = tangentVec;
    t1.v1.position = glm::vec4(p1, 1.0f);
    t1.v1.normal = glm::vec4(normal, 0.0f);
    t1.v1.tangent = tangentVec;
    t1.v2.position = glm::vec4(p2, 1.0f);
    t1.v2.normal = glm::vec4(normal, 1.0f);
    t1.v2.tangent = tangentVec;
    t1.materialId = matId;

    TriangleGPU t2{};
    t2.v0.position = glm::vec4(p0, 0.0f);
    t2.v0.normal = glm::vec4(normal, 0.0f);
    t2.v0.tangent = tangentVec;
    t2.v1.position = glm::vec4(p2, 1.0f);
    t2.v1.normal = glm::vec4(normal, 1.0f);
    t2.v1.tangent = tangentVec;
    t2.v2.position = glm::vec4(p3, 0.0f);
    t2.v2.normal = glm::vec4(normal, 1.0f);
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

    // 4: Wet Reflective Plaza Pavement (Diffuse + High Clearcoat)
    MaterialGPU mat4{};
    mat4.albedo = glm::vec4(0.06f, 0.06f, 0.08f, 1.0f);
    mat4.roughness = 0.16f;
    mat4.metallic = 0.10f;
    mat4.clearcoat = 0.98f;
    mat4.clearcoatRoughness = 0.02f;
    mat4.type = MATERIAL_DIFFUSE;
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
    // 20: Carbon Fiber Weave (Metallic + Clearcoat)
    MaterialGPU mat20{};
    mat20.albedo = glm::vec4(0.12f, 0.12f, 0.13f, 1.0f);
    mat20.roughness = 0.38f;
    mat20.metallic = 0.40f;
    mat20.clearcoat = 0.60f;
    mat20.clearcoatRoughness = 0.10f;
    mat20.type = MATERIAL_METALLIC;
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
    // 30: Neon Cyan 480nm
    MaterialGPU mat30{};
    mat30.albedo = glm::vec4(0.1f, 0.8f, 1.0f, 1.0f);
    mat30.emissive = glm::vec4(2.0f, 32.0f, 42.0f, 1.0f);
    mat30.type = MATERIAL_EMISSIVE;
    materials.push_back(mat30);

    // 31: Neon Magenta 650nm
    MaterialGPU mat31{};
    mat31.albedo = glm::vec4(1.0f, 0.1f, 0.6f, 1.0f);
    mat31.emissive = glm::vec4(42.0f, 2.0f, 22.0f, 1.0f);
    mat31.type = MATERIAL_EMISSIVE;
    materials.push_back(mat31);

    // 32: Neon Blaze Orange 600nm
    MaterialGPU mat32{};
    mat32.albedo = glm::vec4(1.0f, 0.4f, 0.05f, 1.0f);
    mat32.emissive = glm::vec4(45.0f, 15.0f, 1.5f, 1.0f);
    mat32.type = MATERIAL_EMISSIVE;
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

    // 36: Deep Violet Holo-Display 405nm
    MaterialGPU mat36{};
    mat36.albedo = glm::vec4(0.7f, 0.1f, 1.0f, 1.0f);
    mat36.emissive = glm::vec4(25.0f, 2.0f, 45.0f, 1.0f);
    mat36.type = MATERIAL_EMISSIVE;
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
    mat38.type = MATERIAL_EMISSIVE;
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
    CYBER_BLAS_COUNT = 19
};

static void addWindowGrid(std::vector<TriangleGPU>& triangles,
                          glm::vec3 origin, glm::vec3 uAxis, glm::vec3 vAxis,
                          int rows, int cols, float spanU, float spanV,
                          uint32_t mullionMat, uint32_t glassMat, uint32_t emissiveMat,
                          uint32_t seed) {
    float stepU = spanU / static_cast<float>(cols);
    float stepV = spanV / static_cast<float>(rows);
    glm::vec3 normal = glm::normalize(glm::cross(uAxis, vAxis));

    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            glm::vec3 p = origin + uAxis * ((static_cast<float>(c) + 0.5f) * stepU)
                                 + vAxis * ((static_cast<float>(r) + 0.5f) * stepV);
            // Window mullion frame
            addBox(triangles, p, glm::vec3(stepU * 0.94f, stepV * 0.94f, 0.12f), 0.0f, mullionMat);

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
        }
    }
}

static void addDiagridLattice(std::vector<TriangleGPU>& triangles,
                              glm::vec3 origin, glm::vec3 uAxis, glm::vec3 vAxis,
                              int segments, float spanU, float spanV,
                              uint32_t trussMat) {
    float stepU = spanU / static_cast<float>(segments);
    for (int s = 0; s < segments; ++s) {
        glm::vec3 b0 = origin + uAxis * (static_cast<float>(s) * stepU);
        glm::vec3 b1 = origin + uAxis * (static_cast<float>(s + 1) * stepU);
        glm::vec3 t0 = b0 + vAxis * spanV;
        glm::vec3 t1 = b1 + vAxis * spanV;
        addBox(triangles, (b0 + t1) * 0.5f, glm::vec3(stepU * 0.10f, spanV * 1.02f, 0.14f), 45.0f, trussMat);
        addBox(triangles, (b1 + t0) * 0.5f, glm::vec3(stepU * 0.10f, spanV * 1.02f, 0.14f), -45.0f, trussMat);
    }
}

SceneData ProceduralScene::createCyberCityScene() {
    SceneData scene;
    scene.materials = createCyberCityMaterials();

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
        // Basalt plinth base step (Mat 20: Rough Basalt Curb)
        addBox(scene.triangles, glm::vec3(0.0f, 0.2f, 0.0f), glm::vec3(13.6f, 0.4f, 13.6f), 0.0f, 20);
        // 4 heavy corner columns (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(-5.2f, 4.0f, -5.2f), glm::vec3(2.0f, 8.0f, 2.0f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3( 5.2f, 4.0f, -5.2f), glm::vec3(2.0f, 8.0f, 2.0f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3(-5.2f, 4.0f,  5.2f), glm::vec3(2.0f, 8.0f, 2.0f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3( 5.2f, 4.0f,  5.2f), glm::vec3(2.0f, 8.0f, 2.0f), 0.0f, 1);
        // Diagrid corner trusses (Mat 27: Galvanized Steel)
        addDiagridLattice(scene.triangles, glm::vec3(-5.8f, 0.4f, 5.8f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 3, 2.0f, 7.2f, 27);
        addDiagridLattice(scene.triangles, glm::vec3( 3.8f, 0.4f, 5.8f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 3, 2.0f, 7.2f, 27);
        // Entrance portals on all 4 facades (Mat 3: Brushed Brass, Mat 29: Polished Jade trim)
        addBox(scene.triangles, glm::vec3( 0.0f, 2.7f,  6.1f), glm::vec3(4.5f, 4.6f, 0.4f), 0.0f, 3);
        addBox(scene.triangles, glm::vec3( 0.0f, 2.7f, -6.1f), glm::vec3(4.5f, 4.6f, 0.4f), 0.0f, 3);
        addBox(scene.triangles, glm::vec3( 6.1f, 2.7f,  0.0f), glm::vec3(0.4f, 4.6f, 4.5f), 0.0f, 3);
        addBox(scene.triangles, glm::vec3(-6.1f, 2.7f,  0.0f), glm::vec3(0.4f, 4.6f, 4.5f), 0.0f, 3);
        addBox(scene.triangles, glm::vec3( 0.0f, 4.8f,  6.15f), glm::vec3(4.8f, 0.3f, 0.2f), 0.0f, 29);
        addBox(scene.triangles, glm::vec3( 0.0f, 4.8f, -6.15f), glm::vec3(4.8f, 0.3f, 0.2f), 0.0f, 29);
        addBox(scene.triangles, glm::vec3( 6.15f, 4.8f,  0.0f), glm::vec3(0.2f, 0.3f, 4.8f), 0.0f, 29);
        addBox(scene.triangles, glm::vec3(-6.15f, 4.8f,  0.0f), glm::vec3(0.2f, 0.3f, 4.8f), 0.0f, 29);
        // Lower facade window grids: 4 facades x (3 rows x 4 cols = 12 windows)
        // (Mat 1: Steel framing, Mat 7: Smoked Obsidian glass, Mat 41: Warm Sodium interior glow)
        addWindowGrid(scene.triangles, glm::vec3(-4.6f, 1.2f,  6.08f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 3, 4, 9.2f, 4.8f, 1, 7, 41, 101);
        addWindowGrid(scene.triangles, glm::vec3(-4.6f, 1.2f, -6.08f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 3, 4, 9.2f, 4.8f, 1, 7, 41, 103);
        addWindowGrid(scene.triangles, glm::vec3( 6.08f, 1.2f, -4.6f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 3, 4, 9.2f, 4.8f, 1, 7, 41, 107);
        addWindowGrid(scene.triangles, glm::vec3(-6.08f, 1.2f, -4.6f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 3, 4, 9.2f, 4.8f, 1, 7, 41, 109);
        // Emissive entrance beacon strips (Mat 30: Neon Cyan 480nm, Mat 35: Electric Gold)
        addBox(scene.triangles, glm::vec3( 0.0f, 2.36f,  6.25f), glm::vec3(3.2f, 3.88f, 0.12f), 0.0f, 30);
        addBox(scene.triangles, glm::vec3( 0.0f, 2.36f, -6.25f), glm::vec3(3.2f, 3.88f, 0.12f), 0.0f, 30);
        addBox(scene.triangles, glm::vec3( 6.25f, 2.36f,  0.0f), glm::vec3(0.12f, 3.88f, 3.2f), 0.0f, 30);
        addBox(scene.triangles, glm::vec3(-6.25f, 2.36f,  0.0f), glm::vec3(0.12f, 3.88f, 3.2f), 0.0f, 30);
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
        // Vertical neon circuit conduits (Mat 35: Electric Gold)
        addBox(scene.triangles, glm::vec3(-4.95f, 3.0f, -4.95f), glm::vec3(0.2f, 6.0f, 0.2f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3( 4.95f, 3.0f,  4.95f), glm::vec3(0.2f, 6.0f, 0.2f), 0.0f, 35);
        // Dense Curtain Wall Window Grids on 4 facades (5 rows x 4 cols = 20 windows per facade)
        // (Mat 16: Anisotropic Platinum mullions, Mat 12: Iridescent Coated Glass, Mat 30: Neon Cyan office glow)
        addWindowGrid(scene.triangles, glm::vec3(-4.5f, 0.5f,  4.98f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 5, 4, 9.0f, 5.0f, 16, 12, 30, 201);
        addWindowGrid(scene.triangles, glm::vec3(-4.5f, 0.5f, -4.98f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 5, 4, 9.0f, 5.0f, 16, 12, 30, 203);
        addWindowGrid(scene.triangles, glm::vec3( 4.98f, 0.5f, -4.5f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 5, 4, 9.0f, 5.0f, 16, 12, 30, 205);
        addWindowGrid(scene.triangles, glm::vec3(-4.98f, 0.5f, -4.5f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 5, 4, 9.0f, 5.0f, 16, 12, 30, 207);
        recordBlasPrototype("Tower Mid Module A", tStart);
    }

    // Prototype 2: Tower Mid Module B (Terraced Garden & Exoskeleton)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Cantilever terrace slab (Mat 21: High-Gloss Ceramic, Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 0.5f, 0.0f), glm::vec3(11.8f, 1.0f, 11.8f), 0.0f, 21);
        addBox(scene.triangles, glm::vec3(0.0f, 0.05f, 0.0f), glm::vec3(11.9f, 0.1f, 11.9f), 0.0f, 1);
        // Recessed core (Mat 13: Anodized Gunmetal Aluminum)
        addBox(scene.triangles, glm::vec3(0.0f, 3.5f, 0.0f), glm::vec3(8.8f, 5.0f, 8.8f), 0.0f, 13);
        // 4 Terrace structural pillars (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(-5.2f, 3.0f, -5.2f), glm::vec3(0.8f, 5.0f, 0.8f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3( 5.2f, 3.0f, -5.2f), glm::vec3(0.8f, 5.0f, 0.8f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3(-5.2f, 3.0f,  5.2f), glm::vec3(0.8f, 5.0f, 0.8f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3( 5.2f, 3.0f,  5.2f), glm::vec3(0.8f, 5.0f, 0.8f), 0.0f, 2);
        // Balustrade glass railings (Mat 6: Cyan Glass) & Rose Gold trim (Mat 15)
        addBox(scene.triangles, glm::vec3( 0.0f, 1.6f,  5.85f), glm::vec3(11.4f, 1.2f, 0.08f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3( 0.0f, 1.6f, -5.85f), glm::vec3(11.4f, 1.2f, 0.08f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3( 5.85f, 1.6f,  0.0f), glm::vec3(0.08f, 1.2f, 11.4f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3(-5.85f, 1.6f,  0.0f), glm::vec3(0.08f, 1.2f, 11.4f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3( 0.0f, 2.22f,  5.85f), glm::vec3(11.5f, 0.06f, 0.12f), 0.0f, 15);
        addBox(scene.triangles, glm::vec3( 0.0f, 2.22f, -5.85f), glm::vec3(11.5f, 0.06f, 0.12f), 0.0f, 15);
        addBox(scene.triangles, glm::vec3( 5.85f, 2.22f,  0.0f), glm::vec3(0.12f, 0.06f, 11.5f), 0.0f, 15);
        addBox(scene.triangles, glm::vec3(-5.85f, 2.22f,  0.0f), glm::vec3(0.12f, 0.06f, 11.5f), 0.0f, 15);
        // Frosted Amber privacy screens (Mat 8)
        addBox(scene.triangles, glm::vec3(-4.0f, 2.8f,  4.6f), glm::vec3(1.8f, 3.2f, 0.06f), 0.0f, 8);
        addBox(scene.triangles, glm::vec3( 4.0f, 2.8f, -4.6f), glm::vec3(1.8f, 3.2f, 0.06f), 0.0f, 8);
        // Diagrid Exoskeleton Trusses on East & West facades (Mat 27: Galvanized Steel)
        addDiagridLattice(scene.triangles, glm::vec3(-4.2f, 1.0f,  4.5f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 8.4f, 4.8f, 27);
        addDiagridLattice(scene.triangles, glm::vec3(-4.2f, 1.0f, -4.5f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 6, 8.4f, 4.8f, 27);
        // Window grids on North & South recessed core (5 rows x 4 cols)
        addWindowGrid(scene.triangles, glm::vec3( 4.45f, 1.2f, -4.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 5, 4, 8.0f, 4.5f, 1, 6, 31, 301);
        addWindowGrid(scene.triangles, glm::vec3(-4.45f, 1.2f, -4.0f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 5, 4, 8.0f, 4.5f, 1, 6, 32, 303);
        // Neon perimeter rim strips (Mat 31: Magenta, Mat 32: Orange)
        addBox(scene.triangles, glm::vec3(0.0f, 0.8f,  5.95f), glm::vec3(11.6f, 0.2f, 0.1f), 0.0f, 31);
        addBox(scene.triangles, glm::vec3(0.0f, 0.8f, -5.95f), glm::vec3(11.6f, 0.2f, 0.1f), 0.0f, 32);
        recordBlasPrototype("Tower Mid Module B", tStart);
    }

    // Prototype 3: Tower Mid Module C (High-Tech Industrial Core)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Heavy carbon fiber sheathing core (Mat 19: Carbon Fiber Polymer)
        addBox(scene.triangles, glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(10.2f, 5.8f, 10.2f), 0.0f, 19);
        // Floor collar (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 0.2f, 0.0f), glm::vec3(10.8f, 0.4f, 10.8f), 0.0f, 1);
        // Exposed acoustic/ventilation mesh grilles (Mat 28: Perforated Mesh)
        addBox(scene.triangles, glm::vec3( 0.0f, 4.5f,  5.18f), glm::vec3(7.2f, 1.8f, 0.12f), 0.0f, 28);
        addBox(scene.triangles, glm::vec3( 0.0f, 4.5f, -5.18f), glm::vec3(7.2f, 1.8f, 0.12f), 0.0f, 28);
        addBox(scene.triangles, glm::vec3( 5.18f, 4.5f,  0.0f), glm::vec3(0.12f, 1.8f, 7.2f), 0.0f, 28);
        addBox(scene.triangles, glm::vec3(-5.18f, 4.5f,  0.0f), glm::vec3(0.12f, 1.8f, 7.2f), 0.0f, 28);
        // High-voltage copper busbars (Mat 14: Polished Copper)
        addBox(scene.triangles, glm::vec3(-4.8f, 3.0f,  5.22f), glm::vec3(0.25f, 5.8f, 0.15f), 0.0f, 14);
        addBox(scene.triangles, glm::vec3( 4.8f, 3.0f,  5.22f), glm::vec3(0.25f, 5.8f, 0.15f), 0.0f, 14);
        addBox(scene.triangles, glm::vec3(-4.8f, 3.0f, -5.22f), glm::vec3(0.25f, 5.8f, 0.15f), 0.0f, 14);
        addBox(scene.triangles, glm::vec3( 4.8f, 3.0f, -5.22f), glm::vec3(0.25f, 5.8f, 0.15f), 0.0f, 14);
        // Server window banks (5 rows x 4 cols): Mat 17 (Chromium), Mat 7 (Obsidian), Mat 33 (Acid Green status)
        addWindowGrid(scene.triangles, glm::vec3(-4.2f, 0.8f,  5.15f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 5, 4, 8.4f, 3.2f, 17, 7, 33, 401);
        addWindowGrid(scene.triangles, glm::vec3(-4.2f, 0.8f, -5.15f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 5, 4, 8.4f, 3.2f, 17, 7, 33, 403);
        addWindowGrid(scene.triangles, glm::vec3( 5.15f, 0.8f, -4.2f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 5, 4, 8.4f, 3.2f, 17, 7, 33, 405);
        addWindowGrid(scene.triangles, glm::vec3(-5.15f, 0.8f, -4.2f), glm::vec3(0.0f, 0.0f, 1.0f), glm::vec3(0.0f, 1.0f, 0.0f), 5, 4, 8.4f, 3.2f, 17, 7, 33, 407);
        // Crimson strobe safety beacon (Mat 44)
        addBox(scene.triangles, glm::vec3(5.15f, 5.6f, 5.15f), glm::vec3(0.3f, 0.3f, 0.3f), 0.0f, 44);
        addBox(scene.triangles, glm::vec3(-5.15f, 5.6f, -5.15f), glm::vec3(0.3f, 0.3f, 0.3f), 0.0f, 44);
        recordBlasPrototype("Tower Mid Module C", tStart);
    }

    // Prototype 4: Tower Crown & Spire
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Penthouse roof base (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(9.6f, 2.0f, 9.6f), 0.0f, 1);
        // Angular pyramid frustum (Mat 16: Anisotropic Brushed Platinum)
        addFrustum(scene.triangles, glm::vec3(0.0f, 3.0f, 0.0f), glm::vec2(9.6f), glm::vec2(5.4f), 2.0f, 16);
        // Mezzanine deck (Mat 3: Brushed Brass)
        addBox(scene.triangles, glm::vec3(0.0f, 4.2f, 0.0f), glm::vec3(5.6f, 0.4f, 5.6f), 0.0f, 3);
        // Geodesic observation sphere (Mat 11: High-Dispersion Optical Diamond Glass)
        addSphere(scene.triangles, glm::vec3(0.0f, 6.2f, 0.0f), 2.4f, 11, 28, 28);
        // Emerald crystal finials (Mat 9: Emerald Crystal)
        addSphere(scene.triangles, glm::vec3(-2.2f, 4.6f, -2.2f), 0.4f, 9, 14, 14);
        addSphere(scene.triangles, glm::vec3( 2.2f, 4.6f, -2.2f), 0.4f, 9, 14, 14);
        addSphere(scene.triangles, glm::vec3(-2.2f, 4.6f,  2.2f), 0.4f, 9, 14, 14);
        addSphere(scene.triangles, glm::vec3( 2.2f, 4.6f,  2.2f), 0.4f, 9, 14, 14);
        // Interior pedestal (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(0.0f, 5.2f, 0.0f), glm::vec3(0.8f, 1.0f, 0.8f), 0.0f, 2);
        // Antenna mast (Mat 2: Titanium Chrome)
        addCylinder(scene.triangles, glm::vec3(0.0f, 12.0f, 0.0f), 0.18f, 7.8f, 2, 16);
        // Cross arms (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 11.0f, 0.0f), glm::vec3(2.4f, 0.15f, 0.15f), 0.0f, 1);
        addBox(scene.triangles, glm::vec3(0.0f, 13.5f, 0.0f), glm::vec3(1.6f, 0.15f, 0.15f), 0.0f, 1);
        // Xenon floodlight fixtures (Mat 42: Xenon White)
        addBox(scene.triangles, glm::vec3( 1.1f, 10.9f, 0.0f), glm::vec3(0.2f, 0.2f, 0.2f), 0.0f, 42);
        addBox(scene.triangles, glm::vec3(-1.1f, 10.9f, 0.0f), glm::vec3(0.2f, 0.2f, 0.2f), 0.0f, 42);
        // Ruby laser dielectric focusing core (Mat 10: Ruby Laser Crystal)
        addSphere(scene.triangles, glm::vec3(0.0f, 16.0f, 0.0f), 0.35f, 10, 14, 14);
        // Ruby laser beacon tip (Mat 34: Ruby Laser Warning Beacon)
        addBox(scene.triangles, glm::vec3(0.0f, 16.4f, 0.0f), glm::vec3(0.3f, 0.3f, 0.3f), 0.0f, 34);
        recordBlasPrototype("Tower Crown & Spire", tStart);
    }

    // Prototype 5: Rooftop HVAC & Industrial Machinery Pod
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Corten steel mounting skid (Mat 18: Aged Rusted Corten Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 0.4f, 0.0f), glm::vec3(8.0f, 0.8f, 6.0f), 0.0f, 18);
        // Dual large chiller cylinders (Mat 1: Carbon Steel)
        addCylinder(scene.triangles, glm::vec3(-2.2f, 2.2f, 0.0f), 1.4f, 3.2f, 1, 16);
        addCylinder(scene.triangles, glm::vec3( 2.2f, 2.2f, 0.0f), 1.4f, 3.2f, 1, 16);
        // Perforated ventilation exhaust intakes (Mat 28: Perforated Mesh)
        addBox(scene.triangles, glm::vec3(-2.2f, 3.9f, 0.0f), glm::vec3(2.4f, 0.3f, 2.4f), 0.0f, 28);
        addBox(scene.triangles, glm::vec3( 2.2f, 3.9f, 0.0f), glm::vec3(2.4f, 0.3f, 2.4f), 0.0f, 28);
        // Polished copper pipe manifolds (Mat 14: Polished Copper)
        addCylinder(scene.triangles, glm::vec3(0.0f, 2.0f, -1.8f), 0.25f, 5.2f, 14, 12);
        addCylinder(scene.triangles, glm::vec3(0.0f, 2.6f, -1.8f), 0.25f, 5.2f, 14, 12);
        // Gunmetal ductwork housing (Mat 13: Anodized Gunmetal)
        addBox(scene.triangles, glm::vec3(0.0f, 1.8f, 1.8f), glm::vec3(3.2f, 2.2f, 1.8f), 0.0f, 13);
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
        // Diagrid truss sides along both flanks (Mat 27: Galvanized Steel)
        addDiagridLattice(scene.triangles, glm::vec3(-7.2f, 0.2f,  1.74f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 8, 14.4f, 2.8f, 27);
        addDiagridLattice(scene.triangles, glm::vec3(-7.2f, 0.2f, -1.74f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 8, 14.4f, 2.8f, 27);
        // Floor and ceiling structural crown glass (Mat 5: Transmissive Crown Glass)
        addBox(scene.triangles, glm::vec3(0.0f, 0.20f, 0.0f), glm::vec3(15.0f, 0.08f, 3.2f), 0.0f, 5);
        addBox(scene.triangles, glm::vec3(0.0f, 3.00f, 0.0f), glm::vec3(15.0f, 0.08f, 3.2f), 0.0f, 5);
        // Sidewall transmissive panels (Mat 6: Cyan Canopy Glass)
        addBox(scene.triangles, glm::vec3(0.0f, 1.60f,  1.70f), glm::vec3(15.0f, 2.6f, 0.08f), 0.0f, 6);
        addBox(scene.triangles, glm::vec3(0.0f, 1.60f, -1.70f), glm::vec3(15.0f, 2.6f, 0.08f), 0.0f, 6);
        // Interior walkway lounge carpet (Mat 23: Velvet Sheen)
        addBox(scene.triangles, glm::vec3(0.0f, 0.25f, 0.0f), glm::vec3(14.8f, 0.02f, 1.8f), 0.0f, 23);
        // Interior neon guide strips (Mat 30: Neon Cyan)
        addBox(scene.triangles, glm::vec3(0.0f, 0.26f, -0.9f), glm::vec3(14.8f, 0.04f, 0.15f), 0.0f, 30);
        addBox(scene.triangles, glm::vec3(0.0f, 0.26f,  0.9f), glm::vec3(14.8f, 0.04f, 0.15f), 0.0f, 30);
        // Under-bridge electric turquoise display glow strip (Mat 45: Electric Turquoise)
        addBox(scene.triangles, glm::vec3(0.0f, -0.05f, 0.0f), glm::vec3(15.0f, 0.10f, 0.40f), 0.0f, 45);
        recordBlasPrototype("Glass Skybridge", tStart);
    }

    // Prototype 7: Comm Gantry & Microwave Relay (CYBER_BLAS_COMM_GANTRY)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Structural mast (Mat 1: Carbon Steel)
        addBox(scene.triangles, glm::vec3(0.0f, 4.0f, 0.0f), glm::vec3(1.6f, 8.0f, 1.6f), 0.0f, 1);
        // Diagrid lattice bracing (Mat 27: Galvanized Steel)
        addDiagridLattice(scene.triangles, glm::vec3(-0.9f, 0.0f, 0.9f), glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), 2, 1.8f, 7.8f, 27);
        // Flange platforms (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(0.0f, 3.0f, 0.0f), glm::vec3(2.4f, 0.2f, 2.4f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3(0.0f, 6.0f, 0.0f), glm::vec3(2.0f, 0.2f, 2.0f), 0.0f, 2);
        // Microwave receiver dishes (Mat 17: Dark Black Chromium)
        addCylinder(scene.triangles, glm::vec3(0.0f, 6.5f, 1.0f), 0.9f, 0.3f, 17, 16);
        addCylinder(scene.triangles, glm::vec3(1.0f, 4.5f, 0.0f), 0.7f, 0.3f, 17, 16);
        // Gold plated transceivers (Mat 3: Brushed Brass)
        addBox(scene.triangles, glm::vec3(0.0f, 6.5f, 1.3f), glm::vec3(0.18f, 0.18f, 0.4f), 0.0f, 3);
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
        // High-traction rubber guideway bed (Mat 24: Rubber Guideway Bed)
        addBox(scene.triangles, glm::vec3(0.0f, 0.62f, 0.0f), glm::vec3(16.0f, 0.06f, 4.6f), 0.0f, 24);
        // Porous asphalt apron shoulders (Mat 25: Sound-Absorbing Porous Asphalt)
        addBox(scene.triangles, glm::vec3(0.0f, 0.64f, -2.0f), glm::vec3(16.0f, 0.04f, 0.6f), 0.0f, 25);
        addBox(scene.triangles, glm::vec3(0.0f, 0.64f,  2.0f), glm::vec3(16.0f, 0.04f, 0.6f), 0.0f, 25);
        // Dual electromagnetic levitation rails (Mat 35: Electric Gold Emissive Conduit)
        addBox(scene.triangles, glm::vec3(0.0f, 0.72f, -1.1f), glm::vec3(16.0f, 0.12f, 0.4f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3(0.0f, 0.72f,  1.1f), glm::vec3(16.0f, 0.12f, 0.4f), 0.0f, 35);
        // Center copper power rail (Mat 14: Polished Copper)
        addBox(scene.triangles, glm::vec3(0.0f, 0.68f, 0.0f), glm::vec3(16.0f, 0.08f, 0.3f), 0.0f, 14);
        // Titanium aerodynamic crash barriers (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(0.0f, 0.95f, -2.35f), glm::vec3(16.0f, 0.8f, 0.2f), 0.0f, 2);
        addBox(scene.triangles, glm::vec3(0.0f, 0.95f,  2.35f), glm::vec3(16.0f, 0.8f, 0.2f), 0.0f, 2);
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
        // Carbon fiber composite chassis (Mat 19: Carbon Fiber Polymer)
        addBox(scene.triangles, glm::vec3(0.0f, 0.4f, 0.0f), glm::vec3(4.2f, 0.7f, 2.0f), 0.0f, 19);
        // Titanium aerodynamic canards and winglets (Mat 2: Titanium Chrome)
        addBox(scene.triangles, glm::vec3(-0.4f, 0.38f, 0.0f), glm::vec3(2.4f, 0.12f, 3.4f), 0.0f, 2);
        // Smoked obsidian glass teardrop cockpit canopy (Mat 7: Smoked Obsidian Glass)
        addSphere(scene.triangles, glm::vec3(0.6f, 0.85f, 0.0f), 0.82f, 7, 20, 20);
        // Gunmetal VTOL thruster nacelles (Mat 13: Anodized Gunmetal)
        addCylinder(scene.triangles, glm::vec3(-1.8f, 0.45f, -1.1f), 0.38f, 0.8f, 13, 14);
        addCylinder(scene.triangles, glm::vec3(-1.8f, 0.45f,  1.1f), 0.38f, 0.8f, 13, 14);
        // Cyan plasma thruster rings (Mat 30: Neon Cyan)
        addBox(scene.triangles, glm::vec3(-2.22f, 0.45f, -1.1f), glm::vec3(0.08f, 0.42f, 0.42f), 0.0f, 30);
        addBox(scene.triangles, glm::vec3(-2.22f, 0.45f,  1.1f), glm::vec3(0.08f, 0.42f, 0.42f), 0.0f, 30);
        // Ice blue forward transit headlights (Mat 47: Ice Blue Transit Headlight)
        addBox(scene.triangles, glm::vec3(2.05f, 0.38f, -0.65f), glm::vec3(0.18f, 0.18f, 0.25f), 0.0f, 47);
        addBox(scene.triangles, glm::vec3(2.05f, 0.38f,  0.65f), glm::vec3(0.18f, 0.18f, 0.25f), 0.0f, 47);
        // Ruby laser tail safety beacons (Mat 34: Ruby Laser Beacon)
        addBox(scene.triangles, glm::vec3(-2.05f, 0.55f, 0.0f), glm::vec3(0.15f, 0.25f, 0.6f), 0.0f, 34);
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
        // Violet secondary accent border (Mat 36: Deep Violet)
        addBox(scene.triangles, glm::vec3(0.0f, 0.0f, 0.08f), glm::vec3(7.9f, 4.4f, 0.06f), 0.0f, 36);
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
        // Mint phosphor status indicator lights (Mat 39: Mint Phosphor Luminescence)
        addBox(scene.triangles, glm::vec3(0.0f, 0.6f, 0.52f), glm::vec3(0.6f, 0.12f, 0.04f), 0.0f, 39);
        recordBlasPrototype("Solar Roof", tStart);
    }

    // Prototype 15: Ground District Plaza Tile 16m x 16m (CYBER_BLAS_PLAZA_DISTRICT)
    {
        uint32_t tStart = static_cast<uint32_t>(scene.triangles.size());
        // Foundation concrete bed (Mat 0: Foundation Concrete) with subsurface overlap
        addBox(scene.triangles, glm::vec3(0.0f, -0.35f, 0.0f), glm::vec3(16.2f, 0.7f, 16.2f), 0.0f, 0);
        // Wet reflective plaza pavement surface (Mat 4: Wet Reflective Pavement)
        addBox(scene.triangles, glm::vec3(0.0f, 0.02f, 0.0f), glm::vec3(16.0f, 0.04f, 16.0f), 0.0f, 4);
        // Rough basalt curb borders (Mat 20: Rough Basalt)
        addBox(scene.triangles, glm::vec3( 7.9f, 0.08f, 0.0f), glm::vec3(0.2f, 0.16f, 16.0f), 0.0f, 20);
        addBox(scene.triangles, glm::vec3(-7.9f, 0.08f, 0.0f), glm::vec3(0.2f, 0.16f, 16.0f), 0.0f, 20);
        addBox(scene.triangles, glm::vec3(0.0f, 0.08f,  7.9f), glm::vec3(16.0f, 0.16f, 0.2f), 0.0f, 20);
        addBox(scene.triangles, glm::vec3(0.0f, 0.08f, -7.9f), glm::vec3(16.0f, 0.16f, 0.2f), 0.0f, 20);
        // Glazed ceramic planter boxes (Mat 21: White Ceramic) with Jade trim (Mat 29)
        addBox(scene.triangles, glm::vec3(-4.5f, 0.4f, -4.5f), glm::vec3(2.4f, 0.8f, 2.4f), 0.0f, 21);
        addBox(scene.triangles, glm::vec3(-4.5f, 0.82f, -4.5f), glm::vec3(2.5f, 0.08f, 2.5f), 0.0f, 29);
        addBox(scene.triangles, glm::vec3( 4.5f, 0.4f,  4.5f), glm::vec3(2.4f, 0.8f, 2.4f), 0.0f, 21);
        addBox(scene.triangles, glm::vec3( 4.5f, 0.82f,  4.5f), glm::vec3(2.5f, 0.08f, 2.5f), 0.0f, 29);
        // Recessed ground luminescent and pedestrian guidance strips
        // (Mat 30: Neon Cyan 480nm, Mat 35: Electric Gold)
        addBox(scene.triangles, glm::vec3(0.0f, 0.03f, 0.0f), glm::vec3(15.6f, 0.02f, 0.25f), 0.0f, 30);
        addBox(scene.triangles, glm::vec3(0.0f, 0.03f, -4.5f), glm::vec3(0.25f, 0.02f, 7.0f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3(0.0f, 0.03f,  4.5f), glm::vec3(0.25f, 0.02f, 7.0f), 0.0f, 35);
        // Titanium streetlight poles (Mat 2: Titanium Chrome)
        addCylinder(scene.triangles, glm::vec3( 6.8f, 2.5f,  6.8f), 0.08f, 5.0f, 2, 10);
        addCylinder(scene.triangles, glm::vec3(-6.8f, 2.5f, -6.8f), 0.08f, 5.0f, 2, 10);
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

        // Solid white outer road shoulder stripes (Mat 21: White Ceramic)
        addBox(scene.triangles, glm::vec3(-4.85f, 0.045f, 0.0f), glm::vec3(0.15f, 0.01f, 15.9f), 0.0f, 21);
        addBox(scene.triangles, glm::vec3( 4.85f, 0.045f, 0.0f), glm::vec3(0.15f, 0.01f, 15.9f), 0.0f, 21);

        // Raised basalt curbs (Mat 20: Rough Basalt)
        addBox(scene.triangles, glm::vec3(-5.15f, 0.10f, 0.0f), glm::vec3(0.35f, 0.20f, 16.0f), 0.0f, 20);
        addBox(scene.triangles, glm::vec3( 5.15f, 0.10f, 0.0f), glm::vec3(0.35f, 0.20f, 16.0f), 0.0f, 20);

        // Pedestrian sidewalk pavement (Mat 4: Wet Reflective Pavement)
        addBox(scene.triangles, glm::vec3(-6.65f, 0.08f, 0.0f), glm::vec3(2.65f, 0.16f, 16.0f), 0.0f, 4);
        addBox(scene.triangles, glm::vec3( 6.65f, 0.08f, 0.0f), glm::vec3(2.65f, 0.16f, 16.0f), 0.0f, 4);

        // Recessed neon pedestrian guidance strips (Mat 30: Neon Cyan 480nm)
        addBox(scene.triangles, glm::vec3(-6.65f, 0.165f, 0.0f), glm::vec3(0.18f, 0.01f, 15.8f), 0.0f, 30);
        addBox(scene.triangles, glm::vec3( 6.65f, 0.165f, 0.0f), glm::vec3(0.18f, 0.01f, 15.8f), 0.0f, 30);

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

        // Zebra pedestrian crosswalks (Mat 21: White Ceramic) on all 4 directions
        for (int s = -4; s <= 4; ++s) {
            float xPos = static_cast<float>(s) * 1.0f;
            addBox(scene.triangles, glm::vec3(xPos, 0.045f,  6.0f), glm::vec3(0.45f, 0.01f, 2.0f), 0.0f, 21);
            addBox(scene.triangles, glm::vec3(xPos, 0.045f, -6.0f), glm::vec3(0.45f, 0.01f, 2.0f), 0.0f, 21);
        }
        for (int s = -4; s <= 4; ++s) {
            float zPos = static_cast<float>(s) * 1.0f;
            addBox(scene.triangles, glm::vec3( 6.0f, 0.045f, zPos), glm::vec3(2.0f, 0.01f, 0.45f), 0.0f, 21);
            addBox(scene.triangles, glm::vec3(-6.0f, 0.045f, zPos), glm::vec3(2.0f, 0.01f, 0.45f), 0.0f, 21);
        }

        // Corner sidewalks with wet pavement (Mat 4) and basalt curbs (Mat 20)
        for (float sx : {-1.0f, 1.0f}) {
            for (float sz : {-1.0f, 1.0f}) {
                glm::vec3 cPos(sx * 6.5f, 0.08f, sz * 6.5f);
                addBox(scene.triangles, cPos, glm::vec3(2.6f, 0.16f, 2.6f), 0.0f, 4);
                addBox(scene.triangles, cPos + glm::vec3(-sx * 1.35f, 0.02f, 0.0f), glm::vec3(0.2f, 0.20f, 2.8f), 0.0f, 20);
                addBox(scene.triangles, cPos + glm::vec3(0.0f, 0.02f, -sz * 1.35f), glm::vec3(2.8f, 0.20f, 0.2f), 0.0f, 20);
                // Corner illuminated safety bollard (Mat 2 + Mat 43 amber)
                addCylinder(scene.triangles, cPos + glm::vec3(-sx * 0.8f, 0.5f, -sz * 0.8f), 0.10f, 1.0f, 2, 8);
                addBox(scene.triangles, cPos + glm::vec3(-sx * 0.8f, 1.05f, -sz * 0.8f), glm::vec3(0.22f, 0.12f, 0.22f), 0.0f, 43);
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

        // Industrial expansion joints (Mat 35: Electric Gold / Mat 30: Neon Cyan)
        addBox(scene.triangles, glm::vec3(0.0f, 0.022f, 0.0f), glm::vec3(31.6f, 0.015f, 0.15f), 0.0f, 35);
        addBox(scene.triangles, glm::vec3(0.0f, 0.022f, 0.0f), glm::vec3(0.15f, 0.015f, 31.6f), 0.0f, 30);

        // 4 boundary perimeter beacon pylons (Mat 2 + Mat 34 Ruby Laser Warning)
        for (float bx : {-14.5f, 14.5f}) {
            for (float bz : {-14.5f, 14.5f}) {
                addCylinder(scene.triangles, glm::vec3(bx, 0.6f, bz), 0.12f, 1.2f, 2, 8);
                addBox(scene.triangles, glm::vec3(bx, 1.25f, bz), glm::vec3(0.28f, 0.18f, 0.28f), 0.0f, 34);
            }
        }

        recordBlasPrototype("Perimeter Ground", tStart);
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
                numMids = 16u + (seed % 9u); // Super-tall central avenue towers: 16 to 24 mids
            } else if (gx == 4 || gx == 7) {
                numMids = 12u + (seed % 7u); // Inner high-rise towers: 12 to 18 mids
            } else if (gx == 3 || gx == 8) {
                numMids = 10u + (seed % 5u); // Mid-district towers: 10 to 14 mids
            } else {
                numMids = 8u + (seed % 5u);  // Outer district towers: 8 to 12 mids
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
                if (m % 3 == 2 && ((gx + gz + m) % 4 == 0)) {
                    glm::mat4 mGantry = glm::translate(glm::mat4(1.0f), glm::vec3(posX + 4.8f, y + 1.0f, posZ + 4.8f));
                    addInstance(CYBER_BLAS_COMM_GANTRY, mGantry);
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

            // Holographic Billboards on tower facades
            for (float by = 16.0f; by < crownY - 6.0f; by += 14.0f) {
                uint32_t bbSeed = static_cast<uint32_t>(gx * 19 + gz * 31 + static_cast<int>(by)) % 10u;
                if (bbSeed > 6u) continue;

                uint32_t bbProto = CYBER_BLAS_BILLBOARD_CYAN;
                glm::vec3 bbLocalCorner(-3.8f, -2.05f, 0.12f);
                glm::vec3 bbLocalU(7.6f, 0.0f, 0.0f);
                glm::vec3 bbLocalV(0.0f, 4.1f, 0.0f);
                glm::vec3 bbColor(2.0f, 28.0f, 36.0f);

                if (bbSeed % 4 == 1) {
                    bbProto = CYBER_BLAS_BILLBOARD_MAGENTA;
                    bbLocalCorner = glm::vec3(-2.05f, -3.8f, 0.12f);
                    bbLocalU = glm::vec3(4.1f, 0.0f, 0.0f);
                    bbLocalV = glm::vec3(0.0f, 7.6f, 0.0f);
                    bbColor = glm::vec3(35.0f, 2.0f, 18.0f);
                } else if (bbSeed % 4 == 2) {
                    bbProto = CYBER_BLAS_BILLBOARD_ORANGE;
                    bbLocalCorner = glm::vec3(-3.3f, -1.45f, 0.12f);
                    bbLocalU = glm::vec3(6.6f, 0.0f, 0.0f);
                    bbLocalV = glm::vec3(0.0f, 2.9f, 0.0f);
                    bbColor = glm::vec3(36.0f, 12.0f, 1.5f);
                } else if (bbSeed % 4 == 3) {
                    bbProto = CYBER_BLAS_BILLBOARD_VIOLET;
                    bbLocalCorner = glm::vec3(-2.5f, -2.5f, 0.12f);
                    bbLocalU = glm::vec3(5.0f, 0.0f, 0.0f);
                    bbLocalV = glm::vec3(0.0f, 5.0f, 0.0f);
                    bbColor = glm::vec3(20.0f, 2.0f, 35.0f);
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
    // Central Avenue Skybridges (Connecting gx = 5 and gx = 6, 28m span)
    for (uint32_t gz = 0; gz < 12; ++gz) {
        float posZ = -99.0f + static_cast<float>(gz) * 18.0f;
        for (float by : { 24.0f, 44.0f, 64.0f, 84.0f }) {
            glm::mat4 mBridge = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, by, posZ)) *
                                glm::scale(glm::mat4(1.0f), glm::vec3(28.0f / 16.0f, 1.0f, 1.0f));
            addInstance(CYBER_BLAS_SKYBRIDGE, mBridge);
            // Light quad sits beneath bottom face at y = -0.11f
            addBillboardLight(mBridge, glm::vec3(-14.0f / 1.75f, -0.11f, -0.2f), glm::vec3(28.0f / 1.75f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.4f),
                              glm::vec3(2.0f, 28.0f, 36.0f), 1.0f);
        }
    }

    // Inter-Tower Skybridges along X
    for (uint32_t gx = 0; gx < 11; ++gx) {
        if (gx == 5) continue; // Avenue gap
        float posX1 = (gx <= 5) ? (-14.0f - static_cast<float>(5 - gx) * 18.0f) : (+14.0f + static_cast<float>(gx - 6) * 18.0f);
        float posX2 = ((gx + 1) <= 5) ? (-14.0f - static_cast<float>(5 - (gx + 1)) * 18.0f) : (+14.0f + static_cast<float>((gx + 1) - 6) * 18.0f);
        float midX = (posX1 + posX2) * 0.5f;

        for (uint32_t gz = 0; gz < 12; ++gz) {
            float posZ = -99.0f + static_cast<float>(gz) * 18.0f;
            if ((gx + gz) % 2 == 0) {
                float by = 28.0f + static_cast<float>((gx * 7 + gz * 11) % 4) * 16.0f;
                glm::mat4 mBridge = glm::translate(glm::mat4(1.0f), glm::vec3(midX, by, posZ)) *
                                    glm::scale(glm::mat4(1.0f), glm::vec3(18.0f / 16.0f, 1.0f, 1.0f));
                addInstance(CYBER_BLAS_SKYBRIDGE, mBridge);
                addBillboardLight(mBridge, glm::vec3(-32.0f / 9.0f, -0.11f, -0.2f), glm::vec3(64.0f / 9.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.4f),
                                  glm::vec3(2.0f, 28.0f, 36.0f), 1.0f);
            }
        }
    }

    // Inter-Tower Skybridges along Z
    for (uint32_t gx = 0; gx < 12; ++gx) {
        float posX = (gx <= 5) ? (-14.0f - static_cast<float>(5 - gx) * 18.0f) : (+14.0f + static_cast<float>(gx - 6) * 18.0f);
        for (uint32_t gz = 0; gz < 11; ++gz) {
            float posZ1 = -99.0f + static_cast<float>(gz) * 18.0f;
            float posZ2 = -99.0f + static_cast<float>(gz + 1) * 18.0f;
            float midZ = (posZ1 + posZ2) * 0.5f;
            if ((gx * 3 + gz) % 3 == 0) {
                float by = 34.0f + static_cast<float>((gx * 5 + gz * 13) % 4) * 16.0f;
                glm::mat4 mBridge = glm::translate(glm::mat4(1.0f), glm::vec3(posX, by, midZ)) *
                                    glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f)) *
                                    glm::scale(glm::mat4(1.0f), glm::vec3(18.0f / 16.0f, 1.0f, 1.0f));
                addInstance(CYBER_BLAS_SKYBRIDGE, mBridge);
                addBillboardLight(mBridge, glm::vec3(-32.0f / 9.0f, -0.11f, -0.2f), glm::vec3(64.0f / 9.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 0.4f),
                                  glm::vec3(2.0f, 28.0f, 36.0f), 1.0f);
            }
        }
    }

    // 2d. Elevated Maglev Transit Guideways (CYBER_BLAS_TRANSIT_GUIDEWAY)
    // Avenue Guideways North-South
    for (float hx : { -5.0f, 5.0f }) {
        for (int sz = -6; sz <= 6; ++sz) {
            float pz = static_cast<float>(sz) * 16.0f;
            glm::mat4 mTrackLow = glm::translate(glm::mat4(1.0f), glm::vec3(hx, 10.0f, pz)) *
                                  glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            addInstance(CYBER_BLAS_TRANSIT_GUIDEWAY, mTrackLow);

            glm::mat4 mTrackHigh = glm::translate(glm::mat4(1.0f), glm::vec3(hx, 17.0f, pz)) *
                                   glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 1.0f, 0.0f));
            addInstance(CYBER_BLAS_TRANSIT_GUIDEWAY, mTrackHigh);
        }
    }
    // Cross-Avenue Guideways East-West
    for (float cz : { -54.0f, -18.0f, 18.0f, 54.0f }) {
        for (int sx = -6; sx <= 6; ++sx) {
            float px = static_cast<float>(sx) * 16.0f;
            glm::mat4 mTrack = glm::translate(glm::mat4(1.0f), glm::vec3(px, 13.5f, cz));
            addInstance(CYBER_BLAS_TRANSIT_GUIDEWAY, mTrack);
        }
    }

    // 2e. Autonomous Sky Cabs in 3D Flight Corridors (CYBER_BLAS_SKY_CAB)
    for (int i = 0; i < 150; ++i) {
        float f = static_cast<float>(i) / 150.0f;
        float y = 14.0f + std::fmod(f * 76.0f, 72.0f);
        float z = -105.0f + f * 210.0f;
        float x = (i % 2 == 0) ? -2.4f : 2.4f;
        float rot = (i % 2 == 0) ? 90.0f : -90.0f;

        glm::mat4 mVehicle = glm::translate(glm::mat4(1.0f), glm::vec3(x, y, z)) *
                             glm::rotate(glm::mat4(1.0f), glm::radians(rot), glm::vec3(0.0f, 1.0f, 0.0f));
        addInstance(CYBER_BLAS_SKY_CAB, mVehicle);
    }
    for (int i = 0; i < 150; ++i) {
        float f = static_cast<float>(i) / 150.0f;
        float y = 18.0f + std::fmod(f * 65.0f, 60.0f);
        float x = -105.0f + f * 210.0f;
        float z = (i % 2 == 0) ? -45.0f : 45.0f;
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
                scene.materials[tri.materialId].type == MATERIAL_DIELECTRIC) {
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
    scene.cameraPosition = glm::vec3(0.0f, 32.0f, 60.0f);
    scene.cameraTarget = glm::vec3(0.0f, 25.0f, -40.0f);
    scene.cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    scene.cameraFov = 62.0f;
    scene.focalDistance = glm::length(scene.cameraPosition - scene.cameraTarget);
    scene.centralTarget = scene.cameraTarget;

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

    // Test mesh ranges if available
    if (!meshRanges.empty()) {
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
