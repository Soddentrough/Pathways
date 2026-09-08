#include "scene/ProceduralScene.hpp"
#include <cmath>

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

SceneData ProceduralScene::createCornellBox() {
    SceneData scene;

    // Materials:
    // 0: White diffuse (walls/floor/ceiling)
    MaterialGPU matWhite{};
    matWhite.albedo = glm::vec4(0.75f, 0.75f, 0.75f, 1.0f);
    matWhite.roughness = 0.9f;
    matWhite.metallic = 0.0f;
    matWhite.type = MATERIAL_DIFFUSE;
    scene.materials.push_back(matWhite);

    // 1: Red diffuse (left wall)
    MaterialGPU matRed{};
    matRed.albedo = glm::vec4(0.75f, 0.12f, 0.12f, 1.0f);
    matRed.roughness = 0.9f;
    matRed.metallic = 0.0f;
    matRed.type = MATERIAL_DIFFUSE;
    scene.materials.push_back(matRed);

    // 2: Green diffuse (right wall)
    MaterialGPU matGreen{};
    matGreen.albedo = glm::vec4(0.12f, 0.75f, 0.15f, 1.0f);
    matGreen.roughness = 0.9f;
    matGreen.metallic = 0.0f;
    matGreen.type = MATERIAL_DIFFUSE;
    scene.materials.push_back(matGreen);

    // 3: Emissive Light (ceiling area light)
    MaterialGPU matLight{};
    matLight.albedo = glm::vec4(1.0f);
    matLight.emissive = glm::vec4(18.0f, 18.0f, 15.0f, 1.0f);
    matLight.type = MATERIAL_EMISSIVE;
    scene.materials.push_back(matLight);

    // 4: Dielectric Refraction (Glass Sphere)
    MaterialGPU matGlass{};
    matGlass.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
    matGlass.roughness = 0.01f;
    matGlass.metallic = 0.0f;
    matGlass.ior = 1.52f; // Crown glass
    matGlass.transmission = 1.0f;
    matGlass.type = MATERIAL_DIELECTRIC;
    scene.materials.push_back(matGlass);

    // 5: Metallic Specular (Mirror / Metal Sphere)
    MaterialGPU matMetal{};
    matMetal.albedo = glm::vec4(0.95f, 0.85f, 0.65f, 1.0f); // Gold / polished brass tint
    matMetal.roughness = 0.04f;
    matMetal.metallic = 1.0f;
    matMetal.type = MATERIAL_METALLIC;
    scene.materials.push_back(matMetal);

    // 6: Blue diffuse box
    MaterialGPU matBlue{};
    matBlue.albedo = glm::vec4(0.2f, 0.35f, 0.8f, 1.0f);
    matBlue.roughness = 0.8f;
    matBlue.metallic = 0.0f;
    matBlue.type = MATERIAL_DIFFUSE;
    scene.materials.push_back(matBlue);

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
