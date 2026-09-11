#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace pathways {

struct ReservoirGPU {
    uint32_t lightIdx = 0;
    float    uvX = 0.0f;
    float    uvY = 0.0f;
    float    wSum = 0.0f;
    float    M = 0.0f;
    float    W = 0.0f;
    float    targetPdf = 0.0f;
    uint32_t pad = 0; // Packed geometry: lower 16 bits = oct normal, upper 16 bits = half depth
};

struct ReservoirGIGPU {
    uint32_t packedDir = 0;        // Oct-encoded secondary ray direction
    float    hitDist = 0.0f;       // Distance to secondary hit point
    uint32_t packedRadRG = 0;      // FP16 RG incoming radiance
    uint32_t packedRadB_normS = 0; // FP16 B radiance + oct-encoded secondary normal
    float    wSum = 0.0f;          // Sum of candidate weights
    float    M = 0.0f;             // Effective sample count
    float    W = 0.0f;             // Unbiased contribution weight
    uint32_t primaryGeom = 0;      // Oct-encoded primary normal + half depth
};
static_assert(sizeof(ReservoirGIGPU) == 32, "ReservoirGIGPU must be exactly 32 bytes");

struct RawGISampleGPU {
    uint32_t packedDir = 0;          // Secondary ray direction
    float    hitDist = 0.0f;         // Distance to secondary hit
    uint32_t packedRadRG = 0;        // FP16 RG incoming radiance
    uint32_t packedRadB_normS = 0;   // FP16 B radiance + oct secondary normal
    uint32_t primaryGeom = 0;        // Oct-encoded primary normal + half depth
    uint32_t primaryAlbedo = 0;      // FP16 RG primary diffuse albedo
    uint32_t primaryAlbedoB_pad = 0; // FP16 B primary albedo
    uint32_t pad = 0;
};
static_assert(sizeof(RawGISampleGPU) == 32, "RawGISampleGPU must be exactly 32 bytes");

struct CameraUniform {
    glm::mat4 viewInverse;
    glm::mat4 projInverse;
    glm::mat4 prevViewProj;
    glm::vec4 position;
    glm::vec4 viewParams; // x: fov, y: aspect, z: near, w: far
    uint32_t frameIndex;
    uint32_t spp;
    uint32_t maxBounces;
    uint32_t flags; // bit 0: direct, 1: indirect, 2: specular, 3: refraction, 4: shadows, 5: hasNonOpaque, 6: restirDI, 7: restirSpatial, bits 8..11: spatialSamples, bits 12..19: spatialRadius, bit 20: shadowDenoiser, bit 21: taa, bit 22: restirGI
    glm::mat4 unjitteredViewProj;
    glm::vec4 jitterOffset; // xy = pixel jitter [-0.5, 0.5], zw = NDC jitter
};

// Standard Halton sequence generator for low-discrepancy subpixel jittering
inline float halton(uint32_t index, uint32_t base) {
    float f = 1.0f;
    float r = 0.0f;
    while (index > 0) {
        f = f / static_cast<float>(base);
        r = r + f * static_cast<float>(index % base);
        index = index / base;
    }
    return r;
}

// 8-phase Halton(2, 3) offset centered at 0
inline glm::vec2 getHaltonJitter(uint32_t phaseIndex) {
    uint32_t idx = (phaseIndex % 8) + 1; // 1-indexed to avoid (0, 0)
    return glm::vec2(halton(idx, 2) - 0.5f, halton(idx, 3) - 0.5f);
}

class Camera {
public:
    Camera(glm::vec3 position = glm::vec3(0.0f, 1.0f, 3.5f),
           glm::vec3 target = glm::vec3(0.0f, 1.0f, 0.0f),
           float fov = 45.0f, float aspect = 16.0f / 9.0f);

    void setAspect(float aspect);
    void setFov(float fov);
    float getFov() const { return m_fov; }
    void lookAt(glm::vec3 position, glm::vec3 target, glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f));
    void update(float deltaTime);

    // Movement controls
    void processKeyboard(char direction, float deltaTime);
    void processFpsInput(float forward, float strafe, float vertical, float deltaTime, bool sprint, bool crawl = false, bool arcStrafe = false);
    void processMouseMovement(float xoffset, float yoffset, bool orbit = false);

    // Arc-strafe / Orbit controls
    void startOrbit(glm::vec3 pivot);
    void endOrbit();
    bool isOrbiting() const { return m_orbiting; }
    glm::vec3 getOrbitPivot() const { return m_orbitPivot; }
    float getOrbitRadius() const { return m_orbitRadius; }
    void setOrbitRadius(float r) { m_orbitRadius = std::max(r, 0.05f); }

    void setSpeed(float speed);
    float getSpeed() const { return m_speed; }
    void setSceneScale(float sceneRadius, float focalDistance = 0.0f, glm::vec3 centralTarget = glm::vec3(0.0f, 1.0f, 0.0f));
    float getSceneScale() const { return m_sceneScale; }
    float getFocalDistance() const { return m_focalDistance; }
    glm::vec3 getCentralTarget() const { return m_centralTarget; }
    float getCurrentTargetDistance() const;
    float getEffectiveSpeed(bool sprint = false, bool crawl = false) const;
    float getBaseSpeed() const { return m_baseSpeed; }
    float getMinSpeed() const { return m_minSpeed; }
    float getMaxSpeed() const { return m_maxSpeed; }
    void adjustSpeedByWheel(float wheelDelta);
    void setSensitivity(float sens);
    float getSensitivity() const { return m_sensitivity; }
    bool isDynamicScaling() const { return m_dynamicScaling; }
    void setDynamicScaling(bool enable) { m_dynamicScaling = enable; }
    void focusOnTarget(glm::vec3 target, float targetRadius = 0.0f);
    void resetToDefault();
    void setDefaultFraming(glm::vec3 position, glm::vec3 target, float fov) {
        m_defaultPosition = position;
        m_defaultTarget = target;
        m_defaultFov = fov;
    }

    void adaptFovForAspect(float aspect);
    bool isAdaptiveFov() const { return m_adaptiveFov; }
    void setAdaptiveFov(bool adaptive) { m_adaptiveFov = adaptive; }

    glm::mat4 getViewMatrix() const;
    glm::mat4 getProjectionMatrix() const;
    glm::vec3 getPosition() const { return m_position; }
    glm::vec3 getFront() const { return m_front; }
    glm::vec3 getUp() const { return m_up; }
    glm::vec3 getRight() const { return m_right; }
    float getYaw() const { return m_yaw; }
    float getPitch() const { return m_pitch; }

    CameraUniform getUniformData(uint32_t frameIndex, uint32_t spp, uint32_t maxBounces, uint32_t flags,
                                 bool enableTaa = false, uint32_t width = 0, uint32_t height = 0, uint32_t phaseOffset = 0) const;

    bool hasMoved() const { return m_moved; }
    void resetMoved() { m_moved = false; }
    void resetPrevViewProj() { m_hasPrevViewProj = false; }

private:
    void updateVectors();

    glm::vec3 m_position;
    glm::vec3 m_front;
    glm::vec3 m_up;
    glm::vec3 m_right;
    glm::vec3 m_worldUp;

    glm::vec3 m_defaultPosition{ 0.0f, 1.0f, 3.5f };
    glm::vec3 m_defaultTarget{ 0.0f, 1.0f, 0.0f };
    float m_defaultFov = 45.0f;

    float m_yaw = -90.0f;
    float m_pitch = 0.0f;
    float m_sceneScale = 2.0f;
    float m_focalDistance = 2.0f;
    glm::vec3 m_centralTarget{ 0.0f, 1.0f, 0.0f };
    glm::vec3 m_orbitPivot{ 0.0f, 1.0f, 0.0f };
    float m_orbitRadius = 2.0f;
    bool m_orbiting = false;
    bool m_dynamicScaling = false;
    float m_baseSpeed = 3.0f;
    float m_minSpeed = 0.05f;
    float m_maxSpeed = 50.0f;
    float m_speed = 3.0f;
    float m_sensitivity = 0.1f;
    float m_fov = 45.0f;
    float m_aspect = 16.0f / 9.0f;
    float m_near = 0.1f;
    float m_far = 100.0f;
    bool m_adaptiveFov = true;

    bool m_moved = true;
    mutable glm::mat4 m_prevViewProj{ 1.0f };
    mutable bool m_hasPrevViewProj = false;
};

} // namespace pathways
