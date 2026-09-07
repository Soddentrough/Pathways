#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace pathways {

struct CameraUniform {
    glm::mat4 viewInverse;
    glm::mat4 projInverse;
    glm::vec4 position;
    glm::vec4 viewParams; // x: fov, y: aspect, z: near, w: far
    uint32_t frameIndex;
    uint32_t spp;
    uint32_t maxBounces;
    uint32_t flags; // bit 0: direct, 1: indirect, 2: specular, 3: refraction, 4: shadows
};

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
    void processFpsInput(float forward, float strafe, float vertical, float deltaTime, bool sprint);
    void processMouseMovement(float xoffset, float yoffset);

    void setSpeed(float speed);
    float getSpeed() const { return m_speed; }
    void setSensitivity(float sens);
    float getSensitivity() const { return m_sensitivity; }
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

    CameraUniform getUniformData(uint32_t frameIndex, uint32_t spp, uint32_t maxBounces, uint32_t flags) const;

    bool hasMoved() const { return m_moved; }
    void resetMoved() { m_moved = false; }

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
    float m_speed = 3.0f;
    float m_sensitivity = 0.1f;
    float m_fov = 45.0f;
    float m_aspect = 16.0f / 9.0f;
    float m_near = 0.1f;
    float m_far = 100.0f;
    bool m_adaptiveFov = true;

    bool m_moved = true;
};

} // namespace pathways
