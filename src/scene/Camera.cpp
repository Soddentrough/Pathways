#include "scene/Camera.hpp"
#include <algorithm>
#include <cmath>

namespace pathways {

Camera::Camera(glm::vec3 position, glm::vec3 target, float fov, float aspect)
    : m_position(position), m_worldUp(0.0f, 1.0f, 0.0f),
      m_defaultPosition(position), m_defaultTarget(target), m_defaultFov(fov),
      m_fov(fov), m_aspect(aspect) {

    glm::vec3 direction = glm::normalize(target - position);
    m_pitch = glm::degrees(std::asin(std::clamp(direction.y, -0.999f, 0.999f)));
    m_yaw = glm::degrees(std::atan2(direction.z, direction.x));

    updateVectors();
}

void Camera::setAspect(float aspect) {
    if (std::abs(m_aspect - aspect) > 0.001f) {
        if (m_adaptiveFov) {
            adaptFovForAspect(aspect);
        } else {
            m_aspect = aspect;
            m_moved = true;
        }
    }
}

void Camera::adaptFovForAspect(float aspect) {
    m_aspect = aspect;
    if (m_adaptiveFov) {
        if (aspect < 1.05f) {
            // Portrait mode: preserve horizontal framing to prevent clipping Cornell box walls.
            // Cornell box spans x in [-1, +1]. At z=3.4 distance, half-width span is z * tan(half_fov_x).
            // A horizontal half-angle of 28 degrees gives a visible span of 3.4 * tan(28 deg) = 1.80,
            // which comfortably frames the Cornell box with ~25% margin on both sides.
            float targetHalfHorizAngleRad = glm::radians(28.0f);
            float halfFovYRad = std::atan(std::tan(targetHalfHorizAngleRad) / aspect);
            float adaptedFovY = glm::degrees(halfFovYRad) * 2.0f;
            m_fov = std::clamp(adaptedFovY, 45.0f, 90.0f);
        } else {
            m_fov = m_defaultFov;
        }
    }
    m_moved = true;
}

void Camera::setFov(float fov) {
    if (std::abs(m_fov - fov) > 0.01f) {
        m_fov = fov;
        m_moved = true;
    }
}

void Camera::setSpeed(float speed) {
    m_speed = std::clamp(speed, 0.1f, 50.0f);
}

void Camera::setSensitivity(float sens) {
    m_sensitivity = std::clamp(sens, 0.01f, 2.0f);
}

void Camera::resetToDefault() {
    lookAt(m_defaultPosition, m_defaultTarget);
    setFov(m_defaultFov);
}

void Camera::update(float deltaTime) {
    // Reserved for camera smoothing / animations
    (void)deltaTime;
}

void Camera::lookAt(glm::vec3 position, glm::vec3 target, glm::vec3 up) {
    m_position = position;
    m_worldUp = up;
    glm::vec3 direction = glm::normalize(target - position);
    m_pitch = glm::degrees(std::asin(std::clamp(direction.y, -0.999f, 0.999f)));
    m_yaw = glm::degrees(std::atan2(direction.z, direction.x));
    updateVectors();
    m_moved = true;
}

void Camera::updateVectors() {
    glm::vec3 front;
    front.x = std::cos(glm::radians(m_yaw)) * std::cos(glm::radians(m_pitch));
    front.y = std::sin(glm::radians(m_pitch));
    front.z = std::sin(glm::radians(m_yaw)) * std::cos(glm::radians(m_pitch));
    m_front = glm::normalize(front);

    m_right = glm::normalize(glm::cross(m_front, m_worldUp));
    m_up = glm::normalize(glm::cross(m_right, m_front));
}

void Camera::processFpsInput(float forward, float strafe, float vertical, float deltaTime, bool sprint) {
    if (std::abs(forward) < 0.001f && std::abs(strafe) < 0.001f && std::abs(vertical) < 0.001f) {
        return;
    }

    // Move in 3D look-direction for forward/back, horizontal for strafe, and world up for vertical
    glm::vec3 moveDir = m_front * forward + m_right * strafe + m_worldUp * vertical;
    if (glm::length(moveDir) > 0.0001f) {
        moveDir = glm::normalize(moveDir);
        float currentSpeed = m_speed * (sprint ? 2.5f : 1.0f);
        m_position += moveDir * (currentSpeed * deltaTime);
        m_moved = true;
    }
}

void Camera::processKeyboard(char direction, float deltaTime) {
    float velocity = m_speed * deltaTime;
    glm::vec3 prevPos = m_position;

    if (direction == 'W' || direction == 'w') m_position += m_front * velocity;
    if (direction == 'S' || direction == 's') m_position -= m_front * velocity;
    if (direction == 'A' || direction == 'a') m_position -= m_right * velocity;
    if (direction == 'D' || direction == 'd') m_position += m_right * velocity;
    if (direction == 'E' || direction == 'e') m_position += m_worldUp * velocity;
    if (direction == 'Q' || direction == 'q') m_position -= m_worldUp * velocity;

    if (glm::length(m_position - prevPos) > 0.0001f) {
        m_moved = true;
    }
}

void Camera::processMouseMovement(float xoffset, float yoffset) {
    if (std::abs(xoffset) < 0.0001f && std::abs(yoffset) < 0.0001f) {
        return;
    }

    xoffset *= m_sensitivity;
    yoffset *= m_sensitivity;

    m_yaw += xoffset;
    m_pitch -= yoffset; // Inverted for standard FPS mouse look

    m_pitch = std::clamp(m_pitch, -89.0f, 89.0f);
    updateVectors();
    m_moved = true;
}

glm::mat4 Camera::getViewMatrix() const {
    return glm::lookAt(m_position, m_position + m_front, m_up);
}

glm::mat4 Camera::getProjectionMatrix() const {
    glm::mat4 proj = glm::perspective(glm::radians(m_fov), m_aspect, m_near, m_far);
    proj[1][1] *= -1.0f; // Vulkan Y-flip
    return proj;
}

CameraUniform Camera::getUniformData(uint32_t frameIndex, uint32_t spp, uint32_t maxBounces, uint32_t flags) const {
    CameraUniform ubo{};
    glm::mat4 view = getViewMatrix();
    glm::mat4 proj = getProjectionMatrix();

    ubo.viewInverse = glm::inverse(view);
    ubo.projInverse = glm::inverse(proj);
    ubo.position = glm::vec4(m_position, 1.0f);
    ubo.viewParams = glm::vec4(m_fov, m_aspect, m_near, m_far);
    ubo.frameIndex = frameIndex;
    ubo.spp = spp;
    ubo.maxBounces = maxBounces;
    ubo.flags = flags;

    return ubo;
}

} // namespace pathways
