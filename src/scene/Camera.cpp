#include "scene/Camera.hpp"
#include <algorithm>
#include <cmath>

namespace pathways {

Camera::Camera(glm::vec3 position, glm::vec3 target, float fov, float aspect)
    : m_position(position), m_worldUp(0.0f, 1.0f, 0.0f),
      m_defaultPosition(position), m_defaultTarget(target), m_defaultFov(fov),
      m_fov(fov), m_aspect(aspect) {

    m_focalDistance = glm::length(target - position);
    if (m_focalDistance < 0.1f) m_focalDistance = 2.0f;
    m_centralTarget = target;

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

void Camera::setSceneScale(float sceneRadius, float focalDistance, glm::vec3 centralTarget) {
    m_sceneScale = std::max(sceneRadius, 0.05f);
    m_focalDistance = (focalDistance > 0.01f) ? focalDistance : m_sceneScale;
    m_centralTarget = centralTarget;

    // Constant Half-Distance Time Law:
    // Move half the distance to the central object in T_half = 2.0 seconds:
    // v_base = D / (2 * T_half) = 0.25 * D
    // Near a cup (D=0.85m), v_base = ~0.21 m/s (moves 3.4mm/frame, precise centering)
    // Near a car (D=11-17m), v_base = ~2.7-4.4 m/s (natural vehicle walkthrough)
    // In vast scenes / distant dragon (D=80m), v_base = ~20 m/s (smooth, prompt relocation)
    float baseRate = 0.25f; // s^-1 (closes half the distance in 2.0 seconds)
    m_baseSpeed = std::clamp(m_focalDistance * baseRate, 0.05f, 100.0f);
    m_speed = m_baseSpeed;
    m_minSpeed = std::max(m_baseSpeed * 0.02f, 0.005f);
    m_maxSpeed = std::min(m_baseSpeed * 30.0f, 500.0f);

    // Adapt near and far clipping planes proportionally
    m_near = std::clamp(m_sceneScale * 0.001f, 0.001f, 0.1f);
    m_far = std::max(m_sceneScale * 50.0f, 200.0f);
    m_moved = true;
}

float Camera::getCurrentTargetDistance() const {
    float dist = glm::length(m_centralTarget - m_position);
    float minD = std::max(0.05f, 0.05f * m_sceneScale);
    return std::max(dist, minD);
}

float Camera::getEffectiveSpeed(bool sprint, bool crawl) const {
    float currentDist = getCurrentTargetDistance();
    float distRatio = (m_focalDistance > 0.01f) ? (currentDist / m_focalDistance) : 1.0f;
    float distFactor = m_dynamicScaling ? std::clamp(distRatio, 0.15f, 5.0f) : 1.0f;

    float gearMultiplier = 1.0f;
    if (crawl) {
        gearMultiplier = 0.25f; // Precision crawl gear for micro-centering
    } else if (sprint) {
        gearMultiplier = 3.0f;  // Sprint relocation gear
    }

    return std::clamp(m_speed * distFactor * gearMultiplier, m_minSpeed * 0.1f, m_maxSpeed * 3.0f);
}

void Camera::focusOnTarget(glm::vec3 target, float targetRadius) {
    glm::vec3 dir = m_position - target;
    float currentDist = glm::length(dir);
    if (currentDist < 0.001f) {
        dir = glm::vec3(0.0f, 0.5f, 1.0f);
        currentDist = 1.0f;
    }
    dir = glm::normalize(dir);

    float desiredDist = (targetRadius > 0.01f) ? (targetRadius * 2.5f) : m_focalDistance;
    lookAt(target + dir * desiredDist, target);
    m_speed = m_baseSpeed;
    m_moved = true;
}

void Camera::adjustSpeedByWheel(float wheelDelta) {
    if (wheelDelta > 0.0f) {
        m_speed = std::min(m_speed * 1.25f, m_maxSpeed);
    } else if (wheelDelta < 0.0f) {
        m_speed = std::max(m_speed / 1.25f, m_minSpeed);
    }
}

void Camera::setSpeed(float speed) {
    m_speed = std::clamp(speed, m_minSpeed, m_maxSpeed);
}

void Camera::setSensitivity(float sens) {
    m_sensitivity = std::clamp(sens, 0.01f, 2.0f);
}

void Camera::resetToDefault() {
    lookAt(m_defaultPosition, m_defaultTarget);
    setFov(m_defaultFov);
    m_speed = m_baseSpeed;
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
    float dist = glm::length(target - position);
    if (dist > 0.05f) {
        m_focalDistance = dist;
    }
    m_centralTarget = target;
    updateVectors();
    m_moved = true;
    m_hasPrevViewProj = false;
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

void Camera::startOrbit(glm::vec3 pivot) {
    m_orbitPivot = pivot;
    float dist = glm::length(m_position - pivot);
    m_orbitRadius = std::max(dist, 0.05f);
    m_orbiting = true;
}

void Camera::endOrbit() {
    m_orbiting = false;
}

void Camera::processFpsInput(float forward, float strafe, float vertical, float deltaTime, bool sprint, bool crawl, bool arcStrafe) {
    if (std::abs(forward) < 0.001f && std::abs(strafe) < 0.001f && std::abs(vertical) < 0.001f) {
        return;
    }

    if (arcStrafe && m_orbiting) {
        // Arc-strafe / turntable orbit around m_orbitPivot
        glm::vec3 r = m_position - m_orbitPivot;
        float currentDist = glm::length(r);
        if (currentDist < 0.01f) {
            r = -m_front * std::max(m_orbitRadius, 0.5f);
            currentDist = glm::length(r);
        }

        float speed = getEffectiveSpeed(sprint, crawl);
        float angularSpeed = speed / currentDist; // Constant angular rate from half-distance law

        // 1. Horizontal arc strafe (A / D) around world up
        if (std::abs(strafe) > 0.001f) {
            float angle = strafe * angularSpeed * deltaTime;
            glm::mat4 rotY = glm::rotate(glm::mat4(1.0f), angle, m_worldUp);
            r = glm::vec3(rotY * glm::vec4(r, 0.0f));
        }

        // 2. Vertical elevation arc (Space/E up, C/Q down) around m_right
        if (std::abs(vertical) > 0.001f) {
            float vAngle = -vertical * angularSpeed * deltaTime;
            glm::mat4 rotRight = glm::rotate(glm::mat4(1.0f), vAngle, m_right);
            glm::vec3 candidateR = glm::vec3(rotRight * glm::vec4(r, 0.0f));

            // Clamp pitch to [-85 deg, +85 deg] to prevent gimbal inversion
            glm::vec3 candDir = glm::normalize(candidateR);
            float pitchRad = std::asin(std::clamp(candDir.y, -0.996f, 0.996f));
            if (std::abs(glm::degrees(pitchRad)) <= 85.0f) {
                r = candidateR;
            }
        }

        // 3. Radial dolly (W forward / S backward)
        if (std::abs(forward) > 0.001f) {
            float minDist = std::max(0.05f, 0.02f * m_sceneScale);
            currentDist = std::max(currentDist - forward * speed * deltaTime, minDist);
            r = glm::normalize(r) * currentDist;
            m_orbitRadius = currentDist;
        }

        m_position = m_orbitPivot + r;

        // Keep camera facing directly at pivot
        glm::vec3 dir = glm::normalize(m_orbitPivot - m_position);
        m_pitch = glm::degrees(std::asin(std::clamp(dir.y, -0.999f, 0.999f)));
        m_yaw = glm::degrees(std::atan2(dir.z, dir.x));
        updateVectors();
        m_moved = true;
        return;
    }

    // Standard FPS movement
    glm::vec3 moveDir = m_front * forward + m_right * strafe + m_worldUp * vertical;
    if (glm::length(moveDir) > 0.0001f) {
        moveDir = glm::normalize(moveDir);
        float currentSpeed = getEffectiveSpeed(sprint, crawl);
        m_position += moveDir * (currentSpeed * deltaTime);
        m_moved = true;
    }
}

void Camera::processKeyboard(char direction, float deltaTime) {
    float velocity = getEffectiveSpeed(false, false) * deltaTime;
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

void Camera::processMouseMovement(float xoffset, float yoffset, bool orbit) {
    if (std::abs(xoffset) < 0.0001f && std::abs(yoffset) < 0.0001f) {
        return;
    }

    xoffset *= m_sensitivity;
    yoffset *= m_sensitivity;

    if (orbit && m_orbiting) {
        glm::vec3 r = m_position - m_orbitPivot;
        float currentDist = glm::length(r);
        if (currentDist < 0.01f) {
            r = -m_front * std::max(m_orbitRadius, 0.5f);
        }

        // Orbit horizontally
        float angleH = -glm::radians(xoffset);
        glm::mat4 rotY = glm::rotate(glm::mat4(1.0f), angleH, m_worldUp);
        r = glm::vec3(rotY * glm::vec4(r, 0.0f));

        // Orbit vertically
        float angleV = glm::radians(yoffset);
        glm::mat4 rotRight = glm::rotate(glm::mat4(1.0f), angleV, m_right);
        glm::vec3 candidateR = glm::vec3(rotRight * glm::vec4(r, 0.0f));

        glm::vec3 candDir = glm::normalize(candidateR);
        float pitchRad = std::asin(std::clamp(candDir.y, -0.996f, 0.996f));
        if (std::abs(glm::degrees(pitchRad)) <= 85.0f) {
            r = candidateR;
        }

        m_position = m_orbitPivot + r;
        glm::vec3 dir = glm::normalize(m_orbitPivot - m_position);
        m_pitch = glm::degrees(std::asin(std::clamp(dir.y, -0.999f, 0.999f)));
        m_yaw = glm::degrees(std::atan2(dir.z, dir.x));
        updateVectors();
        m_moved = true;
        return;
    }

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
    glm::mat4 currentViewProj = proj * view;

    ubo.viewInverse = glm::inverse(view);
    ubo.projInverse = glm::inverse(proj);
    ubo.prevViewProj = m_hasPrevViewProj ? m_prevViewProj : currentViewProj;
    m_prevViewProj = currentViewProj;
    m_hasPrevViewProj = true;

    ubo.position = glm::vec4(m_position, 1.0f);
    ubo.viewParams = glm::vec4(m_fov, m_aspect, m_near, m_far);
    ubo.frameIndex = frameIndex;
    ubo.spp = spp;
    ubo.maxBounces = maxBounces;
    ubo.flags = flags;

    return ubo;
}

} // namespace pathways
