#pragma once

#include <glm/glm.hpp>
#include <string>
#include <vector>
#include <memory>

namespace pathways {

enum class PathInterpolation {
    CatmullRom = 0,
    Bezier,
    Linear
};

struct CameraKeyframe {
    float time = 0.0f;               // Timestamp in seconds (must be non-decreasing)
    glm::vec3 position{ 0.0f };       // World-space camera position
    glm::vec3 target{ 0.0f };         // World-space look-at target
    glm::vec3 up{ 0.0f, 1.0f, 0.0f }; // World-space camera up vector
    float fov = 45.0f;               // Vertical FOV in degrees

    // Cubic Bézier control handles (optional)
    // C0 = position + outTangent, C1 = next_position + inTangent
    glm::vec3 inTangent{ 0.0f };
    glm::vec3 outTangent{ 0.0f };
    glm::vec3 targetInTangent{ 0.0f };
    glm::vec3 targetOutTangent{ 0.0f };
};

struct CameraSample {
    glm::vec3 position{ 0.0f };
    glm::vec3 target{ 0.0f };
    glm::vec3 up{ 0.0f, 1.0f, 0.0f };
    float fov = 45.0f;
    glm::vec3 velocity{ 0.0f };      // Instantaneous camera velocity vector (m/s)
    float speed = 0.0f;              // Length of velocity (m/s)
    bool isStationary = false;       // True if camera is holding still (speed < 1e-4)
};

class CameraPath {
public:
    CameraPath() = default;
    explicit CameraPath(const std::string& name, PathInterpolation mode = PathInterpolation::CatmullRom);

    void addKeyframe(const CameraKeyframe& kf);
    void clearKeyframes() { m_keyframes.clear(); }

    void setInterpolation(PathInterpolation mode) { m_interpolation = mode; }
    PathInterpolation getInterpolation() const { return m_interpolation; }

    void setLoop(bool loop) { m_loop = loop; }
    bool isLoop() const { return m_loop; }

    const std::string& getName() const { return m_name; }
    void setName(const std::string& name) { m_name = name; }

    size_t getKeyframeCount() const { return m_keyframes.size(); }
    const std::vector<CameraKeyframe>& getKeyframes() const { return m_keyframes; }

    float getStartTime() const;
    float getEndTime() const;
    float getDuration() const;

    bool isValid() const { return m_keyframes.size() >= 2; }

    // Evaluates continuous camera state at the specified time (in seconds)
    CameraSample evaluate(float timeSeconds, bool loopOverride = false) const;

    // File I/O & Serialization
    static std::unique_ptr<CameraPath> loadFromFile(const std::string& filepath);
    static std::unique_ptr<CameraPath> loadFromString(const std::string& jsonString);
    bool saveToFile(const std::string& filepath) const;
    std::string serializeToJSON() const;

private:
    std::string m_name = "CameraPath";
    PathInterpolation m_interpolation = PathInterpolation::CatmullRom;
    bool m_loop = false;
    std::vector<CameraKeyframe> m_keyframes;

    // Spline mathematics
    static glm::vec3 evaluateCentripetalCatmullRom(
        const glm::vec3& p0, const glm::vec3& p1,
        const glm::vec3& p2, const glm::vec3& p3,
        float t0, float t1, float t2, float t3,
        float t, float alpha = 0.5f);

    static glm::vec3 evaluateCubicBezier(
        const glm::vec3& p0, const glm::vec3& c0,
        const glm::vec3& c1, const glm::vec3& p1,
        float u);

    static glm::vec3 evaluateCubicBezierDerivative(
        const glm::vec3& p0, const glm::vec3& c0,
        const glm::vec3& c1, const glm::vec3& p1,
        float u);
};

} // namespace pathways
