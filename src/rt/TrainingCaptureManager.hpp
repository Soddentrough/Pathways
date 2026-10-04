#pragma once

#include <cstdint>
#include <string>
#include <memory>
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

namespace pathways {

class Engine;
class Camera;

/**
 * @brief Subsystem managing offline ML dataset capture and camera trajectory choreography
 *        for neural denoiser and continuous upscaler (Upways / PTTD) training.
 */
class TrainingCaptureManager {
public:
    explicit TrainingCaptureManager(Engine* engine);
    ~TrainingCaptureManager() = default;

    TrainingCaptureManager(const TrainingCaptureManager&) = delete;
    TrainingCaptureManager& operator=(const TrainingCaptureManager&) = delete;

    void run();

    void captureTrainingFrame(uint32_t frameIdx, bool isReference, uint32_t spp);
    void updateGamingChoreography(Camera* camera, uint32_t frameIdx, uint32_t totalFrames, const std::string& sceneName);
    void updateCaptureCamera(Camera* camera, uint32_t frameIdx, uint32_t totalFrames, const std::string& sceneName);

    void resetChoreography();

private:
    Engine* m_engine = nullptr;

    bool m_choreoInitialized = false;
    glm::vec3 m_choreoInitialPos{0.0f};
    float m_choreoInitialYaw = 0.0f;
    float m_choreoInitialPitch = 0.0f;
    float m_choreoInitialFov = 45.0f;
};

} // namespace pathways
