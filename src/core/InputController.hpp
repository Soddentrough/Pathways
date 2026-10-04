#pragma once

#include <SDL3/SDL.h>
#include <glm/glm.hpp>
#include <functional>
#include <algorithm>

namespace pathways {

class Window;
class Camera;
class GuiManager;
struct Config;

/**
 * @brief Subsystem encapsulating keyboard, mouse, and SDL3 gamepad navigation and interaction mode.
 */
class InputController {
public:
    struct SceneNavInfo {
        glm::vec3 centralTarget{ 0.0f };
        float focalRadius = 1.0f;
        float gpuCenterDepth = 0.0f;
        bool hasGpuCenterDepth = false;
    };

    InputController(Window* window, GuiManager* gui, const Config& config);
    ~InputController();

    InputController(const InputController&) = delete;
    InputController& operator=(const InputController&) = delete;

    void init();
    void setCamera(Camera* camera) noexcept { m_camera = camera; }
    void setSceneTarget(const glm::vec3& target, float focalRadius) noexcept {
        m_centralTarget = target;
        m_focalRadius = focalRadius;
    }
    void setFullscreenToggleCallback(std::function<void()> cb) { m_onToggleFullscreen = std::move(cb); }

    /**
     * @brief Process an incoming SDL event.
     * @return true if the event was fully consumed and should not be processed further.
     */
    bool handleEvent(const SDL_Event& e);

    /**
     * @brief Updates camera pose, gamepad look/movement, and keyboard navigation for current frame.
     */
    void update(float dt, float gpuCenterDepth, bool hasGpuCenterDepth);

    void setCameraMode(bool active);
    [[nodiscard]] bool isCameraMode() const noexcept { return m_cameraMode; }

    void rumble(uint16_t lowFreq, uint16_t highFreq, uint32_t durationMs);
    void updateLed();

    [[nodiscard]] SDL_Gamepad* getGamepad() const noexcept { return m_gamepad; }
    [[nodiscard]] bool isGamepadPs5() const noexcept { return m_gamepadIsPs5; }

private:
    void initGamepad();
    void handleGamepadAdded(SDL_JoystickID which);
    void handleGamepadRemoved(SDL_JoystickID which);

    Window* m_window = nullptr;
    GuiManager* m_gui = nullptr;
    const Config& m_config;
    Camera* m_camera = nullptr;

    glm::vec3 m_centralTarget{ 0.0f };
    float m_focalRadius = 1.0f;
    std::function<void()> m_onToggleFullscreen;

    bool m_cameraMode = false;

    // Gamepad controller state (SDL3 Gamepad API)
    SDL_Gamepad* m_gamepad = nullptr;
    bool m_gamepadIsPs5 = false;
    float m_gamepadLeftX = 0.0f;
    float m_gamepadLeftY = 0.0f;
    float m_gamepadRightX = 0.0f;
    float m_gamepadRightY = 0.0f;
    float m_gamepadLeftTrigger = 0.0f;
    float m_gamepadRightTrigger = 0.0f;
    bool m_gamepadBtnA = false;
    bool m_gamepadBtnB = false;
    bool m_gamepadBtnOrbit = false;
    bool m_gamepadBtnBumperUp = false;
    bool m_gamepadBtnBumperDown = false;
};

} // namespace pathways
