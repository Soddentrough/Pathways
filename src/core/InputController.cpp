#include "InputController.hpp"
#include "core/Window.hpp"
#include "core/Config.hpp"
#include "core/Logger.hpp"
#include "scene/Camera.hpp"
#include "ui/GuiManager.hpp"

#include <cmath>
#include <algorithm>

namespace pathways {

InputController::InputController(Window* window, GuiManager* gui, const Config& config)
    : m_window(window), m_gui(gui), m_config(config)
{
}

InputController::~InputController() {
    if (m_gamepad) {
        SDL_CloseGamepad(m_gamepad);
        m_gamepad = nullptr;
    }
}

void InputController::init() {
    initGamepad();
}

void InputController::initGamepad() {
    if (m_gamepad) return;
    int count = 0;
    SDL_JoystickID* gamepads = SDL_GetGamepads(&count);
    if (gamepads && count > 0) {
        m_gamepad = SDL_OpenGamepad(gamepads[0]);
        if (m_gamepad) {
            m_gamepadIsPs5 = (SDL_GetGamepadType(m_gamepad) == SDL_GAMEPAD_TYPE_PS5);
            Logger::info("Gamepad connected (at startup): {} ({})",
                         SDL_GetGamepadName(m_gamepad),
                         m_gamepadIsPs5 ? "PS5 DualSense" : "Standard Gamepad");
            updateLed();
        }
    }
    SDL_free(gamepads);
}

void InputController::handleGamepadAdded(SDL_JoystickID which) {
    if (!m_gamepad) {
        m_gamepad = SDL_OpenGamepad(which);
        if (m_gamepad) {
            m_gamepadIsPs5 = (SDL_GetGamepadType(m_gamepad) == SDL_GAMEPAD_TYPE_PS5);
            Logger::info("Gamepad connected: {} ({})",
                         SDL_GetGamepadName(m_gamepad),
                         m_gamepadIsPs5 ? "PS5 DualSense" : "Standard Gamepad");
            updateLed();
        }
    }
}

void InputController::handleGamepadRemoved(SDL_JoystickID which) {
    if (m_gamepad && which == SDL_GetGamepadID(m_gamepad)) {
        Logger::info("Gamepad disconnected.");
        SDL_CloseGamepad(m_gamepad);
        m_gamepad = nullptr;
        m_gamepadIsPs5 = false;
        m_gamepadLeftX = 0.0f;
        m_gamepadLeftY = 0.0f;
        m_gamepadRightX = 0.0f;
        m_gamepadRightY = 0.0f;
        m_gamepadLeftTrigger = 0.0f;
        m_gamepadRightTrigger = 0.0f;
        m_gamepadBtnA = false;
        m_gamepadBtnB = false;
        m_gamepadBtnOrbit = false;
        m_gamepadBtnBumperUp = false;
        m_gamepadBtnBumperDown = false;
    }
}

void InputController::updateLed() {
    if (!m_gamepad) return;
    if (m_cameraMode) {
        if (m_gamepadBtnOrbit || (m_camera && m_camera->isOrbiting())) {
            // Amber / Orange in Target Orbit mode
            SDL_SetGamepadLED(m_gamepad, 255, 140, 0);
        } else {
            // Emerald Cyan in 3D Scene Navigation mode
            SDL_SetGamepadLED(m_gamepad, 0, 220, 180);
        }
    } else {
        // Royal Blue in UI Control Panel mode
        SDL_SetGamepadLED(m_gamepad, 30, 80, 255);
    }
}

void InputController::rumble(uint16_t lowFreq, uint16_t highFreq, uint32_t durationMs) {
    if (m_gamepad) {
        SDL_RumbleGamepad(m_gamepad, lowFreq, highFreq, durationMs);
    }
}

void InputController::setCameraMode(bool active) {
    if (m_cameraMode == active) return;
    m_cameraMode = active;
    if (!m_cameraMode) {
        m_gamepadLeftX = 0.0f;
        m_gamepadLeftY = 0.0f;
        m_gamepadRightX = 0.0f;
        m_gamepadRightY = 0.0f;
        m_gamepadLeftTrigger = 0.0f;
        m_gamepadRightTrigger = 0.0f;
        m_gamepadBtnA = false;
        m_gamepadBtnB = false;
        m_gamepadBtnOrbit = false;
        m_gamepadBtnBumperUp = false;
        m_gamepadBtnBumperDown = false;
    }
    if (m_window) {
        m_window->setRelativeMouseMode(m_cameraMode);
    }
    updateLed();
    if (m_cameraMode) {
        rumble(0x3500, 0x6000, 90); // Crisp confirmation haptic pulse on entering 3D flight
    } else {
        rumble(0x2000, 0x2000, 70); // Gentle confirmation pulse on returning to UI
    }
    Logger::info("Interaction Mode: {}", m_cameraMode ? "FPS Scene Navigation (Mouse grabbed, WASD active)" : "UI Control Panel (Mouse released)");
}

bool InputController::handleEvent(const SDL_Event& e) {
    if (e.type == SDL_EVENT_QUIT) {
        return false;
    }

    // 1. F11 key: toggle fullscreen (intercepted first so ImGui never swallows F11)
    if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F11 && !e.key.repeat) {
        Logger::info("Received SDLK_F11 -> queueing fullscreen toggle.");
        if (m_onToggleFullscreen) {
            m_onToggleFullscreen();
        }
        return true;
    }
    if (e.type == SDL_EVENT_KEY_UP && e.key.key == SDLK_F11) {
        return true;
    }

    // 2. F12 or PrintScreen key: save current frame to high quality PNG (intercepted first so ImGui never swallows it)
    if (e.type == SDL_EVENT_KEY_DOWN && (e.key.key == SDLK_F12 || e.key.key == SDLK_PRINTSCREEN) && !e.key.repeat) {
        Logger::info("Screenshot hotkey pressed (F12 / PrintScreen) -> capturing high-quality frame.");
        if (m_onScreenshot) {
            m_onScreenshot();
        }
        return true;
    }
    if (e.type == SDL_EVENT_KEY_UP && (e.key.key == SDLK_F12 || e.key.key == SDLK_PRINTSCREEN)) {
        return true;
    }

    // 2. Window-level events
    if (e.type >= SDL_EVENT_WINDOW_FIRST && e.type <= SDL_EVENT_WINDOW_LAST) {
        if (e.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
            if (m_cameraMode) {
                setCameraMode(false);
            }
        }
        if (m_gui) {
            m_gui->processEvent(e);
        }
        return false;
    }

    // 3. TAB key toggles between UI control panel and FPS scene navigation
    if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_TAB) {
        setCameraMode(!m_cameraMode);
        return true; // Consume event so ImGui navigation does not swallow TAB
    }

    // 4. Release mouse if window focus is lost while in camera mode
    if (e.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
        if (m_cameraMode) {
            setCameraMode(false);
        }
        return false;
    }

    // 5. ESC key: if in camera mode, return to UI mode; if already in UI mode, exit the application
    if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) {
        if (m_cameraMode) {
            setCameraMode(false);
            return true;
        }
        Logger::info("ESC pressed in UI mode -> exiting application.");
        if (m_window) {
            m_window->close();
        }
        return true;
    }

    // 6. Gamepad Hotplug Events (Universal across UI and Camera modes)
    if (e.type == SDL_EVENT_GAMEPAD_ADDED) {
        handleGamepadAdded(e.gdevice.which);
        if (m_gui) {
            m_gui->processEvent(e);
        }
        return true;
    }
    if (e.type == SDL_EVENT_GAMEPAD_REMOVED) {
        handleGamepadRemoved(e.gdevice.which);
        if (m_gui) {
            m_gui->processEvent(e);
        }
        return true;
    }

    // 7. Universal Mode Toggle Gamepad Buttons (Start / Options / Touchpad / Back)
    if (e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) {
        if (e.gbutton.button == SDL_GAMEPAD_BUTTON_START ||
            e.gbutton.button == SDL_GAMEPAD_BUTTON_BACK ||
            e.gbutton.button == SDL_GAMEPAD_BUTTON_TOUCHPAD) {
            setCameraMode(!m_cameraMode);
            return true;
        }
    }

    // 8. In Camera Mode (FPS navigation)
    if (m_cameraMode) {
        if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F) {
            if (m_camera) {
                m_camera->focusOnTarget(m_centralTarget, m_focalRadius);
            }
            return true;
        }
        if (e.type == SDL_EVENT_MOUSE_MOTION) {
            if (m_camera) {
                const bool* keyState = SDL_GetKeyboardState(nullptr);
                bool ctrl = keyState && (keyState[SDL_SCANCODE_LCTRL] || keyState[SDL_SCANCODE_RCTRL]);
                m_camera->processMouseMovement(e.motion.xrel, e.motion.yrel, ctrl);
            }
            return true;
        }
        if (e.type == SDL_EVENT_MOUSE_WHEEL) {
            if (m_camera) {
                m_camera->adjustSpeedByWheel(e.wheel.y);
            }
            return true;
        }
        // Consume mouse clicks in FPS mode so they don't hit underlying ImGui elements
        if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN || e.type == SDL_EVENT_MOUSE_BUTTON_UP) {
            return true;
        }

        // Gamepad Axis Motion (Analog Sticks & Triggers)
        if (e.type == SDL_EVENT_GAMEPAD_AXIS_MOTION) {
            float deadzone = std::clamp(m_config.gamepad_deadzone, 0.01f, 0.50f);
            float rawVal = static_cast<float>(e.gaxis.value) / 32767.0f;
            float val = 0.0f;
            if (std::abs(rawVal) >= deadzone) {
                val = std::copysign((std::abs(rawVal) - deadzone) / (1.0f - deadzone), rawVal);
            }

            if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTX) m_gamepadLeftX = val;
            else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTY) m_gamepadLeftY = -val;
            else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHTX) m_gamepadRightX = val;
            else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHTY) m_gamepadRightY = val;
            else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER) m_gamepadLeftTrigger = std::max(0.0f, val);
            else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) m_gamepadRightTrigger = std::max(0.0f, val);
            return true;
        }

        // Gamepad Buttons in Camera Mode
        if (e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) {
            if (e.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH) {
                m_gamepadBtnA = true;
            } else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_EAST) {
                m_gamepadBtnB = true;
            } else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_WEST) {
                m_gamepadBtnOrbit = true;
            } else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_STICK) {
                m_gamepadBtnOrbit = !m_gamepadBtnOrbit;
                rumble(0x3000, 0x4000, 80);
            } else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_STICK) {
                if (m_camera) {
                    m_camera->focusOnTarget(m_centralTarget, m_focalRadius);
                    rumble(0x4000, 0x6000, 100);
                }
            } else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER ||
                       e.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_PADDLE1 ||
                       e.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_PADDLE2) {
                m_gamepadBtnBumperUp = true;
            } else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_SHOULDER ||
                       e.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_PADDLE1 ||
                       e.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_PADDLE2) {
                m_gamepadBtnBumperDown = true;
            } else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_DPAD_UP) {
                if (m_camera) {
                    m_camera->adjustSpeedByWheel(1.0f);
                    rumble(0x1500, 0x2500, 40);
                }
            } else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_DPAD_DOWN) {
                if (m_camera) {
                    m_camera->adjustSpeedByWheel(-1.0f);
                    rumble(0x1500, 0x2500, 40);
                }
            } else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_NORTH) {
                if (m_camera) {
                    m_camera->setPose(m_camera->getPosition(), 0.0f, m_camera->getYaw());
                    rumble(0x2000, 0x3000, 60);
                }
            }
            return true;
        }

        if (e.type == SDL_EVENT_GAMEPAD_BUTTON_UP) {
            if (e.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH) m_gamepadBtnA = false;
            else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_EAST) m_gamepadBtnB = false;
            else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_WEST) m_gamepadBtnOrbit = false;
            else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER ||
                     e.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_PADDLE1 ||
                     e.gbutton.button == SDL_GAMEPAD_BUTTON_RIGHT_PADDLE2) {
                m_gamepadBtnBumperUp = false;
            } else if (e.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_SHOULDER ||
                       e.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_PADDLE1 ||
                       e.gbutton.button == SDL_GAMEPAD_BUTTON_LEFT_PADDLE2) {
                m_gamepadBtnBumperDown = false;
            }
            return true;
        }

        return false;
    }

    // 9. In UI Mode: route events to ImGui
    if (m_gui) {
        // If mouse wheel happened outside ImGui windows, adjust camera speed
        if (e.type == SDL_EVENT_MOUSE_WHEEL && !m_gui->wantCaptureMouse()) {
            if (m_camera) {
                m_camera->adjustSpeedByWheel(e.wheel.y);
            }
            return true;
        }

        // Process event with Dear ImGui
        m_gui->processEvent(e);

        // If left click happened outside ImGui windows, capture mouse for FPS navigation
        if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT) {
            if (!m_gui->wantCaptureMouse()) {
                setCameraMode(true);
                return true;
            }
            return true;
        }

        // If South (Cross/A) pressed outside ImGui windows/controls, enter FPS navigation
        if (e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN && e.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH) {
            if (!m_gui->wantCaptureMouse() && !m_gui->wantCaptureKeyboard()) {
                setCameraMode(true);
                return true;
            }
            return true;
        }

        if (m_gui->wantCaptureMouse() || m_gui->wantCaptureKeyboard()) {
            return true;
        }
    }

    return false;
}

void InputController::update(float dt, float gpuCenterDepth, bool hasGpuCenterDepth) {
    if (!m_cameraMode || m_config.headless || !m_camera) {
        return;
    }

    const bool* keyState = SDL_GetKeyboardState(nullptr);
    if (!keyState) return;

    glm::vec3 camPos = m_camera->getPosition();
    glm::vec3 camFront = m_camera->getFront();
    bool hitGeometry = false;
    glm::vec3 hitPoint = camPos + camFront * m_camera->getFocalDistance();
    if (hasGpuCenterDepth && gpuCenterDepth > 0.05f) {
        m_camera->setLookDistance(gpuCenterDepth);
        hitGeometry = true;
        hitPoint = camPos + camFront * gpuCenterDepth;
    } else {
        // Looking at open sky or empty space: use cruising scale distance
        float fallbackDist = std::max(m_camera->getFocalDistance() * 2.5f, m_camera->getSceneScale());
        m_camera->setLookDistance(fallbackDist);
    }

    float kbdForward = 0.0f;
    float kbdStrafe = 0.0f;
    float kbdVertical = 0.0f;

    if (keyState[SDL_SCANCODE_W]) kbdForward += 1.0f;
    if (keyState[SDL_SCANCODE_S]) kbdForward -= 1.0f;
    if (keyState[SDL_SCANCODE_D]) kbdStrafe += 1.0f;
    if (keyState[SDL_SCANCODE_A]) kbdStrafe -= 1.0f;
    if (keyState[SDL_SCANCODE_SPACE] || keyState[SDL_SCANCODE_E]) kbdVertical += 1.0f;
    if (keyState[SDL_SCANCODE_C] || keyState[SDL_SCANCODE_Q]) kbdVertical -= 1.0f;

    bool kbdMoving = (std::abs(kbdForward) > 0.001f || std::abs(kbdStrafe) > 0.001f || std::abs(kbdVertical) > 0.001f);
    bool sprint = keyState[SDL_SCANCODE_LSHIFT] || keyState[SDL_SCANCODE_RSHIFT];
    bool crawl = keyState[SDL_SCANCODE_LALT] || keyState[SDL_SCANCODE_RALT];
    bool ctrl = keyState[SDL_SCANCODE_LCTRL] || keyState[SDL_SCANCODE_RCTRL] || m_gamepadBtnOrbit;

    if (ctrl) {
        if (!m_camera->isOrbiting()) {
            if (hitGeometry) {
                m_camera->startOrbit(hitPoint);
            } else {
                // Fallback: project centralTarget or use focal distance along view ray
                glm::vec3 toCenter = m_centralTarget - camPos;
                float proj = glm::dot(toCenter, camFront);
                float dist = (proj > 0.1f) ? proj : m_camera->getFocalDistance();
                m_camera->startOrbit(camPos + camFront * dist);
            }
            updateLed();
        }
    } else {
        if (m_camera->isOrbiting()) {
            m_camera->endOrbit();
            updateLed();
        }
    }

    // Incorporate analog gamepad sticks & buttons (FEAT-01)
    float forward = kbdForward + m_gamepadLeftY;
    float strafe = kbdStrafe + m_gamepadLeftX;
    float vertical = kbdVertical;
    if (m_gamepadBtnA || m_gamepadBtnBumperUp) vertical += 1.0f;
    if (m_gamepadBtnB || m_gamepadBtnBumperDown) vertical -= 1.0f;

    // Analog trigger progressive gear multiplier:
    // Right trigger (RT / R2) progressively sprints from 1.0x up to 3.0x speed
    // Left trigger (LT / L2) progressively crawls from 1.0x down to 0.25x precision speed
    float analogSpeedScale = 1.0f;
    if (m_gamepadRightTrigger > 0.01f) {
        analogSpeedScale *= (1.0f + 2.0f * m_gamepadRightTrigger);
    }
    if (m_gamepadLeftTrigger > 0.01f) {
        analogSpeedScale *= (1.0f - 0.75f * m_gamepadLeftTrigger);
    }

    // Dynamic distance-adaptive speed is reserved for binary keyboard inputs.
    // For gamepad analog inputs, speed maps strictly and proportionally to stick deflection and trigger depth.
    bool applyDynamicScaling = kbdMoving;

    if (std::abs(m_gamepadRightX) > 0.05f || std::abs(m_gamepadRightY) > 0.05f) {
        constexpr float BASE_GAMEPAD_YAW_SPEED = 400.0f; // degrees per second direct rate
        constexpr float BASE_GAMEPAD_PITCH_SPEED = 240.0f; // degrees per second direct rate
        float sensitivity = std::clamp(m_config.gamepad_sensitivity, 0.10f, 5.00f);
        float yawSpeed = BASE_GAMEPAD_YAW_SPEED * sensitivity;
        float pitchSpeed = BASE_GAMEPAD_PITCH_SPEED * sensitivity;
        float pitchDir = m_config.gamepad_invert_y ? -1.0f : 1.0f;

        // Smooth ergonomic response curve: slight exponent for fine center control + fast turn at edge
        float rx = std::copysign(std::pow(std::abs(m_gamepadRightX), 1.25f), m_gamepadRightX);
        float ry = std::copysign(std::pow(std::abs(m_gamepadRightY), 1.25f), m_gamepadRightY);

        m_camera->processGamepadLook(rx * yawSpeed * dt, pitchDir * ry * pitchSpeed * dt, ctrl);
    }

    m_camera->processFpsInput(forward, strafe, vertical, dt, sprint, crawl, ctrl, applyDynamicScaling, analogSpeedScale);
    m_camera->update(dt);
}

} // namespace pathways
