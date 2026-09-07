#pragma once

#include <glm/glm.hpp>
#include <cstdint>

namespace pathways {

enum LightType : uint32_t {
    LIGHT_AREA_QUAD = 0,
    LIGHT_SPOT = 1,
    LIGHT_DIRECTIONAL = 2
};

struct LightGPU {
    glm::vec4 position; // xyz: corner or position, w: type
    glm::vec4 emission; // rgb: color * intensity, w: area
    glm::vec4 u;        // xyz: edge vector 1, w: spot inner angle cos
    glm::vec4 v;        // xyz: edge vector 2, w: spot outer angle cos
    glm::vec4 normal;   // xyz: normal or direction, w: padding
};

} // namespace pathways
