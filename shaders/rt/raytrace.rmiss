#version 460
#extension GL_EXT_ray_tracing : require

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692

struct HitPayload {
    vec3 radiance;
    vec3 throughputMod;
    vec3 nextOrigin;
    vec3 nextDirection;
    vec3 shadowOrigin;
    vec3 shadowDir;
    float shadowDist;
    vec3 shadowLightRad;
    uint seed;
    bool hasShadowRay;
    bool hit;
};

layout(location = 0) rayPayloadInEXT HitPayload prd;

layout(binding = 7) uniform sampler2D environmentMap;

layout(push_constant) uniform PushConstants {
    uint numTriangles;
    uint numSpheres;
    uint numMaterials;
    uint numLights;
    uint tileOffsetX;
    uint tileOffsetY;
    uint tileWidth;
    uint tileHeight;
    uint useHardwareRT;
    uint hasEnvMap;
    float envMapIntensity;
    uint accumulateHistory;
} pc;

vec2 directionToEquirectangular(vec3 dir) {
    float phi = atan(dir.z, dir.x);
    float theta = acos(clamp(dir.y, -1.0, 1.0));
    return vec2((phi + PI) / TWO_PI, theta / PI);
}

void main() {
    prd.hit = false;
    vec3 unitDir = normalize(gl_WorldRayDirectionEXT);
    if (pc.hasEnvMap == 1u) {
        vec2 envUv = directionToEquirectangular(unitDir);
        prd.radiance = texture(environmentMap, envUv).rgb * pc.envMapIntensity;
    } else {
        float t = 0.5 * (unitDir.y + 1.0);
        prd.radiance = mix(vec3(0.02, 0.02, 0.04), vec3(0.05, 0.07, 0.1), t);
    }
}
