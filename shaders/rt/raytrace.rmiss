#version 460
#extension GL_EXT_ray_tracing : require

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692

struct HitPayload {
    vec3 diffuseRadiance;         // 12 bytes: direct diffuse
    uint packedThroughputRG;      //  4 bytes: packHalf2x16(throughputMod.rg)
    vec3 specularRadiance;        // 12 bytes: direct specular + emissive
    uint packedThroughputB_Flags; //  4 bytes: lower 16 bits = half(b), upper 16 bits = flags
    vec3 nextOrigin;              // 12 bytes: next ray origin
    uint packedNextDir;           //  4 bytes: octahedral 32-bit (oct32) unit direction
    float lastBsdfPdf;            //  4 bytes: BSDF PDF for next bounce MIS evaluation
    uint seed;                    //  4 bytes: PCG RNG state
    uint pad;                     //  4 bytes: 48-byte cache-line alignment
    uint packedAlbedoRG;          //  4 bytes: packHalf2x16(baseColor.rg)
    uint packedAlbedoB_Roughness; //  4 bytes: packHalf2x16(vec2(baseColor.b, roughness))
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
    prd.diffuseRadiance = vec3(0.0);
    prd.packedThroughputB_Flags = 0u; // hit = false
    prd.packedThroughputRG = 0u;
    prd.packedAlbedoRG = 0u;
    prd.packedAlbedoB_Roughness = 0u;
    vec3 unitDir = normalize(gl_WorldRayDirectionEXT);
    if (pc.hasEnvMap == 1u) {
        vec2 envUv = directionToEquirectangular(unitDir);
        prd.specularRadiance = texture(environmentMap, envUv).rgb * pc.envMapIntensity;
    } else {
        float t = 0.5 * (unitDir.y + 1.0);
        prd.specularRadiance = mix(vec3(0.02, 0.02, 0.04), vec3(0.05, 0.07, 0.1), t);
    }
}
