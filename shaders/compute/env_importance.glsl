#ifndef ENV_IMPORTANCE_GLSL
#define ENV_IMPORTANCE_GLSL

// =============================================================================
// 2D Equirectangular Environment Importance Table (Binding 46 SSBO)
// =============================================================================
// Self-contained header safe for inclusion from both compute kernels (via
// mis_common.glsl) and ray tracing miss shaders (raytrace.rmiss). Must NOT
// reference lights[], triangles[], instances[], or PushConstants.
// =============================================================================

const float ENV_IMP_PI         = 3.14159265358979323846;
const float ENV_IMP_TWO_PI     = 6.28318530717958647692;
const float ENV_IMP_TWO_PI_SQ  = 19.73920880217871723767; // 2 * pi^2

// Sentinel light index packed into the 24-bit DiReservoir::idxM field when
// a reservoir sample represents an environment-map direction.
// Callers MUST check (lightIdx == RESTIR_LIGHT_IDX_ENV) BEFORE indexing lights[].
const uint RESTIR_LIGHT_IDX_ENV = 0xFFFFFFu;

struct EnvAliasEntry {
    float q;
    uint alias;
    float pdf;
    float pad;
};

layout(std430, binding = 46) readonly buffer EnvImportanceBuffer {
    uint envImpWidth;
    uint envImpHeight;
    float envSelectProb;
    float envTotalFlux;
    EnvAliasEntry envAliasEntries[];
};

bool envNeeActive() {
    return (envImpWidth > 0u) && (envImpHeight > 0u) && (envSelectProb > 0.0);
}

float envNeeSelectProb() {
    return envNeeActive() ? clamp(envSelectProb, 0.0, 1.0) : 0.0;
}

vec2 envDirToUv(vec3 rawDir) {
    float len = length(rawDir);
    vec3 dir = (len > 1e-8) ? (rawDir / len) : vec3(0.0, 1.0, 0.0);
    float phi = atan(dir.z, dir.x);
    float theta = acos(clamp(dir.y, -1.0, 1.0));
    return vec2((phi + ENV_IMP_PI) / ENV_IMP_TWO_PI, theta / ENV_IMP_PI);
}

vec3 envUvToDir(vec2 uv) {
    float phi = uv.x * ENV_IMP_TWO_PI - ENV_IMP_PI;
    float theta = uv.y * ENV_IMP_PI;
    float sinTheta = sin(theta);
    float cosTheta = cos(theta);
    return vec3(sinTheta * cos(phi), cosTheta, sinTheta * sin(phi));
}

float evalEnvMapPdf(vec3 rawDir) {
    if (!envNeeActive()) return 0.0;
    float len = length(rawDir);
    vec3 dir = (len > 1e-8) ? (rawDir / len) : vec3(0.0, 1.0, 0.0);
    vec2 uv = envDirToUv(dir);
    uint x = min(uint(max(uv.x, 0.0) * float(envImpWidth)), envImpWidth - 1u);
    uint y = min(uint(max(uv.y, 0.0) * float(envImpHeight)), envImpHeight - 1u);
    float uvPdf = envAliasEntries[y * envImpWidth + x].pdf;
    float sinTheta = max(sqrt(dir.x * dir.x + dir.z * dir.z), 1e-5);
    return uvPdf / (ENV_IMP_TWO_PI_SQ * sinTheta);
}

bool sampleEnvMapImportance(vec3 u, out vec3 outDir, out vec2 outUv, out float outSolidAnglePdf) {
    outDir = vec3(0.0, 1.0, 0.0);
    outUv = vec2(0.5);
    outSolidAnglePdf = 0.0;
    if (!envNeeActive()) return false;

    uint total = envImpWidth * envImpHeight;
    float scaled = clamp(u.x, 0.0, 0.9999999) * float(total);
    uint col = min(uint(scaled), total - 1u);
    float frac = scaled - float(col);
    EnvAliasEntry e = envAliasEntries[col];
    uint cell = (frac < e.q) ? col : e.alias;
    float uvPdf = (cell == col) ? e.pdf : envAliasEntries[cell].pdf;
    uint cx = cell % envImpWidth;
    uint cy = cell / envImpWidth;
    float jx = clamp(u.y, 1e-4, 1.0 - 1e-4);
    float jy = clamp(u.z, 1e-4, 1.0 - 1e-4);
    outUv = vec2((float(cx) + jx) / float(envImpWidth), (float(cy) + jy) / float(envImpHeight));
    outDir = envUvToDir(outUv);
    float sinTheta = max(sqrt(outDir.x * outDir.x + outDir.z * outDir.z), 1e-5);
    outSolidAnglePdf = uvPdf / (ENV_IMP_TWO_PI_SQ * sinTheta);
    return outSolidAnglePdf > 0.0;
}

uint envEncodeBsdfPdf(float pdf) {
    if (!(pdf > 0.0)) return 1u;
    float c = round((log2(pdf) + 20.0) * 64.0);
    return uint(clamp(c, 1.0, 4095.0));
}

float envDecodeBsdfPdf(uint code) {
    return exp2(float(code) * (1.0 / 64.0) - 20.0);
}

// Balance-heuristic BSDF-hit MIS weight for sky / environment misses.
// uboFlags bits match mis_common.glsl:
//   bit 0:  UBO_FLAG_DIRECT_LIGHT
//   bit 25: UBO_FLAG_REFERENCE
//   bit 28: UBO_FLAG_DBG_NO_BSDF_EMIT
float envHitMisWeightForFlags(vec3 dir, uint pdfCode, uint numLights, uint uboFlags, float lightSampleScale) {
    if (pdfCode == 0u) return 1.0;
    if ((uboFlags & (1u << 28)) != 0u) return 0.0;
    if ((uboFlags & (1u << 0)) == 0u || !envNeeActive()) return 1.0;
    if ((uboFlags & (1u << 25)) != 0u) return 1.0;
    float pEnv = (numLights == 0u) ? 1.0 : envNeeSelectProb();
    if (pEnv <= 0.0) return 1.0;
    float pl = evalEnvMapPdf(dir) * pEnv * max(lightSampleScale, 1.0);
    if (pl <= 0.0) return 1.0;
    float pb = envDecodeBsdfPdf(pdfCode);
    return pb / (pb + pl);
}

#endif // ENV_IMPORTANCE_GLSL
