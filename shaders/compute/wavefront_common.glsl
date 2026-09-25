#ifndef WAVEFRONT_COMMON_GLSL
#define WAVEFRONT_COMMON_GLSL

#extension GL_EXT_ray_query : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_KHR_shader_subgroup_shuffle : enable
#extension GL_EXT_control_flow_attributes : enable
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : enable
#extension GL_EXT_nonuniform_qualifier : enable

#include "nrc_common.glsl"

#define SAMPLE_SCENE_TEXTURE(texId, uv) textureLod(sceneTextures[nonuniformEXT((texId) - 1u)], (uv), 0.0)

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692
#define INV_PI 0.31830988618379067154
#define EPSILON 0.0005

#define RAY_MASK_OPAQUE 0x01
#define RAY_MASK_NON_OPAQUE 0x02
#define RAY_MASK_ALL 0xFF

// 128-byte cache-line aligned shading triangle (Option 3 / RDNA 4 vector cache line)
struct Triangle {
    vec4 normal0_u0; // xyz: normal0, w: uv0.x
    vec4 normal1_u1; // xyz: normal1, w: uv1.x
    vec4 normal2_u2; // xyz: normal2, w: uv2.x
    vec4 tan0_v0;    // xyz: tan0,    w: uv0.y
    vec4 tan1_v1;    // xyz: tan1,    w: uv1.y
    vec4 tan2_v2;    // xyz: tan2,    w: uv2.y
    vec4 tanSigns;   // x: tan0.w, y: tan1.w, z: tan2.w, w: unused
    uint materialId;
    uint padding[3];
};

vec3 getTriangleNormal(in Triangle tri, in vec2 bary) {
    float u = bary.x;
    float v = bary.y;
    float w = 1.0 - u - v;
    return normalize(w * tri.normal0_u0.xyz + u * tri.normal1_u1.xyz + v * tri.normal2_u2.xyz);
}

vec2 getTriangleUV(in Triangle tri, in vec2 bary) {
    float u = bary.x;
    float v = bary.y;
    float w = 1.0 - u - v;
    return w * vec2(tri.normal0_u0.w, tri.tan0_v0.w) +
           u * vec2(tri.normal1_u1.w, tri.tan1_v1.w) +
           v * vec2(tri.normal2_u2.w, tri.tan2_v2.w);
}

vec3 getTriangleTangent(in Triangle tri, in vec2 bary) {
    float u = bary.x;
    float v = bary.y;
    float w = 1.0 - u - v;
    vec3 geomTan = w * tri.tan0_v0.xyz + u * tri.tan1_v1.xyz + v * tri.tan2_v2.xyz;
    float len = length(geomTan);
    return len > 1e-4 ? (geomTan / len) : vec3(1.0, 0.0, 0.0);
}

float getTriangleTangentSign(in Triangle tri) {
    return tri.tanSigns.x != 0.0 ? tri.tanSigns.x : 1.0;
}

struct Sphere {
    vec4 centerRadius; // xyz: center, w: radius
    uint materialId;
    uint padding[3];
};

struct InstanceGPU {
    uint firstTriangle;
    uint numOpaqueTriangles;
    uint materialOffset;
    uint flags;
};

layout(std430, binding = 30) readonly buffer InstancesBuffer {
    InstanceGPU instances[];
};

layout(std430, binding = 34) readonly buffer MaterialArchetypesBuffer {
    uint materialArchetypes[];
};

struct ShadeMaterial {
    vec4 albedo;
    vec4 emissive_spec;
    vec4 pbrParams;
    uvec4 tex_flags;
};

layout(std430, binding = 35) readonly buffer ShadeMaterialsBuffer {
    ShadeMaterial shadeMaterials[];
};

struct Material {
    vec4 albedo;
    vec4 emissive;
    float roughness;
    float metallic;
    float ior;
    float transmission;
    uint type; // 0: diffuse, 1: metallic, 2: dielectric, 3: emissive
    uint albedoTex;
    uint normalTex;
    uint mrTex;
    uint emissiveTex;
    uint occlusionTex;
    uint transmissionTex;
    float alphaCutoff;
    uint alphaMode;
    float occlusionStrength;
    float normalScale;
    uint thicknessTex;

    // Extended glTF 2.0 PBR Properties (offsets 96-144)
    vec4 attenuationColor;
    float clearcoat;
    float clearcoatRoughness;
    uint clearcoatTex;
    uint clearcoatRoughnessTex;
    uint clearcoatNormalTex;
    float thickness;
    float specularFactor;
    uint specularTex;

    // Tier 2 glTF Extensions (offsets 144-192)
    float anisotropyStrength;
    float anisotropyRotation;
    uint anisotropyTex;
    float dispersion;
    vec3 sheenColor;
    float sheenRoughness;
    float iridescence;
    float iridescenceIor;
    float iridescenceThickness;
    uint sheenTex;

    // Thin-Walled Diffuse Transmission (offsets 192-208)
    float diffuseTransmission;
    uint diffuseTransmissionTex;
    vec2 diffuseTransPad;
};

struct Light {
    vec4 position; // xyz: pos/corner, w: type (0: area, 1: spot, 2: directional)
    vec4 emission; // rgb: color, w: area
    vec4 u;        // xyz: edge1, w: spot inner cos
    vec4 v;        // xyz: edge2, w: spot outer cos
    vec4 normal;   // xyz: normal/dir, w: padding
    vec4 sampling; // x: q (prob threshold), y: as uint aliasIdx, z: discrete selection pdf, w: flux (96 bytes)
};

// O(1) Vose Alias Table sampling for discrete radiant flux distribution
#define SAMPLE_LIGHT_ALIAS(numLights, seed, outIdx, outPdf) \
    do { \
        if ((numLights) <= 1u) { \
            outIdx = 0u; \
            outPdf = 1.0; \
        } else { \
            float u_alias_ = randFloat(seed); \
            float f_alias_ = u_alias_ * float(numLights); \
            uint b_alias_ = min(uint(f_alias_), (numLights) - 1u); \
            float remap_alias_ = f_alias_ - float(b_alias_); \
            vec4 samp_ = lights[b_alias_].sampling; \
            if (remap_alias_ < samp_.x) { \
                outIdx = b_alias_; \
                outPdf = samp_.z; \
            } else { \
                outIdx = floatBitsToUint(samp_.y); \
                outPdf = lights[outIdx].sampling.z; \
            } \
            outPdf = max(outPdf, 1e-6); \
        } \
    } while(false)

// 64-byte hierarchical 3D Light Tree node layout (FEAT-02)
struct LightTreeNode {
    vec4 bboxMin;   // xyz: bboxMin, w: radiant flux (Phi)
    vec4 bboxMax;   // xyz: bboxMax, w: coneAngleCos
    vec4 coneAxis;  // xyz: cone center axis, w: padding
    uvec4 children; // x: leftChild (or lightIdx if leaf), y: rightChild, z: isLeaf, w: padding
};

// Evaluate spatial importance metric of a LightTree node from surface hit point x with normal n
float evalLightTreeNodeImportance(vec3 hitPoint, vec3 hitNormal, LightTreeNode node) {
    float flux = node.bboxMin.w;
    if (flux <= 1e-7) return 0.0;

    vec3 bMin = node.bboxMin.xyz;
    vec3 bMax = node.bboxMax.xyz;

    // Point-to-box minimum squared distance
    vec3 clampedPoint = clamp(hitPoint, bMin, bMax);
    vec3 delta = hitPoint - clampedPoint;
    float distSq = max(dot(delta, delta), 1e-4);

    // Direction vector from hitPoint to node centroid
    vec3 boxCentroid = 0.5 * (bMin + bMax);
    vec3 toNode = boxCentroid - hitPoint;
    float centerDist = length(toNode);
    vec3 dir = (centerDist > 1e-4) ? (toNode / centerDist) : hitNormal;

    // Surface normal orientation: clamp to small positive floor to allow grazing lights
    float cosTheta = max(dot(hitNormal, dir), 0.05);

    // Light orientation bounding cone weighting
    float coneCos = node.bboxMax.w;
    float cosCone = 1.0;
    if (coneCos > -0.99) {
        vec3 coneAxis = node.coneAxis.xyz;
        float cosLightDir = dot(coneAxis, -dir);
        cosCone = clamp((cosLightDir - coneCos) / max(1.0 - coneCos, 1e-3), 0.05, 1.0);
    }

    return (flux * cosTheta * cosCone) / distSq;
}

// O(log N) stochastic descent down the 3D Light Tree (FEAT-02)
#define SAMPLE_LIGHT_TREE(numLights, hitPoint, hitNormal, seed, outIdx, outPdf) \
    do { \
        if ((numLights) <= 1u) { \
            outIdx = 0u; \
            outPdf = 1.0; \
        } else { \
            uint nodeIdx_ = 0u; \
            float pathProb_ = 1.0; \
            for (uint depth_ = 0u; depth_ < 32u; ++depth_) { \
                LightTreeNode node_ = lightNodes[nodeIdx_]; \
                if (node_.children.z == 1u /* isLeaf */) { \
                    outIdx = node_.children.x; \
                    break; \
                } \
                uint leftIdx_ = node_.children.x; \
                uint rightIdx_ = node_.children.y; \
                float impLeft_ = evalLightTreeNodeImportance(hitPoint, hitNormal, lightNodes[leftIdx_]); \
                float impRight_ = evalLightTreeNodeImportance(hitPoint, hitNormal, lightNodes[rightIdx_]); \
                float totalImp_ = impLeft_ + impRight_; \
                if (totalImp_ <= 1e-12) { \
                    impLeft_ = 0.5; \
                    totalImp_ = 1.0; \
                } \
                float probLeft_ = clamp(impLeft_ / totalImp_, 0.001, 0.999); \
                if (randFloat(seed) < probLeft_) { \
                    nodeIdx_ = leftIdx_; \
                    pathProb_ *= probLeft_; \
                } else { \
                    nodeIdx_ = rightIdx_; \
                    pathProb_ *= (1.0 - probLeft_); \
                } \
            } \
            if (lightNodes[nodeIdx_].children.z == 1u) { \
                outIdx = lightNodes[nodeIdx_].children.x; \
            } \
            outPdf = max(pathProb_, 1e-6); \
        } \
    } while(false)

// 32-bit Octahedral normal/direction encoding (Cigolle et al.)
vec2 octSign(vec2 v) {
    return vec2((v.x >= 0.0) ? 1.0 : -1.0, (v.y >= 0.0) ? 1.0 : -1.0);
}

vec2 octEncode(vec3 v) {
    v /= (abs(v.x) + abs(v.y) + abs(v.z));
    return (v.z >= 0.0) ? v.xy : (vec2(1.0) - abs(v.yx)) * octSign(v.xy);
}

vec3 octDecode(vec2 f) {
    vec3 v = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    if (v.z < 0.0) {
        v.xy = (vec2(1.0) - abs(v.yx)) * octSign(v.xy);
    }
    return normalize(v);
}

uint packOct32(vec3 v) {
    return packSnorm2x16(octEncode(normalize(v)));
}

vec3 unpackOct32(uint p) {
    return octDecode(unpackSnorm2x16(p));
}

// Classify 3D ray direction into 8 octants based on sign
uint getDirectionalOctant(vec3 dir) {
    return (dir.x >= 0.0 ? 1u : 0u) | (dir.y >= 0.0 ? 2u : 0u) | (dir.z >= 0.0 ? 4u : 0u);
}

// Build orthonormal basis (b1, b2) from unit normal n (Duff et al.)
void buildOrthonormalBasis(vec3 n, out vec3 b1, out vec3 b2) {
    float sign = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (sign + n.z);
    float b = n.x * n.y * a;
    b1 = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = vec3(b, sign + n.y * n.y * a, -n.y);
}

uint packTangentAngle13(vec3 normal, vec3 tangent) {
    vec3 b1, b2;
    buildOrthonormalBasis(normal, b1, b2);
    float x = dot(tangent, b1);
    float y = dot(tangent, b2);
    float angle = atan(y, x);
    float u = (angle + 3.141592653589793) * (8192.0 / 6.283185307179586);
    return uint(clamp(u, 0.0, 8191.0));
}

vec3 unpackTangentAngle13(vec3 normal, uint code) {
    vec3 b1, b2;
    buildOrthonormalBasis(normal, b1, b2);
    float angle = (float(code) + 0.5) * (6.283185307179586 / 8192.0) - 3.141592653589793;
    return normalize(b1 * cos(angle) + b2 * sin(angle));
}

// 16-byte packed ray geometry (read by intersect & shade)
struct RayGeometry {
    vec4 originPackedDir; // xyz: origin, w: uintBitsToFloat(packOct32(direction)) (16 bytes)
};

// 16-byte packed pre-interpolated ray hit (written by intersect/classify, read by shade)
struct RayHit {
    vec4 hitData0;
    // x: hitT (float)
    // y: uintBitsToFloat(packOct32(hitNormal))
    // z: uintBitsToFloat(packHalf2x16(hitUv))
    // w: uintBitsToFloat(packedInfo)
    // packedInfo:
    // bits 0..15: matId (16 bits)
    // bits 16..17: hitType (2 bits: 0: triangle, 1: sphere, 2: miss)
    // bit 18: tanSign (1 bit: 0: positive +1.0, 1: negative -1.0)
    // bits 19..31: tanAngle13 (13 bits: angle in [0, 8191])
};

RayHit packRayHit(float hitT, uint matId, uint hitType, vec3 hitNormal, vec2 hitUv, vec3 geomTan, float tanSign) {
    RayHit hit;
    uint tSignBit = (tanSign < 0.0) ? 1u : 0u;
    uint angleCode = packTangentAngle13(hitNormal, geomTan);
    uint packedInfo = (matId & 0xFFFFu) | ((hitType & 0x3u) << 16u) | (tSignBit << 18u) | ((angleCode & 0x1FFFu) << 19u);
    hit.hitData0 = vec4(hitT,
                        uintBitsToFloat(packOct32(hitNormal)),
                        uintBitsToFloat(packHalf2x16(hitUv)),
                        uintBitsToFloat(packedInfo));
    return hit;
}

void unpackRayHit(RayHit hit, out float hitT, out uint matId, out uint hitType, out vec3 hitNormal, out vec2 hitUv, out vec3 geomTan, out float tanSign) {
    hitT = hit.hitData0.x;
    hitNormal = unpackOct32(floatBitsToUint(hit.hitData0.y));
    hitUv = unpackHalf2x16(floatBitsToUint(hit.hitData0.z));
    uint packedInfo = floatBitsToUint(hit.hitData0.w);
    matId = packedInfo & 0xFFFFu;
    hitType = (packedInfo >> 16u) & 0x3u;
    tanSign = ((packedInfo >> 18u) & 1u) != 0u ? -1.0 : 1.0;
    geomTan = unpackTangentAngle13(hitNormal, (packedInfo >> 19u) & 0x1FFFu);
}

// 16-byte packed ray state (read/written by shade, NEVER touched by intersect)
struct RayState {
    uvec4 stateData;
    // x: packHalf2x16(throughput.rg)
    // y: packHalf2x16(vec2(throughput.b, 0.0)) | (flags << 16u)
    // z: seed
    // w: pixelIndex | specularFlag
};

#define SPECULAR_FLAG_BIT             (1u << 31)
#define CAUSTIC_APPLIED_BIT           (1u << 30)
#define PIXEL_INDEX_MASK              (0x3FFFFFFFu)

RayState packRayState(f16vec3 throughput, uint seed, uint pixelIndex, bool isSpecularPath, bool causticApplied) {
    RayState s;
    s.stateData.x = packHalf2x16(vec2(throughput.rg));
    uint flags = (isSpecularPath ? 1u : 0u) | (causticApplied ? 2u : 0u);
    s.stateData.y = (packHalf2x16(vec2(throughput.b, 0.0)) & 0xFFFFu) | (flags << 16u);
    s.stateData.z = seed;
    s.stateData.w = (pixelIndex & PIXEL_INDEX_MASK) | (isSpecularPath ? SPECULAR_FLAG_BIT : 0u) | (causticApplied ? CAUSTIC_APPLIED_BIT : 0u);
    return s;
}

void unpackRayState(RayState s, out f16vec3 throughput, out uint seed, out uint pixelIndex, out bool isSpecularPath, out bool causticApplied) {
    vec2 rg = unpackHalf2x16(s.stateData.x);
    vec2 bz = unpackHalf2x16(s.stateData.y & 0xFFFFu);
    throughput = f16vec3(rg.x, rg.y, bz.x);
    seed = s.stateData.z;
    uint rawPixel = s.stateData.w;
    pixelIndex = rawPixel & PIXEL_INDEX_MASK;
    isSpecularPath = (rawPixel & SPECULAR_FLAG_BIT) != 0u;
    causticApplied = (rawPixel & CAUSTIC_APPLIED_BIT) != 0u;
}

#define MATERIAL_ARCHETYPE_STANDARD   0u
#define MATERIAL_ARCHETYPE_COMPLEX    1u
#define MATERIAL_ARCHETYPE_DIELECTRIC 2u
#define MATERIAL_ARCHETYPE_EMISSIVE   3u
#define NUM_MATERIAL_ARCHETYPES       4u

// Backward compatibility aliases
#define MATERIAL_ARCHETYPE_DIFFUSE    0u
#define MATERIAL_ARCHETYPE_CONDUCTOR  0u
#define MATERIAL_ARCHETYPE_ALPHAMASK  0u

#define MATERIAL_TYPE_MASK               0x000000FFu
#define MATERIAL_FLAG_PROCEDURAL_TERRAIN (1u << 9)
#define MATERIAL_FLAG_PROCEDURAL_WATER   (1u << 11)
#define MATERIAL_FLAG_PROCEDURAL_PUDDLE  (1u << 12)
#define MATERIAL_FLAG_PROCEDURAL_HOLO    (1u << 13)
#define MATERIAL_FLAG_HOLO_VIDEO         (1u << 14)

uint getMaterialArchetype(Material mat) {
    // 1. Pure emissive mesh lights: pure emitters without scattering BSDF
    if ((mat.type & 0xFFu) == 3u /* MATERIAL_EMISSIVE */ ||
        (length(mat.emissive.rgb) > 0.1 && mat.albedoTex == 0u && length(mat.albedo.rgb) < 0.05 && mat.metallic < 0.01 && mat.transmission < 0.01)) {
        return MATERIAL_ARCHETYPE_EMISSIVE;
    }
    // 2. Multi-layer complex PBR (Clearcoat on top of substrate, Sheen, Anisotropy, or Iridescence)
    if (mat.clearcoat > 0.001 || mat.clearcoatTex > 0u || length(mat.sheenColor) > 0.001 || mat.sheenTex > 0u ||
        mat.anisotropyStrength > 0.001 || mat.anisotropyTex > 0u || mat.iridescence > 0.001) {
        return MATERIAL_ARCHETYPE_COMPLEX;
    }
    // 3. Pure dielectric transmission / refraction / glass / dispersion / procedural water
    if (mat.transmission > 0.001 || (mat.type & 0xFFu) == 2u /* MATERIAL_DIELECTRIC */ || (mat.type & (1u << 11)) != 0u || mat.dispersion > 0.001) {
        return MATERIAL_ARCHETYPE_DIELECTRIC;
    }
    // 4. Standard PBR (Dielectric diffuse base + GGX specular dual-lobe reflection, and Metallic conductors)
    return MATERIAL_ARCHETYPE_STANDARD;
}

// 32-byte clean aligned packed shadow ray
struct PackedShadowRay {
    vec4 originDist;   // xyz: origin, w: lightDist (16 bytes)
    uvec4 dirPixelRad; // x: packOct32(direction), y: pixelIndex, z: packHalf2x16(radiance.rg), w: packHalf2x16(vec2(radiance.b, 0.0)) (16 bytes)
};

// Legacy 96-byte ray payload preserved for compatibility
struct RayPayload {
    vec4 origin;     // xyz: origin, w: uintBitsToFloat(flags: active=1)
    vec4 direction;  // xyz: direction, w: uintBitsToFloat(bounce)
    vec4 throughput; // rgb: throughput, w: uintBitsToFloat(seed)
    vec4 radiance;   // rgb: accumulated radiance, w: uintBitsToFloat(pixelIndex)
    vec4 hitNormal;  // xyz: shading normal, w: uintBitsToFloat(materialId)
    vec4 hitExtra;   // xy: uv, z: hit t, w: uintBitsToFloat(frontFace | hitType)
};

struct DispatchCommand {
    uint x;
    uint y;
    uint z;
    uint pad;
};

struct DGCCommand {
    uint pipelineIndex; // Token 0: EXECUTION_SET_EXT
    uint x;             // Token 1: DISPATCH_EXT (strictly last!)
    uint y;
    uint z;
};

struct BounceDispatch {
    DispatchCommand shadeDispatch;     // Offset 0
    DispatchCommand shadowDispatch;    // Offset 16
    DispatchCommand intersectDispatch; // Offset 32
};

struct BounceDGC {
    DGCCommand shadeCmd;     // Offset 0
    DGCCommand shadowCmd;    // Offset 16
    DGCCommand intersectCmd; // Offset 32
};

struct BounceMaterialDispatch {
    DispatchCommand shadeDiffuse;      // Offset 0
    DispatchCommand shadeDielectric;   // Offset 16
    DispatchCommand shadeConductor;    // Offset 32
    DispatchCommand shadeComplex;      // Offset 48
    DispatchCommand shadeEmissive;     // Offset 64
    DispatchCommand shadeAlphamask;    // Offset 80
    DispatchCommand shadowDispatch;    // Offset 96
    DispatchCommand intersectDispatch; // Offset 112
};

struct BounceMaterialDGC {
    DGCCommand shadeDiffuse;      // Offset 0
    DGCCommand shadeDielectric;   // Offset 16
    DGCCommand shadeConductor;    // Offset 32
    DGCCommand shadeComplex;      // Offset 48
    DGCCommand shadeEmissive;     // Offset 64
    DGCCommand shadeAlphamask;    // Offset 80
    DGCCommand shadowCmd;         // Offset 96
    DGCCommand intersectCmd;      // Offset 112
};

struct VkDispatchIndirectCommand {
    uint x;
    uint y;
    uint z;
};

// PCG Random Number Generator
uint pcg_hash(inout uint seed) {
    seed = seed * 747796405u + 2891336453u;
    uint word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    return (word >> 22u) ^ word;
}

float randFloat(inout uint seed) {
    return float(pcg_hash(seed)) / 4294967296.0;
}

vec2 randVec2(inout uint seed) {
    return vec2(randFloat(seed), randFloat(seed));
}

vec3 randVec3(inout uint seed) {
    return vec3(randFloat(seed), randFloat(seed), randFloat(seed));
}

// Clean floating-point radiance accumulation.
vec3 addFp16Stochastic(vec3 accum, vec3 val, inout uint seed) {
    if (isnan(val.r) || isnan(val.g) || isnan(val.b) || isinf(val.r) || isinf(val.g) || isinf(val.b)) return accum;
    return accum + max(val, vec3(0.0));
}

vec2 directionToEquirectangular(vec3 dir) {
    float phi = atan(dir.z, dir.x);
    float theta = acos(clamp(dir.y, -1.0, 1.0));
    return vec2((phi + PI) / TWO_PI, theta / PI);
}

// 2D Morton Z-curve mapping with 2D Macro-Tile Workgroup Swizzling
ivec2 getPixelCoordsMorton8x4(uint linearIdx, uint width, uint height) {
    uint tilesX = (width + 7u) / 8u;
    uint tilesY = (height + 3u) / 4u;
    uint tileIdx = linearIdx / 32u;
    uint inTile = linearIdx % 32u;

    // Macro-Tile Swizzle: Cluster workgroups into 8x8 2D macro-tiles (64x32 pixels)
    // to maximize L0/L1/L2 cache locality during BVH traversal without host barriers
    uint TW = 8u;
    uint TH = 8u;
    uint chunkSize = 64u;
    uint stripSize = TW * tilesY;
    uint stripIdx = (stripSize > 0u) ? (tileIdx / stripSize) : 0u;
    uint inStrip = (stripSize > 0u) ? (tileIdx % stripSize) : 0u;
    uint fullChunksH = tilesY / TH;
    uint remainderStart = fullChunksH * chunkSize;

    uint tileX;
    uint tileY;
    if (inStrip < remainderStart) {
        uint chunkIdx = inStrip / chunkSize;
        uint inChunk = inStrip % chunkSize;
        uint localX = ((inChunk >> 0u) & 1u) | (((inChunk >> 2u) & 1u) << 1u) | (((inChunk >> 4u) & 1u) << 2u);
        uint localY = ((inChunk >> 1u) & 1u) | (((inChunk >> 3u) & 1u) << 1u) | (((inChunk >> 5u) & 1u) << 2u);
        tileX = stripIdx * TW + localX;
        tileY = chunkIdx * TH + localY;
    } else {
        uint rem = inStrip - remainderStart;
        tileX = stripIdx * TW + (rem % TW);
        tileY = fullChunksH * TH + (rem / TW);
    }

    if (tileX >= tilesX || tileY >= tilesY) {
        tileX = (tilesX > 0u) ? (tileIdx % tilesX) : 0u;
        tileY = (tilesX > 0u) ? (tileIdx / tilesX) : 0u;
    }

    // Intra-wave 2D Morton Z-curve mapping (8x4 pixels per 32-lane wave)
    uint mx = ((inTile >> 0u) & 1u) | (((inTile >> 2u) & 1u) << 1u) | (((inTile >> 4u) & 1u) << 2u);
    uint my = ((inTile >> 1u) & 1u) | (((inTile >> 3u) & 1u) << 1u);

    return ivec2(int(tileX * 8u + mx), int(tileY * 4u + my));
}

// 3D Morton encoding (10 bits per axis, 30-bit total code)
uint part1By2(uint x) {
    x &= 0x000003ffu;
    x = (x ^ (x << 16u)) & 0xff0000ffu;
    x = (x ^ (x <<  8u)) & 0x0300f00fu;
    x = (x ^ (x <<  4u)) & 0x030c30c3u;
    x = (x ^ (x <<  2u)) & 0x09249249u;
    return x;
}

uint morton3D_10bit(uvec3 v) {
    return (part1By2(v.z) << 2u) | (part1By2(v.y) << 1u) | part1By2(v.x);
}

// 2D Morton encoding for 8-bit coordinates (16-bit total code)
uint morton2D_8bit(uvec2 v) {
    uint x = v.x & 0x000000FFu;
    x = (x | (x << 4u)) & 0x0F0Fu;
    x = (x | (x << 2u)) & 0x3333u;
    x = (x | (x << 1u)) & 0x5555u;

    uint y = v.y & 0x000000FFu;
    y = (y | (y << 4u)) & 0x0F0Fu;
    y = (y | (y << 2u)) & 0x3333u;
    y = (y | (y << 1u)) & 0x5555u;

    return x | (y << 1u);
}

// Cosine-weighted hemisphere sampling
vec3 sampleCosineHemisphere(vec3 normal, inout uint seed) {
    vec2 r = randVec2(seed);
    float phi = TWO_PI * r.x;
    float cosTheta = sqrt(r.y);
    float sinTheta = sqrt(max(0.0, 1.0 - r.y));

    vec3 tangent = abs(normal.x) > 0.1 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    tangent = normalize(cross(normal, tangent));
    vec3 bitangent = cross(normal, tangent);

    return normalize(tangent * (cos(phi) * sinTheta) +
                     bitangent * (sin(phi) * sinTheta) +
                     normal * cosTheta);
}

// Direct Coherent Cosine-Weighted Hemisphere Sampling (Xiang et al. 2023)
// Interleaved Subgroup Grouping: Partitions 32-lane wave into 8 disjoint groups of 4 lanes (or 4 of 8)
// 2D Spatial Stride Leader Lane Computation for On-Chip Subgroup Shuffle
// Uses a 2D spatial stride (min pixel dist >= 2.23) to eliminate spatial correlation in 3x3 filter windows
// while preserving SIMD execution coherence during BVH traversal.
uint getCoherentLeaderLane(uint clusterSize, uint frameIndex) {
    uint k = (clusterSize == 8u) ? 8u : 4u;
    uint lane = gl_SubgroupInvocationID;
    uint lane32 = lane & 31u;
    uint waveBase = lane & ~31u;
    uint row = lane32 >> 3u;
    uint col = lane32 & 7u;

    uvec4 activeLanes = subgroupBallot(true);

    uint leaderLane = lane;
    if (k == 4u) {
        // 8 groups of 4 lanes. Spatial stride: (col + row * 5) % 8
        uint g = (col + row * 5u) & 7u;
        uint rot = frameIndex & 3u;
        const uint colShifts[4] = uint[4](0u, 3u, 6u, 1u);
        uint preferred = waveBase + ((g + colShifts[rot]) & 7u) + (rot << 3u);

        bool prefActive = ((activeLanes[preferred >> 5u] >> (preferred & 31u)) & 1u) != 0u;
        if (prefActive) {
            leaderLane = preferred;
        } else {
            leaderLane = preferred;
            for (uint i = 1u; i < 4u; ++i) {
                uint tryRot = (rot + i) & 3u;
                uint cand = waveBase + ((g + colShifts[tryRot]) & 7u) + (tryRot << 3u);
                if (((activeLanes[cand >> 5u] >> (cand & 31u)) & 1u) != 0u) {
                    leaderLane = cand;
                    break;
                }
            }
        }
    } else {
        // 4 groups of 8 lanes. Spatial stride: (col + row * 3) % 4
        uint g = (col + row * 3u) & 3u;
        uint rot = frameIndex & 7u;
        uint rotRow = rot >> 1u;
        uint rotColBase = (g + (rotRow * 1u)) & 3u;
        uint rotCol = rotColBase + ((rot & 1u) << 2u);
        uint preferred = waveBase + (rotRow << 3u) + rotCol;

        bool prefActive = ((activeLanes[preferred >> 5u] >> (preferred & 31u)) & 1u) != 0u;
        if (prefActive) {
            leaderLane = preferred;
        } else {
            leaderLane = preferred;
            for (uint i = 1u; i < 8u; ++i) {
                uint tryRot = (rot + i) & 7u;
                uint tryRow = tryRot >> 1u;
                uint tryCol = ((g + (tryRow * 1u)) & 3u) + ((tryRot & 1u) << 2u);
                uint cand = waveBase + (tryRow << 3u) + tryCol;
                if (((activeLanes[cand >> 5u] >> (cand & 31u)) & 1u) != 0u) {
                    leaderLane = cand;
                    break;
                }
            }
        }
    }
    return leaderLane;
}

// Coherent Cosine Hemisphere Sampling via On-Chip Subgroup Shuffle (Xiang et al. 2023)
// Uses Duff et al. continuous orthonormal basis to eliminate tangent frame boundary flips.
vec3 sampleCosineHemisphereCoherent(vec3 normal, inout uint seed, uint clusterSize, uint frameIndex) {
    uint leaderLane = getCoherentLeaderLane(clusterSize, frameIndex);

    // Advance PRNG for every lane (guarantees cross-bounce randomness)
    vec2 rLocal = randVec2(seed);

    // Broadcast leader's tangent-space direction sample across cluster
    vec2 r;
    r.x = subgroupShuffle(rLocal.x, leaderLane);
    r.y = subgroupShuffle(rLocal.y, leaderLane);

    // Continuous orthonormal basis projection (Duff et al.)
    float phi = TWO_PI * r.x;
    float cosTheta = sqrt(r.y);
    float sinTheta = sqrt(max(0.0, 1.0 - r.y));

    vec3 b1, b2;
    buildOrthonormalBasis(normal, b1, b2);

    return normalize(b1 * (cos(phi) * sinTheta) +
                     b2 * (sin(phi) * sinTheta) +
                     normal * cosTheta);
}

// Coherent PRNG Float Sample via Subgroup Shuffle (for coherent Fresnel reflection/refraction coin-flips)
float randFloatCoherent(inout uint seed, uint clusterSize, uint frameIndex) {
    uint leaderLane = getCoherentLeaderLane(clusterSize, frameIndex);
    float rLocal = randFloat(seed);
    return subgroupShuffle(rLocal, leaderLane);
}

// Schlick's approximation for Fresnel reflectance
float fresnelSchlick(float cosTheta, float refIdx) {
    float r0 = (1.0 - refIdx) / (1.0 + refIdx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Physically correct dielectric Fresnel supporting internal and external boundaries
float fresnelDielectric(float cosThetaI, float eta) {
    float sinThetaI2 = max(0.0, 1.0 - cosThetaI * cosThetaI);
    float sinThetaT2 = (eta * eta) * sinThetaI2;
    if (sinThetaT2 >= 1.0) {
        return 1.0; // Total Internal Reflection (TIR)
    }
    float cosT = sqrt(max(0.0, 1.0 - sinThetaT2));
    float r0 = (1.0 - eta) / (1.0 + eta);
    r0 = r0 * r0;
    float cosEval = (eta > 1.0) ? cosT : cosThetaI;
    float x = clamp(1.0 - cosEval, 0.0, 1.0);
    return r0 + (1.0 - r0) * (x * x * x * x * x);
}

// Fresnel-Schlick approximation with vector F0 (Cook-Torrance PBR)
vec3 fresnelSchlickVec(float cosTheta, vec3 F0) {
    return F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

f16vec3 fresnelSchlickVec(float16_t cosTheta, f16vec3 F0) {
    float16_t omc = clamp(float16_t(1.0) - cosTheta, float16_t(0.0), float16_t(1.0));
    float16_t omc2 = omc * omc;
    float16_t omc5 = omc2 * omc2 * omc;
    return F0 + (f16vec3(1.0) - F0) * omc5;
}

// GGX / Trowbridge-Reitz Normal Distribution Function D
float distributionGGX(float NdotH, float alpha) {
    float a2 = max(alpha * alpha, 1e-6);
    float d = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / (PI * d * d);
}

// Smith Joint Correlated Visibility Function V = G / (4 * NdotL * NdotV)
float visibilitySmithGGXCorrelated(float NdotL, float NdotV, float alpha) {
    float a2 = max(alpha * alpha, 1e-6);
    float gv = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float gl = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-6);
}

// Importance sample GGX microfacet distribution for half-vector H in world space
vec3 sampleGGX(vec3 N, float alpha, inout uint seed) {
    vec2 xi = randVec2(seed);
    float a2 = max(alpha * alpha, 1e-6);
    float phi = TWO_PI * xi.x;
    float cosTheta = sqrt(clamp((1.0 - xi.y) / max(1.0 + (a2 - 1.0) * xi.y, 1e-7), 0.0, 1.0));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    vec3 H_local = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    vec3 up = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);

    return normalize(tangent * H_local.x + bitangent * H_local.y + N * H_local.z);
}

// Visible Normal Distribution Function (VNDF) Sampling (Dupuy & Heitz 2023)
vec3 sampleVNDF_GGX(vec3 V_local, float alpha_x, float alpha_y, inout uint seed) {
    vec2 u = randVec2(seed);
    vec3 Vh = normalize(vec3(alpha_x * V_local.x, alpha_y * V_local.y, V_local.z));
    float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
    vec3 T1 = lensq > 1e-7 ? vec3(-Vh.y, Vh.x, 0.0) / sqrt(lensq) : vec3(1.0, 0.0, 0.0);
    vec3 T2 = cross(Vh, T1);
    float r = sqrt(u.x);
    float phi = TWO_PI * u.y;
    float t1 = r * cos(phi);
    float t2 = r * sin(phi);
    float s = 0.5 * (1.0 + Vh.z);
    t2 = (1.0 - s) * sqrt(max(0.0, 1.0 - t1 * t1)) + s * t2;
    vec3 Nh = t1 * T1 + t2 * T2 + sqrt(max(0.0, 1.0 - t1 * t1 - t2 * t2)) * Vh;
    return normalize(vec3(alpha_x * Nh.x, alpha_y * Nh.y, max(0.0, Nh.z)));
}

// Height-correlated Smith G2 / G1 ratio for exact VNDF Monte Carlo estimator weighting
float ratioSmithG2OverG1(float NdotV, float NdotL, float alpha) {
    float a2 = max(alpha * alpha, 1e-6);
    float sqV = sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float sqL = sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    float G1_V_denom = NdotV + sqV;
    float num = NdotL * G1_V_denom;
    float den = NdotL * sqV + NdotV * sqL;
    return clamp(num / max(den, 1e-6), 0.0, 1.0);
}

// Anisotropic GGX Normal Distribution Function D
float distributionAnisotropicGGX(float TdotH, float BdotH, float NdotH, float ax, float ay) {
    float d = (TdotH * TdotH) / max(ax * ax, 1e-6) + (BdotH * BdotH) / max(ay * ay, 1e-6) + NdotH * NdotH;
    return 1.0 / max(PI * ax * ay * d * d, 1e-6);
}

// Sample Anisotropic GGX microfacet normal
vec3 sampleAnisotropicGGX(vec3 N, vec3 T, vec3 B, float ax, float ay, inout uint seed) {
    vec2 xi = randVec2(seed);
    float phi = atan(ay * sin(TWO_PI * xi.x), ax * cos(TWO_PI * xi.x));
    if (phi < 0.0) phi += TWO_PI;
    float cosTheta = sqrt(clamp((1.0 - xi.y) / max(xi.y * (ax * cos(phi) * ax * cos(phi) + ay * sin(phi) * ay * sin(phi) - 1.0) + 1.0, 1e-7), 0.0, 1.0));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    vec3 H_local = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    return normalize(T * H_local.x + B * H_local.y + N * H_local.z);
}

// Coherent GGX Microfacet Sampling via Subgroup Shuffle (Xiang et al. 2023)
vec3 sampleGGXCoherent(vec3 N, float alpha, inout uint seed, uint clusterSize, uint frameIndex) {
    uint leaderLane = getCoherentLeaderLane(clusterSize, frameIndex);
    vec2 rLocal = randVec2(seed);

    vec2 xi;
    xi.x = subgroupShuffle(rLocal.x, leaderLane);
    xi.y = subgroupShuffle(rLocal.y, leaderLane);

    float a2 = max(alpha * alpha, 1e-6);
    float phi = TWO_PI * xi.x;
    float cosTheta = sqrt(clamp((1.0 - xi.y) / max(1.0 + (a2 - 1.0) * xi.y, 1e-7), 0.0, 1.0));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    vec3 H_local = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);

    vec3 b1, b2;
    buildOrthonormalBasis(N, b1, b2);

    return normalize(b1 * H_local.x + b2 * H_local.y + N * H_local.z);
}

// Coherent Anisotropic GGX Microfacet Sampling via Subgroup Shuffle
vec3 sampleAnisotropicGGXCoherent(vec3 N, vec3 T, vec3 B, float ax, float ay, inout uint seed, uint clusterSize, uint frameIndex) {
    uint leaderLane = getCoherentLeaderLane(clusterSize, frameIndex);
    vec2 rLocal = randVec2(seed);

    vec2 xi;
    xi.x = subgroupShuffle(rLocal.x, leaderLane);
    xi.y = subgroupShuffle(rLocal.y, leaderLane);

    float phi = atan(ay * sin(TWO_PI * xi.x), ax * cos(TWO_PI * xi.x));
    if (phi < 0.0) phi += TWO_PI;
    float cosTheta = sqrt(clamp((1.0 - xi.y) / max(xi.y * (ax * cos(phi) * ax * cos(phi) + ay * sin(phi) * ay * sin(phi) - 1.0) + 1.0, 1e-7), 0.0, 1.0));
    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));
    vec3 H_local = vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
    return normalize(T * H_local.x + B * H_local.y + N * H_local.z);
}

// Charlie Micro-Fiber Sheen Distribution (glTF KHR_materials_sheen)
float evalSheenCharlie(float NdotH, float sheenRoughness) {
    float invR = 1.0 / max(sheenRoughness, 0.01);
    float cos2 = NdotH * NdotH;
    float sin2 = max(1.0 - cos2, 1e-6);
    return (2.0 + invR) * pow(sin2, invR * 0.5) / TWO_PI;
}

// Airy Thin-Film Iridescence Phase Interference (glTF KHR_materials_iridescence)
vec3 evalThinFilmIridescence(float cosTheta, float iridIor, float thicknessNm) {
    float sinTheta2 = 1.0 - cosTheta * cosTheta;
    float cosThetaT2 = 1.0 - sinTheta2 / max(iridIor * iridIor, 1e-4);
    float cosThetaT = sqrt(max(0.0, cosThetaT2));
    float opd = 2.0 * iridIor * thicknessNm * cosThetaT; // Optical Path Difference in nm
    // Constructive/destructive interference for RGB center wavelengths (650nm, 530nm, 460nm)
    vec3 phase = (TWO_PI * opd) / vec3(650.0, 530.0, 460.0);
    return clamp(0.5 + 0.5 * cos(phase), 0.0, 1.0);
}

// Evaluate combined BSDF PDF for Multiple Importance Sampling (MIS)
float evalBSDFPdf(vec3 V, vec3 L, vec3 N, vec3 Nc, float alphaRoughness, float clearcoatAlpha, float clearcoatProb, float baseSpecProb, float clearcoat) {
    float NdotL = dot(N, L);
    if (NdotL <= 0.0) return 0.0;

    vec3 H = normalize(V + L);
    float NdotH = clamp(dot(N, H), 0.0, 1.0);
    float VdotH = clamp(dot(V, H), 0.0, 1.0);

    float diffPdf = NdotL * INV_PI;

    float D = distributionGGX(NdotH, alphaRoughness);
    float specPdf = (D * NdotH) / max(4.0 * VdotH, 1e-4);

    float clearcoatPdf = 0.0;
    if (clearcoat > 0.001) {
        float NcDotL = dot(Nc, L);
        if (NcDotL > 0.0) {
            float NcDotH = clamp(dot(Nc, H), 0.0, 1.0);
            float VDotHc = clamp(dot(V, H), 0.0, 1.0);
            float Dc = distributionGGX(NcDotH, clearcoatAlpha);
            clearcoatPdf = (Dc * NcDotH) / max(4.0 * VDotHc, 1e-4);
        }
    }

    float diffProb = max(1.0 - clearcoatProb - baseSpecProb, 0.0);
    return diffProb * diffPdf + baseSpecProb * specPdf + clearcoatProb * clearcoatPdf;
}


// Procedural sphere ray intersection
bool intersectSphere(vec3 origin, vec3 dir, Sphere sphere, float tMin, float tMax, out float outT, out vec3 outNormal) {
    vec3 oc = origin - sphere.centerRadius.xyz;
    float radius = sphere.centerRadius.w;
    float a = dot(dir, dir);
    float halfB = dot(oc, dir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = halfB * halfB - a * c;

    if (discriminant < 0.0) return false;
    float sqrtd = sqrt(discriminant);

    float root = (-halfB - sqrtd) / a;
    if (root < tMin || root > tMax) {
        root = (-halfB + sqrtd) / a;
        if (root < tMin || root > tMax) return false;
    }

    outT = root;
    outNormal = (origin + root * dir - sphere.centerRadius.xyz) / radius;
    return true;
}



// Chromaticity-preserving luminance clamping for indirect / secondary bounces
vec3 clampIndirectRadiance(vec3 rad, float maxLum) {
    if (maxLum <= 0.0) return rad;
    float lum = dot(rad, vec3(0.2126, 0.7152, 0.0722));
    if (lum > maxLum) {
        return rad * (maxLum / lum);
    }
    return rad;
}

#ifdef GL_EXT_shader_explicit_arithmetic_types_float16
f16vec3 clampIndirectRadiance(f16vec3 rad, float maxLum) {
    if (maxLum <= 0.0) return rad;
    float lum = float(dot(rad, f16vec3(0.2126, 0.7152, 0.0722)));
    if (lum > maxLum) {
        return rad * float16_t(maxLum / lum);
    }
    return rad;
}
#endif

// Progressive depth-aware Russian Roulette for unbiased path termination
bool applyRussianRoulette(inout vec3 throughput, float pathLum, uint bounce, inout uint seed) {
    if (pathLum < 0.001) {
        return true;
    }
    if (bounce >= 1u) {
        float depthDecay = (bounce >= 2u) ? pow(0.85, float(bounce - 1u)) : 1.0;
        float p = clamp(pathLum * depthDecay, 0.05, 0.95);
        if (randFloat(seed) > p) {
            return true;
        }
        throughput /= p;
    }
    return false;
}

#ifdef GL_EXT_shader_explicit_arithmetic_types_float16
bool applyRussianRoulette16(inout f16vec3 throughput, float pathLum, uint bounce, inout uint seed) {
    if (pathLum < 0.001) {
        return true;
    }
    if (bounce >= 1u) {
        float depthDecay = (bounce >= 2u) ? pow(0.85, float(bounce - 1u)) : 1.0;
        float p = clamp(pathLum * depthDecay, 0.05, 0.95);
        if (randFloat(seed) > p) {
            return true;
        }
        throughput /= float16_t(p);
    }
    return false;
}
#endif

#endif // WAVEFRONT_COMMON_GLSL

