#ifndef WAVEFRONT_COMMON_GLSL
#define WAVEFRONT_COMMON_GLSL

#extension GL_EXT_ray_query : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_control_flow_attributes : enable

#include "nrc_common.glsl"

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692
#define INV_PI 0.31830988618379067154
#define EPSILON 0.0005

struct Vertex {
    vec4 position; // xyz: pos, w: u
    vec4 normal;   // xyz: norm, w: v
    vec4 tangent;  // xyz: tangent, w: sign
};

struct Triangle {
    Vertex v0;
    Vertex v1;
    Vertex v2;
    uint materialId;
    uint padding[3];
};

struct Sphere {
    vec4 centerRadius; // xyz: center, w: radius
    uint materialId;
    uint padding[3];
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

// 16-byte packed ray geometry (read by intersect & shade)
struct RayGeometry {
    vec4 originPackedDir; // xyz: origin, w: uintBitsToFloat(packOct32(direction)) (16 bytes)
};

// 16-byte cache-line aligned ray hit (written by intersect/classify, read by shade)
struct RayHit {
    vec4 hitData;
    // x: hitT (float)
    // y: uintBitsToFloat(primitiveIndex)
    // z: uintBitsToFloat(packHalf2x16(barycentrics))
    // w: uintBitsToFloat(hitType) (0: triangle, 1: sphere, 2: miss)
};

// 32-byte cache-line aligned ray state (read/written by shade, NEVER touched by intersect)
struct RayState {
    vec4 throughputSeed;  // rgb: throughput, w: uintBitsToFloat(seed) (16 bytes)
    vec4 radiancePixel;   // rgb: accumRadiance, w: uintBitsToFloat(pixelIndex) (16 bytes)
};

#define MATERIAL_ARCHETYPE_DIFFUSE    0u
#define MATERIAL_ARCHETYPE_DIELECTRIC 1u
#define MATERIAL_ARCHETYPE_CONDUCTOR  2u
#define MATERIAL_ARCHETYPE_COMPLEX    3u
#define MATERIAL_ARCHETYPE_EMISSIVE   4u
#define MATERIAL_ARCHETYPE_ALPHAMASK  5u
#define NUM_MATERIAL_ARCHETYPES       6u

uint getMaterialArchetype(Material mat) {
    // 1. Alpha cutout passthrough: only true alpha-masked surfaces with textures
    if (mat.alphaMode == 1u /* ALPHA_MODE_MASK */ && mat.albedoTex > 0u) {
        return MATERIAL_ARCHETYPE_ALPHAMASK;
    }
    // 2. Pure emissive mesh lights: pure emitters without scattering BSDF
    if (mat.type == 3u /* MATERIAL_EMISSIVE */ ||
        (length(mat.emissive.rgb) > 0.1 && mat.albedoTex == 0u && length(mat.albedo.rgb) < 0.05 && mat.metallic < 0.01 && mat.transmission < 0.01)) {
        return MATERIAL_ARCHETYPE_EMISSIVE;
    }
    // 3. Multi-layer complex PBR (Clearcoat on top of substrate, or Sheen)
    if (mat.clearcoat > 0.001 || mat.clearcoatTex > 0u || length(mat.sheenColor) > 0.001 || mat.sheenTex > 0u) {
        return MATERIAL_ARCHETYPE_COMPLEX;
    }
    // 4. Pure dielectric transmission / refraction / glass / dispersion
    if (mat.transmission > 0.001 || mat.type == 2u /* MATERIAL_DIELECTRIC */ || mat.dispersion > 0.001) {
        return MATERIAL_ARCHETYPE_DIELECTRIC;
    }
    // 5. Metallic conductors (GGX microfacet specular reflection, anisotropy, iridescence)
    if (mat.type == 1u /* MATERIAL_METALLIC */ || mat.metallic > 0.5 || mat.anisotropyStrength > 0.001 || mat.iridescence > 0.001) {
        return MATERIAL_ARCHETYPE_CONDUCTOR;
    }
    // 6. Dielectric diffuse base + GGX specular dual-lobe PBR (plastics, wood, stone, cloth)
    return MATERIAL_ARCHETYPE_DIFFUSE;
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

vec2 directionToEquirectangular(vec3 dir) {
    float phi = atan(dir.z, dir.x);
    float theta = acos(clamp(dir.y, -1.0, 1.0));
    return vec2((phi + PI) / TWO_PI, theta / PI);
}

// 2D Morton Z-curve mapping for an 8x4 Wave32 block
ivec2 getPixelCoordsMorton8x4(uint linearIdx, uint width, uint height) {
    uint tilesX = (width + 7u) / 8u;
    uint tileIdx = linearIdx / 32u;
    uint inTile = linearIdx % 32u;
    uint tileX = tileIdx % tilesX;
    uint tileY = tileIdx / tilesX;

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

// Schlick's approximation for Fresnel reflectance
float fresnelSchlick(float cosTheta, float refIdx) {
    float r0 = (1.0 - refIdx) / (1.0 + refIdx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Fresnel-Schlick approximation with vector F0 (Cook-Torrance PBR)
vec3 fresnelSchlickVec(float cosTheta, vec3 F0) {
    return F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
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

// Möller-Trumbore ray-triangle intersection
bool intersectTriangle(vec3 origin, vec3 dir, Triangle tri, float tMin, float tMax, out float outT, out vec2 outBary) {
    vec3 v0 = tri.v0.position.xyz;
    vec3 v1 = tri.v1.position.xyz;
    vec3 v2 = tri.v2.position.xyz;

    vec3 edge1 = v1 - v0;
    vec3 edge2 = v2 - v0;
    vec3 pvec = cross(dir, edge2);
    float det = dot(edge1, pvec);

    if (abs(det) < 1e-7) return false;
    float invDet = 1.0 / det;

    vec3 tvec = origin - v0;
    float u = dot(tvec, pvec) * invDet;
    if (u < 0.0 || u > 1.0) return false;

    vec3 qvec = cross(tvec, edge1);
    float v = dot(dir, qvec) * invDet;
    if (v < 0.0 || u + v > 1.0) return false;

    float t = dot(edge2, qvec) * invDet;
    if (t < tMin || t > tMax) return false;

    outT = t;
    outBary = vec2(u, v);
    return true;
}

#endif // WAVEFRONT_COMMON_GLSL
