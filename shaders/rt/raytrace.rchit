#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : enable
#extension GL_EXT_nonuniform_qualifier : enable

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692
#define INV_PI 0.31830988618379067154
#define EPSILON 0.0005
#define MATERIAL_TYPE_MASK 0x000000FFu

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
hitAttributeEXT vec2 attribs;

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

// 64-byte cache-line aligned shading triangle (2 triangles per 128B RDNA 4 vector cache line)
struct Triangle {
    uint octNormal0;
    uint octNormal1;
    uint octNormal2;
    uint octTan0;

    uint octTan1;
    uint octTan2;
    uint tanSigns;
    uint materialId;

    float uv0x;
    float uv0y;
    float uv1x;
    float uv1y;

    float uv2x;
    float uv2y;
    uint padding0;
    uint reserved0;
};

vec3 getTriangleNormal(in Triangle tri, in vec2 bary) {
    float u = bary.x;
    float v = bary.y;
    float w = 1.0 - u - v;
    vec3 n0 = unpackOct32(tri.octNormal0);
    vec3 n1 = unpackOct32(tri.octNormal1);
    vec3 n2 = unpackOct32(tri.octNormal2);
    return normalize(w * n0 + u * n1 + v * n2);
}

vec2 getTriangleUV(in Triangle tri, in vec2 bary) {
    float u = bary.x;
    float v = bary.y;
    float w = 1.0 - u - v;
    vec2 uv0 = vec2(tri.uv0x, tri.uv0y);
    vec2 uv1 = vec2(tri.uv1x, tri.uv1y);
    vec2 uv2 = vec2(tri.uv2x, tri.uv2y);
    return w * uv0 + u * uv1 + v * uv2;
}

vec3 getTriangleTangent(in Triangle tri, in vec2 bary) {
    float u = bary.x;
    float v = bary.y;
    float w = 1.0 - u - v;
    vec3 t0 = unpackOct32(tri.octTan0);
    vec3 t1 = unpackOct32(tri.octTan1);
    vec3 t2 = unpackOct32(tri.octTan2);
    vec3 geomTan = w * t0 + u * t1 + v * t2;
    float len = length(geomTan);
    return len > 1e-4 ? (geomTan / len) : vec3(1.0, 0.0, 0.0);
}

float getTriangleTangentSign(in Triangle tri) {
    return ((tri.tanSigns & 1u) != 0u) ? 1.0 : -1.0;
}



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

    // Extended glTF 2.0 PBR Properties
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

// 128-byte record: one RDNA 4 vector cache line per light (no straddling fetches)
struct Light {
    vec4 position; // xyz: pos/corner, w: type
    vec4 emission; // rgb: color, w: area
    vec4 u;        // xyz: edge1, w: spot inner cos
    vec4 v;        // xyz: edge2, w: spot outer cos
    vec4 normal;   // xyz: normal/dir, w: padding
    vec4 sampling; // x: q, y: aliasIdx, z: pdf, w: flux
    vec4 pad0;    // cache-line padding (128 bytes total)
    vec4 pad1;    // cache-line padding (128 bytes total)
};

layout(binding = 1) uniform CameraUBO {
    mat4 viewInverse;
    mat4 projInverse;
    mat4 prevViewProj;
    vec4 position;
    vec4 viewParams;
    uint frameIndex;
    uint spp;
    uint maxBounces;
    uint flags;
    mat4 unjitteredViewProj;
    vec4 jitterOffset;
} ubo;

layout(std430, binding = 2) readonly buffer TrianglesBuffer {
    Triangle triangles[];
};

layout(std430, binding = 3) readonly buffer SpheresBuffer {
    Sphere spheres[];
};

layout(std430, binding = 4) readonly buffer MaterialsBuffer {
    Material materials[];
};

layout(std430, binding = 5) readonly buffer LightsBuffer {
    Light lights[];
};

layout(binding = 6) uniform accelerationStructureEXT topLevelAS;
layout(binding = 8) uniform sampler2D sceneTextures[512];

vec4 sampleSceneTexture(uint texId, vec2 uv) {
    uint idx = texId - 1u;
    return textureLod(sceneTextures[nonuniformEXT(idx)], uv, 0.0);
}
#define SAMPLE_SCENE_TEXTURE(texId, uv) sampleSceneTexture((texId), (uv))

struct InstanceGPU {
    uint firstTriangle;
    uint numOpaqueTriangles;
    uint materialOffset;
    uint flags;
    mat4 transform;
    mat4 invTransform;
};

struct LightTreeNode {
    vec4 bboxMin;
    vec4 bboxMax;
    vec4 coneAxis;
    uvec4 children;
};

layout(std430, binding = 25) readonly buffer LightTreeBuffer {
    LightTreeNode lightNodes[];
};

layout(std430, binding = 30) readonly buffer InstancesBuffer {
    InstanceGPU instances[];
};

layout(binding = 12, rgba16f) uniform image2D uNormalDepthImage;
layout(binding = 15, rgba16f) uniform readonly image2D uCausticImage;

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
    float fractionalSpp;
    uint totalCompositeSpp;
    uint numOpaqueTriangles;
    float indirectClamp;
} pc;

// RNG
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

// In-shader hardware ray query shadow occluder test (aligned with wavefront_shadow_inline.glsl:
// by default every non-cutout surface occludes; UBO bit 26 enables --transparent-shadows fast mode)
bool isShadowOccluded(vec3 origin, vec3 dir, float tMin, float tMax) {
    bool hasNonOpaque = (ubo.flags & (1u << 5)) != 0u;
    bool hasAlphaShadow = (ubo.flags & (1u << 27)) != 0u;
    bool transparentShadows = (ubo.flags & (1u << 26)) != 0u;
    bool enableCaustics = (ubo.flags & (1u << 8)) != 0u && (pc.numLights > 0u);

    if (!hasAlphaShadow && (!transparentShadows || !enableCaustics)) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topLevelAS,
                               gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                               (transparentShadows && hasNonOpaque) ? 0x01u : 0xFFu,
                               origin, tMin, dir, tMax);
        rayQueryProceedEXT(rq);
        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            return true;
        }
    } else {
        rayQueryEXT rq;
        uint rqFlags = gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT
                     | (transparentShadows ? gl_RayFlagsNoOpaqueEXT : 0u);
        rayQueryInitializeEXT(rq, topLevelAS, rqFlags, 0xFFu, origin, tMin, dir, tMax);
        while (rayQueryProceedEXT(rq)) {
            if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
                uint cInstIdx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false);
                InstanceGPU cInst = instances[cInstIdx];
                uint geomIdx = rayQueryGetIntersectionGeometryIndexEXT(rq, false);
                uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
                uint triIdx = cInst.firstTriangle + ((geomIdx == 0u) ? primIdx : (primIdx + cInst.numOpaqueTriangles));
                uint matId = triangles[triIdx].materialId + cInst.materialOffset;
                Material mat = materials[matId];
                bool isDielectric = ((mat.type & MATERIAL_TYPE_MASK) == 2u /* DIELECTRIC */ || mat.transmission > 0.05);
                if (transparentShadows) {
                    if ((mat.type & MATERIAL_TYPE_MASK) == 3u /* Skip EMISSIVE */ || (!enableCaustics && isDielectric)) {
                        continue;
                    }
                    if (enableCaustics && isDielectric && mat.thickness <= 0.001) {
                        continue;
                    }
                }
                if (mat.alphaMode == 1u /* MASK */ || mat.alphaMode == 2u /* BLEND */) {
                    vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, false);
                    float cw = 1.0 - bary.x - bary.y;
                    vec2 cuv = cw * vec2(triangles[triIdx].uv0x, triangles[triIdx].uv0y)
                             + bary.x * vec2(triangles[triIdx].uv1x, triangles[triIdx].uv1y)
                             + bary.y * vec2(triangles[triIdx].uv2x, triangles[triIdx].uv2y);
                    float calpha = mat.albedo.a;
                    if (mat.albedoTex > 0u && mat.albedoTex <= 512u) {
                        calpha *= SAMPLE_SCENE_TEXTURE(mat.albedoTex, cuv).a;
                    }
                    float cutoff = (mat.alphaMode == 1u) ? mat.alphaCutoff : 0.5;
                    if (calpha < cutoff) {
                        continue; // Skip transparent pixel
                    }
                }
                rayQueryConfirmIntersectionEXT(rq);
                rayQueryTerminateEXT(rq);
            }
        }
        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            return true;
        }
    }

    for (uint i = 0; i < pc.numSpheres; ++i) {
        Material sMat = materials[spheres[i].materialId];
        uint sType = sMat.type & MATERIAL_TYPE_MASK;
        bool isDielectric = (sType == 2u || sMat.transmission > 0.05);
        if (transparentShadows) {
            if (sType == 3u || (!enableCaustics && isDielectric)) continue;
            if (enableCaustics && isDielectric && sMat.thickness <= 0.001) continue;
        }
        float spT;
        vec3 spNorm;
        if (intersectSphere(origin, dir, spheres[i], tMin, tMax, spT, spNorm)) {
            return true;
        }
    }
    return false;
}

// PBR & Shared BSDF/NEE/MIS Helpers
vec3 offsetRayOrigin(vec3 p, vec3 n) {
    float maxCoord = max(1.0, max(abs(p.x), max(abs(p.y), abs(p.z))) * 0.03125);
    return p + n * (0.0005 * maxCoord);
}

vec3 adjustShadingNormal(vec3 N, vec3 V) {
    const float k = 0.05;
    float backFacing = (max(-k, dot(-V, N) + 1e-4) + k) * (1.0 / (1.0 + k));
    return (backFacing > 0.0) ? normalize(N + backFacing * V) : N;
}

void buildOrthonormalBasis(vec3 n, out vec3 b1, out vec3 b2) {
    float sign = (n.z >= 0.0) ? 1.0 : -1.0;
    float a = -1.0 / (sign + n.z);
    float b = n.x * n.y * a;
    b1 = vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    b2 = vec3(b, sign + n.y * n.y * a, -n.y);
}

vec3 sampleCosineHemisphereVec2(vec3 normal, vec2 r) {
    float phi = TWO_PI * r.x;
    float cosTheta = sqrt(r.y);
    float sinTheta = sqrt(max(0.0, 1.0 - r.y));
    vec3 tangent = abs(normal.x) > 0.1 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    tangent = normalize(cross(normal, tangent));
    vec3 bitangent = cross(normal, tangent);
    return normalize(tangent * (cos(phi) * sinTheta) +
                     bitangent * (sin(phi) * sinTheta) +
                     normal * cosTheta);
}

vec3 sampleCosineHemisphere(vec3 normal, inout uint seed) {
    return sampleCosineHemisphereVec2(normal, randVec2(seed));
}

float fresnelSchlick(float cosTheta, float refIdx) {
    float r0 = (1.0 - refIdx) / (1.0 + refIdx);
    r0 = r0 * r0;
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x2 = x * x;
    return r0 + (1.0 - r0) * (x2 * x2 * x);
}

float fresnelDielectric(float cosThetaI, float eta) {
    float sinThetaI2 = max(0.0, 1.0 - cosThetaI * cosThetaI);
    float sinThetaT2 = (eta * eta) * sinThetaI2;
    if (sinThetaT2 >= 1.0) {
        return 1.0;
    }
    float cosT = sqrt(max(0.0, 1.0 - sinThetaT2));
    float r0 = (1.0 - eta) / (1.0 + eta);
    r0 = r0 * r0;
    float cosEval = (eta > 1.0) ? cosT : cosThetaI;
    float x = clamp(1.0 - cosEval, 0.0, 1.0);
    float x2 = x * x;
    return r0 + (1.0 - r0) * (x2 * x2 * x);
}

vec3 fresnelSchlickVec(float cosTheta, vec3 F0) {
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
    float x2 = x * x;
    return F0 + (vec3(1.0) - F0) * (x2 * x2 * x);
}

float distributionGGX(float NdotH, float alpha) {
    float a2 = max(alpha * alpha, 1e-6);
    float d = (NdotH * NdotH * (a2 - 1.0) + 1.0);
    return a2 / (PI * d * d);
}

float visibilitySmithGGXCorrelated(float NdotL, float NdotV, float alpha) {
    float a2 = max(alpha * alpha, 1e-6);
    float gv = NdotL * sqrt(NdotV * NdotV * (1.0 - a2) + a2);
    float gl = NdotV * sqrt(NdotL * NdotL * (1.0 - a2) + a2);
    return 0.5 / max(gv + gl, 1e-6);
}

vec3 sampleVNDF_GGX_Vec2(vec3 V_local, float alpha_x, float alpha_y, vec2 u) {
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

float distributionAnisotropicGGX(float TdotH, float BdotH, float NdotH, float ax, float ay) {
    float d = (TdotH * TdotH) / max(ax * ax, 1e-6) + (BdotH * BdotH) / max(ay * ay, 1e-6) + NdotH * NdotH;
    return 1.0 / max(PI * ax * ay * d * d, 1e-6);
}

float evalSheenCharlie(float NdotH, float sheenRoughness) {
    float invR = 1.0 / max(sheenRoughness, 0.01);
    float cos2 = NdotH * NdotH;
    float sin2 = max(1.0 - cos2, 1e-6);
    return (2.0 + invR) * pow(sin2, invR * 0.5) / TWO_PI;
}

vec3 evalThinFilmIridescence(float cosTheta, float iridIor, float thicknessNm) {
    float sinTheta2 = 1.0 - cosTheta * cosTheta;
    float cosThetaT2 = 1.0 - sinTheta2 / max(iridIor * iridIor, 1e-4);
    float cosThetaT = sqrt(max(0.0, cosThetaT2));
    float opd = 2.0 * iridIor * thicknessNm * cosThetaT;
    vec3 phase = (TWO_PI * opd) / vec3(650.0, 530.0, 460.0);
    return clamp(0.5 + 0.5 * cos(phase), 0.0, 1.0);
}

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

float evalLightTreeNodeImportance(vec3 hitPoint, vec3 hitNormal, LightTreeNode node) {
    float flux = node.bboxMin.w;
    if (flux <= 1e-7) return 0.0;
    vec3 bMin = node.bboxMin.xyz;
    vec3 bMax = node.bboxMax.xyz;
    vec3 clampedPoint = clamp(hitPoint, bMin, bMax);
    vec3 delta = hitPoint - clampedPoint;
    float distSq = max(dot(delta, delta), 1e-4);
    vec3 boxCentroid = 0.5 * (bMin + bMax);
    vec3 toNode = boxCentroid - hitPoint;
    float centerDist = length(toNode);
    float coneCos = node.bboxMax.w;
    float cosCone = 1.0;
    if (coneCos > -0.99 && centerDist > 1e-4) {
        vec3 dir = toNode / centerDist;
        float cosLightDir = dot(node.coneAxis.xyz, -dir);
        cosCone = clamp((cosLightDir - coneCos) / max(1.0 - coneCos, 1e-3), 0.05, 1.0);
    }
    return (flux * cosCone) / distSq;
}

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
                if (node_.children.z == 1u) { \
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

bool shouldSuppressCausticEmitterHit(bool causticApplied, bool isSpecularPath, uint inPdfCode, uint bounce) {
    return causticApplied && !isSpecularPath && (inPdfCode == 0u) && (bounce >= 2u);
}

#include "mis_common.glsl"
#include "bsdf_standard.glsl"
#define SHADING_CORE_SKIP_SCENE_HIT
#define SHADING_CORE_ENABLE_NEE
#include "shading_core.glsl"

void main() {
    uint inFlags = prd.packedThroughputB_Flags >> 16u;
    bool prevIsDelta = (inFlags & 2u) != 0u;
    bool isPrimary = (inFlags & 4u) != 0u;
    bool causticApplied = (inFlags & 16u) != 0u;
    bool isSpecularPath = (inFlags & 32u) != 0u;
    bool bounceGe2 = (inFlags & 64u) != 0u;
    bool canContinue = (inFlags & 256u) != 0u;
    uint dispersionChannel = (inFlags >> 10u) & 3u;
    uint inPdfCode = prevIsDelta ? 0u : encodeBsdfPdf(prd.lastBsdfPdf);
    uint effBounce = isPrimary ? 0u : (bounceGe2 ? 2u : 1u);

    vec3 accumRadiance = vec3(0.0);
    vec3 accumDiffuse = vec3(0.0);
    vec3 accumSpecular = vec3(0.0);

    InstanceGPU inst = instances[gl_InstanceCustomIndexEXT];
    uint primID = inst.firstTriangle + ((gl_GeometryIndexEXT == 0) ? uint(gl_PrimitiveID) : (uint(gl_PrimitiveID) + inst.numOpaqueTriangles));
    Triangle tri = triangles[primID];

    vec3 hitPoint = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 normal = normalize(transpose(mat3(gl_WorldToObjectEXT)) * getTriangleNormal(tri, attribs));
    vec2 hitUv = getTriangleUV(tri, attribs);
    uint fullW = (pc.tileWidth > 0u) ? pc.tileWidth : gl_LaunchSizeEXT.x;

    bool frontFace = dot(gl_WorldRayDirectionEXT, normal) < 0.0;
    vec3 geomNormal = frontFace ? normal : -normal;
    vec3 hitNormal = geomNormal;

    uint matId = tri.materialId + inst.materialOffset;
    Material mat = materials[matId];

    // Alpha masking & stochastic alpha blending (non-bounce-consuming passthrough)
    vec4 baseColor = mat.albedo;
    if (mat.albedoTex > 0u && mat.albedoTex <= 512u) {
        baseColor *= SAMPLE_SCENE_TEXTURE(mat.albedoTex, hitUv);
    }
    if ((mat.alphaMode == 1u && baseColor.a < mat.alphaCutoff) ||
        (mat.alphaMode == 2u && randFloat(prd.seed) > baseColor.a)) {
        prd.diffuseRadiance = vec3(0.0);
        prd.specularRadiance = vec3(0.0);
        prd.packedAlbedoRG = 0u;
        prd.packedAlbedoB_Roughness = 0u;
        prd.packedThroughputRG = packHalf2x16(vec2(1.0, 1.0));
        prd.nextOrigin = offsetRayOrigin(hitPoint, gl_WorldRayDirectionEXT);
        uint outFlags = (inFlags & (2u | 16u | (3u << 10u))) | 1u | 128u; // preserve isDelta, causticApplied, dispersionChannel; mark passthrough (128u)
        prd.packedThroughputB_Flags = (packHalf2x16(vec2(1.0, 0.0)) & 0xFFFFu) | (outFlags << 16u);
        prd.packedNextDir = packOct32(gl_WorldRayDirectionEXT);
        prd.pad = 0u;
        return;
    }

    vec3 geomTan = mat3(gl_ObjectToWorldEXT) * getTriangleTangent(tri, attribs);
    float tanSign = getTriangleTangentSign(tri);

    // Normal mapping
    if (mat.normalTex > 0u && mat.normalTex <= 512u) {
        vec3 normalMap = SAMPLE_SCENE_TEXTURE(mat.normalTex, hitUv).rgb * 2.0 - 1.0;
        normalMap.xy *= mat.normalScale;
        normalMap = normalize(normalMap);

        vec3 nTan = geomTan - dot(geomTan, hitNormal) * hitNormal;
        if (dot(nTan, nTan) <= 1e-6) {
            vec3 b1, b2;
            buildOrthonormalBasis(hitNormal, b1, b2);
            nTan = b1;
            tanSign = 1.0;
        } else {
            nTan = normalize(nTan);
        }
        vec3 geomBitangent = cross(hitNormal, nTan) * tanSign;
        mat3 tbn = mat3(nTan, geomBitangent, hitNormal);
        hitNormal = adjustShadingNormal(normalize(tbn * normalMap), -gl_WorldRayDirectionEXT);
    }

    // Material attributes
    float roughness = mat.roughness;
    float metallic = mat.metallic;
    if (mat.mrTex > 0u && mat.mrTex <= 512u) {
        vec4 mrSample = SAMPLE_SCENE_TEXTURE(mat.mrTex, hitUv);
        roughness *= mrSample.g;
        metallic *= mrSample.b;
    }
    bool isMirrorDelta = (metallic > 0.99 && roughness <= 0.008);
    roughness = clamp(roughness, 0.001, 1.0);
    metallic = clamp(metallic, 0.0, 1.0);
    float alphaRoughness = roughness * roughness;

    float transmission = mat.transmission;
    if (mat.transmissionTex > 0u && mat.transmissionTex <= 512u) {
        transmission *= SAMPLE_SCENE_TEXTURE(mat.transmissionTex, hitUv).r;
    }

    bool isHoloVideo = ((mat.type & (1u << 14)) != 0u) && (mat.emissiveTex > 0u && mat.emissiveTex <= 512u);
    uint dynVideoBit = (isHoloVideo && ((ubo.flags & (1u << 12)) != 0u)) ? 512u : 0u;
    vec3 emissive = mat.emissive.rgb;
    if (isHoloVideo) {
        vec3 vidSample = SAMPLE_SCENE_TEXTURE(mat.emissiveTex, hitUv).rgb;
        float time = float(ubo.frameIndex) * 0.01666667;
        float sweep = sin(hitPoint.y * 24.0 - time * 4.0) * 0.5 + 0.5;
        float scanMod = 0.94 + 0.06 * sweep;
        float edgeDist = length(hitUv - vec2(0.5)) * 2.0;
        float fringeFactor = pow(clamp((edgeDist - 0.5) * 2.0, 0.0, 1.0), 2.5) * 0.06;
        vec3 chromaDisp = vec3(0.0);
        if (fringeFactor > 1e-4) {
            float phase = hitPoint.y * 48.0 - time * 5.0;
            chromaDisp = vec3(sin(phase), sin(phase + 1.2), sin(phase + 2.4)) * fringeFactor;
        }
        vec3 holoBase = vidSample * (0.65 * scanMod) + chromaDisp;
        vec3 tonemappedVid = holoBase / (vec3(1.0) + 0.20 * max(holoBase - vec3(0.40), vec3(0.0)));
        float lum = dot(vidSample, vec3(0.2126, 0.7152, 0.0722));
        vec3 holoPedestal = vec3(0.008, 0.025, 0.045) * (0.7 + 0.3 * sweep) * (1.0 - smoothstep(0.0, 0.40, lum));
        vec3 rimGlow = vec3(0.0);
        if (frontFace) {
            float cosTheta = clamp(abs(dot(hitNormal, -gl_WorldRayDirectionEXT)), 0.0, 1.0);
            float fres = pow(1.0 - cosTheta, 3.0);
            rimGlow = mix(vec3(0.02, 0.25, 0.70), vec3(0.05, 0.85, 0.95), fres) * (fres * 0.18);
        }
        float glitchPulse = smoothstep(0.986, 0.990, sin(time * 19.0 + hitPoint.y * 23.0)) * 0.08;
        emissive = tonemappedVid * (1.0 + glitchPulse) + holoPedestal + rimGlow;
    } else if (mat.emissiveTex > 0u && mat.emissiveTex <= 512u) {
        emissive *= SAMPLE_SCENE_TEXTURE(mat.emissiveTex, hitUv).rgb;
    }

    // Volumetric Beer-Lambert absorption (physical ray traversal distance)
    float mediumDist = !frontFace ? gl_HitTEXT : 0.0;
    vec3 transmittance = vec3(1.0);
    if (mat.attenuationColor.w > 0.0 && mediumDist > 0.0) {
        vec3 sigma_a = -log(max(mat.attenuationColor.rgb, vec3(0.0001))) / mat.attenuationColor.w;
        transmittance = exp(-sigma_a * mediumDist);
    }

    bool enableDirect = (ubo.flags & (1 << 0)) != 0;
    bool enableSpecular = (ubo.flags & (1 << 2)) != 0;
    bool enableRefraction = (ubo.flags & (1 << 3)) != 0;
    bool enableShadows = (ubo.flags & (1 << 4)) != 0;

    // 1. Emissive contribution with consistent MIS + caustic DS+L suppression
    if (length(emissive) > 1e-3) {
        float misWeight = shouldSuppressCausticEmitterHit(causticApplied, isSpecularPath, inPdfCode, effBounce)
            ? 0.0
            : (isPrimary ? 1.0 : emitterHitMisWeight(0u, primID, gl_InstanceCustomIndexEXT, gl_WorldRayOriginEXT, gl_WorldRayDirectionEXT, gl_HitTEXT, inPdfCode, pc.numLights));
        accumRadiance = emissive * misWeight;
        accumSpecular = emissive * misWeight;
        if ((mat.type & MATERIAL_TYPE_MASK) == 3u /* Emissive */) {
            if (isPrimary) {
                int currentPx = int(prd.pad % fullW);
                int currentPy = int(prd.pad / fullW);
                ivec2 baseCoord = ivec2(currentPx, currentPy);
                vec3 viewGeomNormal = normalize(transpose(mat3(ubo.viewInverse)) * geomNormal);
                vec3 camForward = -ubo.viewInverse[2].xyz;
                float planarDepth = max(dot(hitPoint - ubo.position.xyz, camForward), 0.0);
                imageStore(uNormalDepthImage, baseCoord, vec4(viewGeomNormal, planarDepth));
            }
            vec3 specOut = accumRadiance * transmittance;
            if (!isPrimary && pc.indirectClamp > 0.0) {
                float eLum = dot(specOut, vec3(0.2126, 0.7152, 0.0722));
                if (eLum > pc.indirectClamp) {
                    specOut *= (pc.indirectClamp / eLum);
                }
            }
            prd.diffuseRadiance = vec3(0.0);
            prd.specularRadiance = specOut;
            if (isPrimary) {
                prd.packedAlbedoRG = packHalf2x16(baseColor.rg);
                prd.packedAlbedoB_Roughness = packHalf2x16(vec2(baseColor.b, roughness));
            } else {
                prd.packedAlbedoRG = 0u;
                prd.packedAlbedoB_Roughness = 0u;
            }
            prd.packedThroughputRG = 0u;
            uint outFlags = 1u | (causticApplied ? 16u : 0u) | dynVideoBit | ((dispersionChannel & 3u) << 10u);
            prd.packedThroughputB_Flags = outFlags << 16u;
            prd.packedNextDir = 0u;
            prd.nextOrigin = hitPoint;
            prd.lastBsdfPdf = 0.0;
            prd.pad = 0u;
            return;
        }
    }

    float iorVal = mat.ior > 0.0 ? mat.ior : 1.5;
    float f0Dielectric = pow((iorVal - 1.0) / (iorVal + 1.0), 2.0);
    float specFactor = mat.specularFactor;
    if (mat.specularTex > 0u && mat.specularTex <= 512u) {
        specFactor *= SAMPLE_SCENE_TEXTURE(mat.specularTex, hitUv).a;
    }
    vec3 dielectricF0 = vec3(clamp(f0Dielectric * specFactor, 0.0, 1.0));
    vec3 F0 = mix(dielectricF0, baseColor.rgb, metallic);
    vec3 diffuseColor = baseColor.rgb * (1.0 - metallic);
    vec3 V = -gl_WorldRayDirectionEXT;

    float diffTrans = clamp(mat.diffuseTransmission, 0.0, 1.0);
    if (mat.diffuseTransmissionTex > 0u && mat.diffuseTransmissionTex <= 512u) {
        diffTrans *= SAMPLE_SCENE_TEXTURE(mat.diffuseTransmissionTex, hitUv).r;
    }
    diffTrans = clamp(diffTrans, 0.0, 1.0);

    // Clearcoat
    float clearcoat = clamp(mat.clearcoat, 0.0, 1.0);
    if (mat.clearcoatTex > 0u && mat.clearcoatTex <= 512u) {
        clearcoat *= SAMPLE_SCENE_TEXTURE(mat.clearcoatTex, hitUv).r;
    }
    float clearcoatRoughness = mat.clearcoatRoughness;
    if (mat.clearcoatRoughnessTex > 0u && mat.clearcoatRoughnessTex <= 512u) {
        clearcoatRoughness *= SAMPLE_SCENE_TEXTURE(mat.clearcoatRoughnessTex, hitUv).g;
    }
    clearcoatRoughness = clamp(clearcoatRoughness, 0.001, 1.0);
    float clearcoatAlpha = clearcoatRoughness * clearcoatRoughness;

    vec3 surfTan = geomTan - dot(geomTan, hitNormal) * hitNormal;
    if (dot(surfTan, surfTan) <= 1e-6) {
        vec3 b1, b2;
        buildOrthonormalBasis(hitNormal, b1, b2);
        surfTan = b1;
        tanSign = 1.0;
    } else {
        surfTan = normalize(surfTan);
    }
    vec3 surfBitan = cross(hitNormal, surfTan) * tanSign;

    vec3 clearcoatNormal = hitNormal;
    if (clearcoat > 0.001 && mat.clearcoatNormalTex > 0u && mat.clearcoatNormalTex <= 512u) {
        vec3 cNormMap = SAMPLE_SCENE_TEXTURE(mat.clearcoatNormalTex, hitUv).rgb * 2.0 - 1.0;
        clearcoatNormal = adjustShadingNormal(normalize(mat3(surfTan, surfBitan, hitNormal) * cNormMap), -gl_WorldRayDirectionEXT);
    }

    vec3 sheenColor = mat.sheenColor;
    if (mat.sheenTex > 0u && mat.sheenTex <= 512u) {
        sheenColor *= SAMPLE_SCENE_TEXTURE(mat.sheenTex, hitUv).rgb;
    }
    float sheenRoughness = clamp(mat.sheenRoughness, 0.01, 1.0);

    float anisoStrength = clamp(mat.anisotropyStrength, 0.0, 1.0);
    vec2 anisoRot = vec2(cos(mat.anisotropyRotation), sin(mat.anisotropyRotation));
    vec2 anisoDir = anisoRot;
    if (anisoStrength > 0.001 && mat.anisotropyTex > 0u && mat.anisotropyTex <= 512u) {
        vec3 anisoSample = SAMPLE_SCENE_TEXTURE(mat.anisotropyTex, hitUv).rgb;
        vec2 texDir = anisoSample.xy * 2.0 - 1.0;
        float texLen = length(texDir);
        texDir = (texLen > 1e-4) ? (texDir / texLen) : vec2(1.0, 0.0);
        anisoDir = vec2(texDir.x * anisoRot.x - texDir.y * anisoRot.y,
                        texDir.x * anisoRot.y + texDir.y * anisoRot.x);
        anisoStrength *= anisoSample.b;
    }
    vec3 anisoT = normalize(surfTan * anisoDir.x + surfBitan * anisoDir.y);
    vec3 anisoB = normalize(cross(hitNormal, anisoT));
    float alphaX = mix(alphaRoughness, 1.0, anisoStrength * anisoStrength);
    float alphaY = alphaRoughness;

    StdBsdf bsdf = makeStdBsdf(diffuseColor, F0, alphaRoughness, diffTrans, metallic, dot(hitNormal, V), STD_BSDF_FULL);
    stdBsdfSetLayers(bsdf, enableSpecular, clearcoatNormal, clearcoat, clearcoatAlpha, sheenColor, sheenRoughness);
    stdBsdfSetAnisoIridescence(bsdf, anisoT, anisoB, alphaX, alphaY,
                               mat.iridescence, mat.iridescenceIor, mat.iridescenceThickness);

    vec3 camForward = -ubo.viewInverse[2].xyz;
    float hitDepth = max(dot(hitPoint - ubo.position.xyz, camForward), 0.0);
    vec3 viewNormal = normalize(transpose(mat3(ubo.viewInverse)) * hitNormal);

    bool isDielectricDelta = (transmission > 0.01 || (mat.type & MATERIAL_TYPE_MASK) == 2u);

    // 2. Direct Lighting via shared StdBsdf NEE + LightTree/Alias RIS + MIS
    if (enableDirect && pc.numLights > 0u && (mat.type & MATERIAL_TYPE_MASK) != 3u && !isDielectricDelta && !isMirrorDelta) {
        uint M = (pc.numLights > 1u) ? min(pc.numLights, 4u) : 1u;
        vec3 shadowOrigin, lightDir, directDiffuse, directSpecular;
        float lightDist;
        if (evalStandardDirectNEE(bsdf, hitPoint, hitNormal, V, pc.numLights, M,
                                  canContinue, prd.seed,
                                  shadowOrigin, lightDir, lightDist,
                                  directDiffuse, directSpecular)) {
            bool inShadow = enableShadows
                ? isShadowOccluded(shadowOrigin, lightDir, EPSILON, max(lightDist - EPSILON * 2.0, EPSILON * 2.0))
                : false;
            if (!inShadow) {
                accumRadiance += directDiffuse + directSpecular;
                accumDiffuse += directDiffuse;
                accumSpecular += directSpecular;
            }
        }
    }

    // Forward photon caustics strictly on primary bounce!
    bool enableCaustics = (ubo.flags & (1u << 8)) != 0u && (pc.numLights > 0u);
    if (enableCaustics && isPrimary && !causticApplied && (mat.type & MATERIAL_TYPE_MASK) != 3u && !isDielectricDelta && metallic < 0.2) {
        int currentPx = int(prd.pad % fullW);
        int currentPy = int(prd.pad / fullW);
        ivec2 baseCoord = ivec2(currentPx, currentPy);
        vec3 causticIrr = imageLoad(uCausticImage, baseCoord).rgb;
        if (dot(causticIrr, causticIrr) > 1e-8) {
            vec3 causticDiff = causticIrr * (diffuseColor * INV_PI);
            accumRadiance += causticDiff;
            accumDiffuse += causticDiff;
            causticApplied = true;
        }
    }

    accumRadiance *= transmittance;
    accumDiffuse *= transmittance;
    accumSpecular *= transmittance;

    // 3. BSDF Sampling for Next Direction
    vec3 nextDirection = vec3(0.0, 1.0, 0.0);
    vec3 nextOrigin = offsetRayOrigin(hitPoint, hitNormal);
    vec3 throughputFactor = vec3(0.0);
    bool sampleSpecular = false;
    bool outIsDelta = false;
    float outPdf = 0.0;

    if (isDielectricDelta) {
        vec3 unitDir = normalize(gl_WorldRayDirectionEXT);
        if (!enableRefraction) {
            nextDirection = sampleCosineHemisphere(hitNormal, prd.seed);
            nextOrigin = offsetRayOrigin(hitPoint, hitNormal);
            throughputFactor = baseColor.rgb;
            outPdf = max(dot(hitNormal, nextDirection), 1e-4) * INV_PI;
        } else {
            if (mat.dispersion > 0.001 && dispersionChannel != 0u) {
                iorVal += (float(dispersionChannel - 1u) - 1.0) * (mat.dispersion * 0.03 * iorVal);
            }
            float refractionRatio = frontFace ? (1.0 / iorVal) : iorVal;
            float cosTheta = min(dot(-unitDir, hitNormal), 1.0);
            float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

            bool cannotRefract = refractionRatio * sinTheta > 1.0;
            float reflectProb = fresnelDielectric(cosTheta, refractionRatio);
            vec3 throughputMod = transmittance;

            bool isThinWalled = (mat.thickness <= 0.001);
            if (isThinWalled) {
                if (!frontFace) {
                    reflectProb = 0.0;
                    cannotRefract = false;
                } else {
                    reflectProb = (2.0 * reflectProb) / (1.0 + reflectProb);
                }
            }

            if (cannotRefract || reflectProb > randFloat(prd.seed)) {
                nextDirection = reflect(unitDir, hitNormal);
                prd.nextOrigin = offsetRayOrigin(hitPoint, hitNormal);
            } else if (isThinWalled) {
                nextDirection = unitDir;
                prd.nextOrigin = offsetRayOrigin(hitPoint, unitDir * 2.0);
                throughputMod = baseColor.rgb * transmittance;
            } else {
                vec3 dispMask = vec3(1.0);
                uint candidateChannel = dispersionChannel;
                float effRatio = refractionRatio;
                if (mat.dispersion > 0.001 && dispersionChannel == 0u) {
                    uint channel = min(uint(randFloat(prd.seed) * 3.0), 2u);
                    candidateChannel = channel + 1u;
                    dispMask = vec3(channel == 0u ? 3.0 : 0.0, channel == 1u ? 3.0 : 0.0, channel == 2u ? 3.0 : 0.0);
                    float dispIor = iorVal + (float(channel) - 1.0) * (mat.dispersion * 0.03 * iorVal);
                    effRatio = frontFace ? (1.0 / dispIor) : dispIor;
                }
                vec3 refrVec = refract(unitDir, hitNormal, effRatio);
                if (dot(refrVec, refrVec) > 1e-6) {
                    dispersionChannel = candidateChannel;
                    nextDirection = normalize(refrVec);
                    prd.nextOrigin = offsetRayOrigin(hitPoint, -hitNormal);
                    throughputMod = baseColor.rgb * transmittance * dispMask;
                } else {
                    // Post-shift TIR keeps ray monochromatic only if already split; otherwise reflect unsplit
                    nextDirection = reflect(unitDir, hitNormal);
                    prd.nextOrigin = offsetRayOrigin(hitPoint, hitNormal);
                }
            }
            if (isPrimary) {
                int currentPx = int(prd.pad % fullW);
                int currentPy = int(prd.pad / fullW);
                ivec2 baseCoord = ivec2(currentPx, currentPy);
                imageStore(uNormalDepthImage, baseCoord, vec4(viewNormal, hitDepth));
                prd.packedAlbedoRG = packHalf2x16(baseColor.rg);
                prd.packedAlbedoB_Roughness = packHalf2x16(vec2(baseColor.b, roughness));
            } else {
                prd.packedAlbedoRG = 0u;
                prd.packedAlbedoB_Roughness = 0u;
            }
            prd.diffuseRadiance = vec3(0.0);
            prd.specularRadiance = accumRadiance;
            prd.packedThroughputRG = packHalf2x16(throughputMod.rg);
            uint outFlags = 1u | 2u | 8u | (causticApplied ? 16u : 0u) | dynVideoBit | ((dispersionChannel & 3u) << 10u);
            prd.packedThroughputB_Flags = (packHalf2x16(vec2(throughputMod.b, 0.0)) & 0xFFFFu) | (outFlags << 16u);
            prd.packedNextDir = packOct32(normalize(nextDirection));
            prd.lastBsdfPdf = 0.0;
            prd.pad = 0u;
            return;
        }
    } else if (isMirrorDelta) {
        sampleSpecular = true;
        outIsDelta = true;
        float NdotV = clamp(dot(hitNormal, V), 0.0, 1.0);
        nextDirection = reflect(-V, hitNormal);
        nextOrigin = offsetRayOrigin(hitPoint, hitNormal);
        throughputFactor = fresnelSchlickVec(NdotV, F0);
    } else {
        float uLobe = randFloat(prd.seed);
        float uTrans = randFloat(prd.seed);
        vec2 uDir = randVec2(prd.seed);
        bool isTrans = false;
        if (sampleStdBsdfBounce(bsdf, hitNormal, V, uLobe, uTrans, uDir,
                                nextDirection, throughputFactor, outPdf, sampleSpecular, isTrans)) {
            throughputFactor = min(throughputFactor, vec3(65000.0));
            nextOrigin = offsetRayOrigin(hitPoint, isTrans ? -hitNormal : hitNormal);
        } else {
            throughputFactor = vec3(0.0);
        }
    }

    if (isPrimary) {
        int currentPx = int(prd.pad % fullW);
        int currentPy = int(prd.pad / fullW);
        ivec2 baseCoord = ivec2(currentPx, currentPy);
        imageStore(uNormalDepthImage, baseCoord, vec4(viewNormal, hitDepth));
        prd.packedAlbedoRG = packHalf2x16(baseColor.rg);
        prd.packedAlbedoB_Roughness = packHalf2x16(vec2(baseColor.b, roughness));
    } else {
        prd.packedAlbedoRG = 0u;
        prd.packedAlbedoB_Roughness = 0u;
    }

    if (!isPrimary && pc.indirectClamp > 0.0) {
        float dLum = dot(accumDiffuse, vec3(0.2126, 0.7152, 0.0722));
        if (dLum > pc.indirectClamp) {
            accumDiffuse *= (pc.indirectClamp / dLum);
        }
        float sLum = dot(accumSpecular, vec3(0.2126, 0.7152, 0.0722));
        if (sLum > pc.indirectClamp) {
            accumSpecular *= (pc.indirectClamp / sLum);
        }
    }

    prd.diffuseRadiance = accumDiffuse;
    prd.specularRadiance = accumSpecular;
    prd.nextOrigin = nextOrigin;
    prd.packedNextDir = packOct32(normalize(nextDirection));
    prd.packedThroughputRG = packHalf2x16(throughputFactor.rg * transmittance.rg);
    uint outFlags = 1u | (outIsDelta ? 2u : 0u) | (sampleSpecular ? 8u : 0u) | (causticApplied ? 16u : 0u) | dynVideoBit | ((dispersionChannel & 3u) << 10u);
    prd.packedThroughputB_Flags = (packHalf2x16(vec2(throughputFactor.b * transmittance.b, 0.0)) & 0xFFFFu) | (outFlags << 16u);
    prd.lastBsdfPdf = outIsDelta ? 0.0 : outPdf;
    prd.pad = 0u;
}
