#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : enable
#extension GL_EXT_nonuniform_qualifier : enable

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692
#define INV_PI 0.31830988618379067154
#define EPSILON 0.001

struct HitPayload {
    vec3 radiance;               // 12 bytes: direct emissive + accumulated direct light
    uint packedThroughputRG;     //  4 bytes: packHalf2x16(throughputMod.rg)
    vec3 nextOrigin;             // 12 bytes: next ray origin
    uint packedThroughputB_Flags;//  4 bytes: lower 16 bits = half(b), upper 16 bits = flags
    uint packedNextDir;          //  4 bytes: octahedral 32-bit (oct32) unit direction
    float lastBsdfPdf;           //  4 bytes: BSDF PDF for next bounce MIS evaluation
    uint seed;                   //  4 bytes: PCG RNG state
    uint pad;                    //  4 bytes: 48-byte cache-line alignment
};

layout(location = 0) rayPayloadInEXT HitPayload prd;
hitAttributeEXT vec2 attribs;

// Scene structures
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
};

struct Light {
    vec4 position; // xyz: pos/corner, w: type
    vec4 emission; // rgb: color, w: area
    vec4 u;        // xyz: edge1, w: spot inner cos
    vec4 v;        // xyz: edge2, w: spot outer cos
    vec4 normal;   // xyz: normal/dir, w: padding
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
layout(binding = 8) uniform sampler2D sceneTextures[64];

struct ReservoirDI {
    uint lightIdx;
    float uvX;
    float uvY;
    float wSum;
    float M;
    float W;
    float targetPdf;
    uint pad;
};

layout(std430, binding = 9) buffer CurrentReservoirBuffer {
    ReservoirDI currentReservoirs[];
};

layout(std430, binding = 10) readonly buffer HistoryReservoirBuffer {
    ReservoirDI historyReservoirs[];
};

layout(binding = 11, rgba16f) uniform image2D uDirectLightImage;
layout(binding = 12, rgba16f) uniform image2D uNormalDepthImage;

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

// 32-bit Octahedral Encoding (Cigolle et al.)
vec2 octSign(vec2 v) {
    return vec2((v.x >= 0.0) ? 1.0 : -1.0, (v.y >= 0.0) ? 1.0 : -1.0);
}

vec2 octEncode(vec3 v) {
    float invL1 = 1.0 / (abs(v.x) + abs(v.y) + abs(v.z));
    vec2 p = v.xy * invL1;
    return (v.z >= 0.0) ? p : (vec2(1.0) - abs(p.yx)) * octSign(p);
}

uint packOct32(vec3 v) {
    return packSnorm2x16(octEncode(normalize(v)));
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

// In-shader hardware ray query shadow occluder test
bool isShadowOccluded(vec3 origin, vec3 dir, float tMin, float tMax) {
    bool hasNonOpaque = (ubo.flags & (1u << 5)) != 0u;
    if (!hasNonOpaque) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topLevelAS,
                               gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                               0xFF, origin, tMin, dir, tMax);
        rayQueryProceedEXT(rq);
        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            return true;
        }
        for (uint i = 0; i < pc.numSpheres; ++i) {
            if (materials[spheres[i].materialId].type == 3u) continue;
            float spT;
            vec3 spNorm;
            if (intersectSphere(origin, dir, spheres[i], tMin, tMax, spT, spNorm)) {
                return true;
            }
        }
        return false;
    }

    // Fallback for scenes with alpha masks or transmission
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT, 0xFF, origin, tMin, dir, tMax);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
            uint triIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            uint matId = triangles[triIdx].materialId;
            Material mat = materials[matId];
            if (mat.type == 3u /* Skip EMISSIVE */ || mat.type == 2u /* Skip DIELECTRIC */ || mat.transmission > 0.05) {
                continue;
            }
            if (mat.alphaMode == 1u /* MASK */ || mat.alphaMode == 2u /* BLEND */) {
                vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, false);
                float cu = bary.x, cv = bary.y, cw = 1.0 - cu - cv;
                Triangle ctri = triangles[triIdx];
                vec2 cuv = cw * vec2(ctri.v0.position.w, ctri.v0.normal.w) +
                           cu * vec2(ctri.v1.position.w, ctri.v1.normal.w) +
                           cv * vec2(ctri.v2.position.w, ctri.v2.normal.w);
                float calpha = mat.albedo.a;
                if (mat.albedoTex > 0u && mat.albedoTex <= 64u) {
                    calpha *= texture(sceneTextures[nonuniformEXT(mat.albedoTex - 1u)], cuv).a;
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
        uint triIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        Material m = materials[triangles[triIdx].materialId];
        if (m.type != 3u && m.type != 2u && m.transmission <= 0.05) {
            return true;
        }
    }
    for (uint i = 0; i < pc.numSpheres; ++i) {
        if (materials[spheres[i].materialId].type == 3u) continue;
        float spT;
        vec3 spNorm;
        if (intersectSphere(origin, dir, spheres[i], tMin, tMax, spT, spNorm)) {
            return true;
        }
    }
    return false;
}

// PBR Math
vec3 sampleCosineHemisphere(vec3 normal, inout uint seed) {
    vec2 r = randVec2(seed);
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

float fresnelSchlick(float cosTheta, float refIdx) {
    float r0 = (1.0 - refIdx) / (1.0 + refIdx);
    r0 = r0 * r0;
    float x = clamp(1.0 - cosTheta, 0.0, 1.0);
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

// --- ReSTIR DI Octahedral Normal & Geometry Compression ---
vec3 octDecode(vec2 f) {
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = clamp(-n.z, 0.0, 1.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    return normalize(n);
}

uint packGeom(vec3 norm, float depth) {
    vec2 oct = octEncode(norm);
    uint n16 = packSnorm4x8(vec4(oct, 0.0, 0.0)) & 0xFFFFu;
    uint d16 = packHalf2x16(vec2(depth, 0.0)) & 0xFFFFu;
    return (d16 << 16u) | n16;
}

void unpackGeom(uint p, out vec3 norm, out float depth) {
    depth = unpackHalf2x16(p >> 16u).x;
    vec2 oct = unpackSnorm4x8(p & 0xFFFFu).xy;
    norm = octDecode(oct);
}

// Evaluate candidate light target PDF (unshadowed luminance) and emission/PDF
float evalLightCandidate(
    uint lightIdx,
    vec2 uv,
    vec3 hitP,
    vec3 hitN,
    vec3 viewDir,
    vec3 diffCol,
    vec3 f0,
    float alphaRough,
    bool specEnabled,
    out float outLightPdf
) {
    outLightPdf = 0.0;
    if (lightIdx >= pc.numLights) return 0.0;
    Light l = lights[lightIdx];
    vec3 lDir;
    float lDist = 0.0;
    vec3 lEmiss = l.emission.rgb;
    float lPdf = 0.0;

    if (l.position.w == 0.0) {
        // Area light
        vec3 lPos = l.position.xyz + uv.x * l.u.xyz + uv.y * l.v.xyz;
        vec3 toL = lPos - hitP;
        lDist = length(toL);
        lDir = toL / max(lDist, 1e-4);
        float cosL = clamp(dot(-lDir, l.normal.xyz), 0.0, 1.0);
        float lArea = l.emission.w;
        if (cosL > 0.0 && lArea > 0.0) {
            float geomFactor = cosL / max(lDist * lDist, 1e-4);
            lEmiss *= geomFactor;
            lPdf = 1.0 / (lArea * float(pc.numLights));
        }
    } else if (uint(l.position.w) == 1u) {
        // Spot light
        vec3 toL = l.position.xyz - hitP;
        lDist = length(toL);
        lDir = toL / max(lDist, 1e-4);
        float cosSpot = dot(-lDir, l.normal.xyz);
        if (cosSpot >= l.v.w) {
            float spotFactor = clamp((cosSpot - l.v.w) / max(l.u.w - l.v.w, 1e-4), 0.0, 1.0);
            lEmiss *= spotFactor / max(lDist * lDist, 1e-4);
            lPdf = 1.0 / float(pc.numLights);
        }
    }

    outLightPdf = lPdf;
    float nDotL = dot(hitN, lDir);
    if (nDotL <= 0.0 || lPdf <= 0.0) return 0.0;

    vec3 H = normalize(viewDir + lDir);
    float nDotV = clamp(dot(hitN, viewDir), 0.001, 1.0);
    float nDotH = clamp(dot(hitN, H), 0.0, 1.0);
    float vDotH = clamp(dot(viewDir, H), 0.0, 1.0);
    vec3 F = fresnelSchlickVec(vDotH, f0);
    vec3 diffBRDF = (vec3(1.0) - F) * diffCol * INV_PI;
    vec3 specBRDF = vec3(0.0);
    if (specEnabled) {
        float D = distributionGGX(nDotH, alphaRough);
        float Vis = visibilitySmithGGXCorrelated(nDotL, nDotV, alphaRough);
        specBRDF = D * Vis * F;
    }
    vec3 unshadowed = lEmiss * (diffBRDF + specBRDF) * nDotL;
    return dot(unshadowed, vec3(0.2126, 0.7152, 0.0722));
}

void main() {
    bool prevIsDelta = ((prd.packedThroughputB_Flags >> 16u) & 2u) != 0u;
    vec3 accumRadiance = vec3(0.0);

    uint primID = gl_PrimitiveID;
    Triangle tri = triangles[primID];

    float u = attribs.x;
    float v = attribs.y;
    float w = 1.0 - u - v;

    vec3 hitPoint = gl_WorldRayOriginEXT + gl_HitTEXT * gl_WorldRayDirectionEXT;
    vec3 normal = normalize(w * tri.v0.normal.xyz + u * tri.v1.normal.xyz + v * tri.v2.normal.xyz);
    vec2 hitUv = w * vec2(tri.v0.position.w, tri.v0.normal.w) +
                 u * vec2(tri.v1.position.w, tri.v1.normal.w) +
                 v * vec2(tri.v2.position.w, tri.v2.normal.w);

    bool frontFace = dot(gl_WorldRayDirectionEXT, normal) < 0.0;
    vec3 geomNormal = frontFace ? normal : -normal;
    vec3 hitNormal = geomNormal;

    Material mat = materials[tri.materialId];

    // Alpha masking & stochastic alpha blending
    vec4 baseColor = mat.albedo;
    if (mat.albedoTex > 0u && mat.albedoTex <= 64u) {
        baseColor *= texture(sceneTextures[nonuniformEXT(mat.albedoTex - 1u)], hitUv);
    }
    if ((mat.alphaMode == 1u && baseColor.a < mat.alphaCutoff) ||
        (mat.alphaMode == 2u && randFloat(prd.seed) > baseColor.a)) {
        // Transparent / masked pixel - passthrough ray
        prd.radiance = vec3(0.0);
        prd.packedThroughputRG = packHalf2x16(vec2(1.0, 1.0));
        prd.nextOrigin = hitPoint + gl_WorldRayDirectionEXT * EPSILON;
        uint flags = 1u; // hit = true, isDelta = false
        prd.packedThroughputB_Flags = (packHalf2x16(vec2(1.0, 0.0)) & 0xFFFFu) | (flags << 16u);
        prd.packedNextDir = packOct32(gl_WorldRayDirectionEXT);
        prd.lastBsdfPdf = 1.0;
        prd.pad = 0u;
        return;
    }

    // Normal mapping
    if (mat.normalTex > 0u && mat.normalTex <= 64u) {
        vec4 tan0 = tri.v0.tangent;
        vec4 tan1 = tri.v1.tangent;
        vec4 tan2 = tri.v2.tangent;
        vec3 geomTan = normalize(w * tan0.xyz + u * tan1.xyz + v * tan2.xyz);
        float tanSign = tan0.w != 0.0 ? tan0.w : 1.0;

        vec3 normalMap = texture(sceneTextures[nonuniformEXT(mat.normalTex - 1u)], hitUv).rgb * 2.0 - 1.0;
        normalMap.xy *= mat.normalScale;
        normalMap = normalize(normalMap);

        geomTan = normalize(geomTan - dot(geomTan, hitNormal) * hitNormal);
        vec3 geomBitangent = cross(hitNormal, geomTan) * tanSign;
        mat3 tbn = mat3(geomTan, geomBitangent, hitNormal);
        hitNormal = normalize(tbn * normalMap);
    }

    // Material attributes
    float roughness = mat.roughness;
    float metallic = mat.metallic;
    if (mat.mrTex > 0u && mat.mrTex <= 64u) {
        vec4 mrSample = texture(sceneTextures[nonuniformEXT(mat.mrTex - 1u)], hitUv);
        roughness *= mrSample.g;
        metallic *= mrSample.b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    float alphaRoughness = roughness * roughness;

    float transmission = mat.transmission;
    if (mat.transmissionTex > 0u && mat.transmissionTex <= 64u) {
        transmission *= texture(sceneTextures[nonuniformEXT(mat.transmissionTex - 1u)], hitUv).r;
    }

    vec3 emissive = mat.emissive.rgb;
    if (mat.emissiveTex > 0u && mat.emissiveTex <= 64u) {
        emissive *= texture(sceneTextures[nonuniformEXT(mat.emissiveTex - 1u)], hitUv).rgb;
    }

    bool enableDirect = (ubo.flags & (1 << 0)) != 0;
    bool enableSpecular = (ubo.flags & (1 << 2)) != 0;
    bool enableRefraction = (ubo.flags & (1 << 3)) != 0;
    bool enableShadows = (ubo.flags & (1 << 4)) != 0;
    bool enableShadowDenoiser = (ubo.flags & (1u << 20)) != 0u;

    // 1. Emissive contribution with MIS
    if (length(emissive) > 1e-3) {
        float misWeight = 1.0;
        if (!prevIsDelta && pc.numLights > 0u && enableDirect) {
            float lightPdf = 0.0;
            for (uint l = 0; l < pc.numLights; ++l) {
                Light light = lights[l];
                if (light.position.w == 0.0 /* Area Light */) {
                    float cosLight = dot(-gl_WorldRayDirectionEXT, light.normal.xyz);
                    if (cosLight > 0.0) {
                        float lightArea = light.emission.w;
                        if (lightArea > 0.0 && (mat.type == 3u || dot(hitNormal, light.normal.xyz) > 0.9)) {
                            lightPdf = (gl_HitTEXT * gl_HitTEXT) / (cosLight * lightArea * float(pc.numLights));
                            break;
                        }
                    }
                }
            }
            if (lightPdf > 0.0) {
                misWeight = prd.lastBsdfPdf / (prd.lastBsdfPdf + lightPdf);
            }
        }
        accumRadiance = emissive * misWeight;
        if (mat.type == 3u /* Emissive */) {
            bool enableReSTIR = (ubo.flags & (1u << 6)) != 0u;
            bool isPrimary = ((prd.packedThroughputB_Flags >> 16u) & 4u) != 0u;
            if (enableReSTIR && isPrimary) {
                ReservoirDI emptyR;
                emptyR.lightIdx = 0u;
                emptyR.uvX = 0.0;
                emptyR.uvY = 0.0;
                emptyR.wSum = 0.0;
                emptyR.M = 0.0;
                emptyR.W = 0.0;
                emptyR.targetPdf = 0.0;
                emptyR.pad = 0u;
                currentReservoirs[prd.pad] = emptyR;
            }
            prd.radiance = accumRadiance;
            prd.packedThroughputRG = 0u;
            uint flags = 1u; // hit = true, isDelta = false
            prd.packedThroughputB_Flags = flags << 16u;
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
    if (mat.specularTex > 0u && mat.specularTex <= 64u) {
        specFactor *= texture(sceneTextures[nonuniformEXT(mat.specularTex - 1u)], hitUv).a;
    }
    vec3 dielectricF0 = vec3(clamp(f0Dielectric * specFactor, 0.0, 1.0));
    vec3 F0 = mix(dielectricF0, baseColor.rgb, metallic);
    vec3 diffuseColor = baseColor.rgb * (1.0 - metallic);
    vec3 V = -gl_WorldRayDirectionEXT;

    // Clearcoat
    float clearcoat = mat.clearcoat;
    if (mat.clearcoatTex > 0u && mat.clearcoatTex <= 64u) {
        clearcoat *= texture(sceneTextures[nonuniformEXT(mat.clearcoatTex - 1u)], hitUv).r;
    }
    float clearcoatRoughness = mat.clearcoatRoughness;
    if (mat.clearcoatRoughnessTex > 0u && mat.clearcoatRoughnessTex <= 64u) {
        clearcoatRoughness *= texture(sceneTextures[nonuniformEXT(mat.clearcoatRoughnessTex - 1u)], hitUv).g;
    }
    clearcoatRoughness = clamp(clearcoatRoughness, 0.001, 1.0);
    float clearcoatAlpha = clearcoatRoughness * clearcoatRoughness;
    vec3 clearcoatNormal = geomNormal;
    if (mat.clearcoatNormalTex > 0u && mat.clearcoatNormalTex <= 64u) {
        vec4 tan0 = tri.v0.tangent;
        vec4 tan1 = tri.v1.tangent;
        vec4 tan2 = tri.v2.tangent;
        vec3 cGeomTan = normalize(w * tan0.xyz + u * tan1.xyz + v * tan2.xyz);
        float cTanSign = tan0.w != 0.0 ? tan0.w : 1.0;
        if (length(cGeomTan) < 0.1) {
            vec3 up = abs(geomNormal.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
            cGeomTan = normalize(cross(up, geomNormal));
            cTanSign = 1.0;
        }
        vec3 cNormMap = texture(sceneTextures[nonuniformEXT(mat.clearcoatNormalTex - 1u)], hitUv).rgb * 2.0 - 1.0;
        cNormMap = normalize(cNormMap);
        cGeomTan = normalize(cGeomTan - dot(cGeomTan, geomNormal) * geomNormal);
        vec3 cGeomBitangent = cross(geomNormal, cGeomTan) * cTanSign;
        mat3 cTbn = mat3(cGeomTan, cGeomBitangent, geomNormal);
        clearcoatNormal = normalize(cTbn * cNormMap);
    }

    float clearcoatProb = (clearcoat > 0.001 && enableSpecular) ? (clearcoat * 0.25) : 0.0;
    float baseSpecProb = enableSpecular ? clamp(mix(0.04, 1.0, metallic), 0.05, 0.95) * (1.0 - clearcoatProb) : 0.0;

    // 2. Direct Lighting (Analytical Lights with ReSTIR DI or NEE Fallback)
    bool enableReSTIR = (ubo.flags & (1u << 6)) != 0u;
    bool isPrimary = ((prd.packedThroughputB_Flags >> 16u) & 4u) != 0u;
    float hitDepth = length(hitPoint - ubo.position.xyz);

    if (enableDirect && pc.numLights > 0u && mat.type != 3u && transmission < 0.1 && mat.type != 2u) {
        if (enableReSTIR && isPrimary) {
            ReservoirDI R;
            R.lightIdx = 0u;
            R.uvX = 0.0;
            R.uvY = 0.0;
            R.wSum = 0.0;
            R.M = 0.0;
            R.W = 0.0;
            R.targetPdf = 0.0;
            R.pad = 0u;

            // 1. Initial Candidate Generation (M_init = 8 candidates with Chao's WRS)
            const uint M_init = 8u;
            for (uint c = 0u; c < M_init; ++c) {
                uint candIdx = uint(randFloat(prd.seed) * float(pc.numLights)) % pc.numLights;
                vec2 cUv = randVec2(prd.seed);
                float cLightPdf = 0.0;
                float p_hat = evalLightCandidate(
                    candIdx, cUv, hitPoint, hitNormal, V, diffuseColor, F0, alphaRoughness, enableSpecular, cLightPdf
                );

                float w_i = (cLightPdf > 0.0) ? (p_hat / cLightPdf) : 0.0;
                R.wSum += w_i;
                R.M += 1.0;
                if (randFloat(prd.seed) * R.wSum < w_i) {
                    R.lightIdx = candIdx;
                    R.uvX = cUv.x;
                    R.uvY = cUv.y;
                    R.targetPdf = p_hat;
                }
            }

            // 2. Temporal Resampling (History Reprojection & Cross-Bilateral Validation)
            int currentPx = int(prd.pad % pc.tileWidth);
            int currentPy = int(prd.pad / pc.tileWidth);
            ivec2 baseCoord = ivec2(currentPx, currentPy);

            vec4 prevClip = ubo.prevViewProj * vec4(hitPoint, 1.0);
            if (prevClip.w > 0.0) {
                vec2 prevNDC = prevClip.xy / prevClip.w;
                vec2 prevUV = prevNDC * 0.5 + 0.5;
                if (prevUV.x >= 0.0 && prevUV.x <= 1.0 && prevUV.y >= 0.0 && prevUV.y <= 1.0) {
                    int prevPx = clamp(int(prevUV.x * float(pc.tileWidth)), 0, int(pc.tileWidth) - 1);
                    int prevPy = clamp(int(prevUV.y * float(pc.tileHeight)), 0, int(pc.tileHeight) - 1);
                    uint prevIdx = uint(prevPy * int(pc.tileWidth) + prevPx);

                    ReservoirDI R_prev = historyReservoirs[prevIdx];
                    if (R_prev.M > 0.0 && R_prev.wSum > 0.0 && R_prev.lightIdx < pc.numLights) {
                        bool geomValid = true;
                        if (R_prev.pad != 0u) {
                            vec3 prevNormal;
                            float prevDepth;
                            unpackGeom(R_prev.pad, prevNormal, prevDepth);
                            float normDot = dot(hitNormal, prevNormal);
                            float depthDiff = abs(hitDepth - prevDepth) / max(hitDepth, 1e-3);
                            geomValid = (normDot > 0.90 && depthDiff < 0.10);
                        }

                        if (geomValid) {
                            float historyM = R_prev.M;
                            float dummyPdf;
                            float p_hat_current = evalLightCandidate(
                                R_prev.lightIdx, vec2(R_prev.uvX, R_prev.uvY), hitPoint, hitNormal, V, diffuseColor, F0, alphaRoughness, enableSpecular, dummyPdf
                            );

                            float w_temporal = p_hat_current * R_prev.W * historyM;
                            R.wSum += w_temporal;
                            R.M += historyM;
                            if (randFloat(prd.seed) * R.wSum < w_temporal) {
                                R.lightIdx = R_prev.lightIdx;
                                R.uvX = R_prev.uvX;
                                R.uvY = R_prev.uvY;
                                R.targetPdf = p_hat_current;
                            }
                        }
                    }
                }
            }

            // 3. Spatial Resampling (Neighbor Gathering around current pixel with Cross-Bilateral Validation)
            bool enableSpatial = (ubo.flags & (1u << 7)) != 0u;
            if (enableSpatial) {
                uint spatialSamples = (ubo.flags >> 8u) & 0xFu;
                if (spatialSamples == 0u) spatialSamples = 4u;
                float spatialRadius = float((ubo.flags >> 12u) & 0xFFu);
                if (spatialRadius < 1.0) spatialRadius = 16.0;

                // Detect Interleaved Scanline Multi-GPU mode
                int yStride = (pc.tileOffsetY == 0u && pc.tileOffsetX > 0u) ? 2 : 1;

                for (uint i = 0u; i < spatialSamples; ++i) {
                    float angle = randFloat(prd.seed) * TWO_PI;
                    float rad = sqrt(randFloat(prd.seed)) * spatialRadius;
                    ivec2 offset = ivec2(round(vec2(cos(angle), sin(angle)) * rad));
                    if (offset == ivec2(0)) {
                        offset = (randFloat(prd.seed) > 0.5) ? ivec2(1, 0) : ivec2(0, 1);
                    }
                    offset.y *= yStride;
                    ivec2 nbrCoord = clamp(baseCoord + offset, ivec2(0), ivec2(int(pc.tileWidth) - 1, int(pc.tileHeight) - 1));
                    uint nbrIdx = uint(nbrCoord.y * int(pc.tileWidth) + nbrCoord.x);

                    ReservoirDI R_nbr = historyReservoirs[nbrIdx];
                    if (R_nbr.pad != 0u && R_nbr.M > 0.0 && R_nbr.W > 0.0 && R_nbr.lightIdx < pc.numLights) {
                        vec3 nbrNormal;
                        float nbrDepth;
                        unpackGeom(R_nbr.pad, nbrNormal, nbrDepth);

                        float normDot = dot(hitNormal, nbrNormal);
                        float depthDiff = abs(hitDepth - nbrDepth) / max(hitDepth, 1e-3);

                        if (normDot > 0.90 && depthDiff < 0.10) {
                            float nbrM = R_nbr.M;
                            float dummyPdf;
                            float p_hat_nbr = evalLightCandidate(
                                R_nbr.lightIdx, vec2(R_nbr.uvX, R_nbr.uvY), hitPoint, hitNormal, V, diffuseColor, F0, alphaRoughness, enableSpecular, dummyPdf
                            );

                            float w_spatial = p_hat_nbr * R_nbr.W * nbrM;
                            R.wSum += w_spatial;
                            R.M += nbrM;
                            if (randFloat(prd.seed) * R.wSum < w_spatial) {
                                R.lightIdx = R_nbr.lightIdx;
                                R.uvX = R_nbr.uvX;
                                R.uvY = R_nbr.uvY;
                                R.targetPdf = p_hat_nbr;
                            }
                        }
                    }
                }
            }

            // 4. Compute Final Unbiased Weight W with proper M-capping to prevent infinite history lag
            const float M_max = 24.0;
            if (R.M > M_max) {
                R.wSum *= (M_max / R.M);
                R.M = M_max;
            }

            if (R.targetPdf > 0.0 && R.M > 0.0) {
                R.W = R.wSum / (R.M * R.targetPdf);
            } else {
                R.W = 0.0;
            }
            R.pad = packGeom(hitNormal, hitDepth);
            currentReservoirs[prd.pad] = R;

            // 5. Deferred Shadow Ray Query for Winning Candidate
            if (R.W > 0.0 && R.lightIdx < pc.numLights) {
                Light wLight = lights[R.lightIdx];
                vec3 wLightDir;
                float wLightDist = 0.0;
                vec3 wLightEmiss = wLight.emission.rgb;
                bool validSample = true;

                if (wLight.position.w == 0.0) {
                    vec3 wPos = wLight.position.xyz + R.uvX * wLight.u.xyz + R.uvY * wLight.v.xyz;
                    vec3 toL = wPos - hitPoint;
                    wLightDist = length(toL);
                    wLightDir = toL / max(wLightDist, 1e-4);
                    float cosL = clamp(dot(-wLightDir, wLight.normal.xyz), 0.0, 1.0);
                    if (cosL <= 0.0) validSample = false;
                    else {
                        float geomFactor = cosL / max(wLightDist * wLightDist, 1e-4);
                        wLightEmiss *= geomFactor;
                    }
                } else if (uint(wLight.position.w) == 1u) {
                    vec3 toL = wLight.position.xyz - hitPoint;
                    wLightDist = length(toL);
                    wLightDir = toL / max(wLightDist, 1e-4);
                    float cosSpot = dot(-wLightDir, wLight.normal.xyz);
                    if (cosSpot < wLight.v.w) validSample = false;
                    else {
                        float spotFactor = clamp((cosSpot - wLight.v.w) / max(wLight.u.w - wLight.v.w, 1e-4), 0.0, 1.0);
                        wLightEmiss *= spotFactor / max(wLightDist * wLightDist, 1e-4);
                    }
                }

                float wNdotL = dot(hitNormal, wLightDir);
                if (validSample && wNdotL > 0.0) {
                    bool inShadow = enableShadows ? isShadowOccluded(hitPoint + hitNormal * EPSILON, wLightDir, EPSILON, wLightDist - EPSILON * 2.0) : false;
                    if (!inShadow || enableShadowDenoiser) {
                        vec3 H = normalize(V + wLightDir);
                        float NdotV = clamp(dot(hitNormal, V), 0.001, 1.0);
                        float NdotH = clamp(dot(hitNormal, H), 0.0, 1.0);
                        float VdotH = clamp(dot(V, H), 0.0, 1.0);
                        vec3 F = fresnelSchlickVec(VdotH, F0);
                        vec3 diffBRDF = (vec3(1.0) - F) * diffuseColor * INV_PI;
                        vec3 specBRDF = vec3(0.0);
                        if (enableSpecular) {
                            float D = distributionGGX(NdotH, alphaRoughness);
                            float Vis = visibilitySmithGGXCorrelated(wNdotL, NdotV, alphaRoughness);
                            specBRDF = D * Vis * F;
                        }
                        vec3 brdf = diffBRDF + specBRDF;
                        vec3 directUnshadowed = wLightEmiss * brdf * wNdotL * R.W;
                        if (enableShadowDenoiser) {
                            float rawShadow = inShadow ? 0.0 : 1.0;
                            imageStore(uDirectLightImage, baseCoord, vec4(directUnshadowed, rawShadow));
                            imageStore(uNormalDepthImage, baseCoord, vec4(hitNormal, hitDepth));
                        } else {
                            accumRadiance += directUnshadowed;
                        }
                    }
                }
            }
        } else if (enableReSTIR) {
            // Secondary Bounces with Resampled Importance Sampling (RIS M=4) to suppress indirect fireflies
            uint bestLight = 0u;
            vec2 bestUv = vec2(0.5);
            float wSum = 0.0;
            float bestTargetPdf = 0.0;
            const uint M_sec = 4u;
            for (uint c = 0u; c < M_sec; ++c) {
                uint candIdx = uint(randFloat(prd.seed) * float(pc.numLights)) % pc.numLights;
                vec2 cUv = randVec2(prd.seed);
                float cPdf = 0.0;
                float p_hat = evalLightCandidate(candIdx, cUv, hitPoint, hitNormal, V, diffuseColor, F0, alphaRoughness, enableSpecular, cPdf);
                float wi = (cPdf > 0.0) ? (p_hat / cPdf) : 0.0;
                wSum += wi;
                if (randFloat(prd.seed) * wSum < wi) {
                    bestLight = candIdx;
                    bestUv = cUv;
                    bestTargetPdf = p_hat;
                }
            }
            float W_sec = (bestTargetPdf > 0.0) ? (wSum / (float(M_sec) * bestTargetPdf)) : 0.0;
            if (W_sec > 0.0 && bestLight < pc.numLights) {
                Light wLight = lights[bestLight];
                vec3 wLightDir;
                float wLightDist = 0.0;
                vec3 wLightEmiss = wLight.emission.rgb;
                bool validSample = true;
                if (wLight.position.w == 0.0) {
                    vec3 wPos = wLight.position.xyz + bestUv.x * wLight.u.xyz + bestUv.y * wLight.v.xyz;
                    vec3 toL = wPos - hitPoint;
                    wLightDist = length(toL);
                    wLightDir = toL / max(wLightDist, 1e-4);
                    float cosL = clamp(dot(-wLightDir, wLight.normal.xyz), 0.0, 1.0);
                    if (cosL <= 0.0) validSample = false;
                    else {
                        float geomFactor = cosL / max(wLightDist * wLightDist, 1e-4);
                        wLightEmiss *= geomFactor;
                    }
                } else if (uint(wLight.position.w) == 1u) {
                    vec3 toL = wLight.position.xyz - hitPoint;
                    wLightDist = length(toL);
                    wLightDir = toL / max(wLightDist, 1e-4);
                    float cosSpot = dot(-wLightDir, wLight.normal.xyz);
                    if (cosSpot < wLight.v.w) validSample = false;
                    else {
                        float spotFactor = clamp((cosSpot - wLight.v.w) / max(wLight.u.w - wLight.v.w, 1e-4), 0.0, 1.0);
                        wLightEmiss *= spotFactor / max(wLightDist * wLightDist, 1e-4);
                    }
                }
                float wNdotL = dot(hitNormal, wLightDir);
                if (validSample && wNdotL > 0.0) {
                    bool inShadow = enableShadows ? isShadowOccluded(hitPoint + hitNormal * EPSILON, wLightDir, EPSILON, wLightDist - EPSILON * 2.0) : false;
                    if (!inShadow) {
                        vec3 H = normalize(V + wLightDir);
                        float NdotV = clamp(dot(hitNormal, V), 0.001, 1.0);
                        float NdotH = clamp(dot(hitNormal, H), 0.0, 1.0);
                        float VdotH = clamp(dot(V, H), 0.0, 1.0);
                        vec3 F = fresnelSchlickVec(VdotH, F0);
                        vec3 diffBRDF = (vec3(1.0) - F) * diffuseColor * INV_PI;
                        vec3 specBRDF = vec3(0.0);
                        if (enableSpecular) {
                            float D = distributionGGX(NdotH, alphaRoughness);
                            float Vis = visibilitySmithGGXCorrelated(wNdotL, NdotV, alphaRoughness);
                            specBRDF = D * Vis * F;
                        }
                        vec3 brdf = diffBRDF + specBRDF;
                        accumRadiance += wLightEmiss * brdf * wNdotL * W_sec;
                    }
                }
            }
        } else {
            // Standard Uniform Light Picking NEE with MIS (Baseline fallback when ReSTIR disabled)
            uint lightIdx = uint(randFloat(prd.seed) * float(pc.numLights)) % pc.numLights;
            Light light = lights[lightIdx];

            vec3 lightDir;
            float lightDist;
            vec3 lightEmission = light.emission.rgb;
            float lightPdf = 1.0;

            if (light.position.w == 0.0) {
                // Area light
                vec2 lightUv = randVec2(prd.seed);
                vec3 lightSamplePos = light.position.xyz + lightUv.x * light.u.xyz + lightUv.y * light.v.xyz;
                vec3 toLight = lightSamplePos - hitPoint;
                lightDist = length(toLight);
                lightDir = toLight / max(lightDist, 1e-4);

                float lightArea = light.emission.w;
                float cosLight = dot(-lightDir, light.normal.xyz);
                if (cosLight > 0.0 && lightArea > 0.0) {
                    lightPdf = (lightDist * lightDist) / (cosLight * lightArea * float(pc.numLights));
                } else {
                    lightPdf = 0.0;
                }
            } else if (uint(light.position.w) == 1u /* SPOT */) {
                vec3 toLight = light.position.xyz - hitPoint;
                lightDist = length(toLight);
                lightDir = toLight / max(lightDist, 1e-4);

                float cosSpot = dot(-lightDir, light.normal.xyz);
                float innerCos = light.u.w;
                float outerCos = light.v.w;
                if (cosSpot < outerCos) {
                    lightPdf = 0.0;
                } else {
                    float spotFactor = clamp((cosSpot - outerCos) / max(innerCos - outerCos, 1e-4), 0.0, 1.0);
                    lightEmission *= spotFactor / max(lightDist * lightDist, 1e-4);
                    lightPdf = 1.0 / float(pc.numLights);
                }
            } else {
                lightPdf = 0.0;
            }

            float NdotL = dot(hitNormal, lightDir);
            if (NdotL > 0.0 && lightPdf > 0.0) {
                // Inline shadow test using hardware ray query (bypassed if shadows disabled)
                bool inShadow = enableShadows ? isShadowOccluded(hitPoint + hitNormal * EPSILON, lightDir, EPSILON, lightDist - EPSILON * 2.0) : false;
                if (!inShadow || (enableShadowDenoiser && isPrimary)) {
                    vec3 H = normalize(V + lightDir);
                    float NdotV = clamp(dot(hitNormal, V), 0.001, 1.0);
                    float NdotH = clamp(dot(hitNormal, H), 0.0, 1.0);
                    float VdotH = clamp(dot(V, H), 0.0, 1.0);

                    vec3 F = fresnelSchlickVec(VdotH, F0);
                    vec3 diffBRDF = (vec3(1.0) - F) * diffuseColor * INV_PI;
                    vec3 specBRDF = vec3(0.0);
                    if (enableSpecular) {
                        float D = distributionGGX(NdotH, alphaRoughness);
                        float Vis = visibilitySmithGGXCorrelated(NdotL, NdotV, alphaRoughness);
                        specBRDF = D * Vis * F;
                    }

                    vec3 brdf = diffBRDF + specBRDF;

                    // MIS balance heuristic for NEE
                    float misWeightLight = 1.0;
                    if (light.position.w == 0.0 /* Area Light */) {
                        float bsdfPdf = evalBSDFPdf(V, lightDir, hitNormal, clearcoatNormal, alphaRoughness, clearcoatAlpha, clearcoatProb, baseSpecProb, clearcoat);
                        misWeightLight = lightPdf / (lightPdf + bsdfPdf);
                    }

                    vec3 directUnshadowed = lightEmission * brdf * NdotL * misWeightLight / lightPdf;
                    if (enableShadowDenoiser && isPrimary) {
                        int currentPx = int(prd.pad % pc.tileWidth);
                        int currentPy = int(prd.pad / pc.tileWidth);
                        ivec2 baseCoord = ivec2(currentPx, currentPy);
                        float rawShadow = inShadow ? 0.0 : 1.0;
                        imageStore(uDirectLightImage, baseCoord, vec4(directUnshadowed, rawShadow));
                        imageStore(uNormalDepthImage, baseCoord, vec4(hitNormal, hitDepth));
                    } else {
                        accumRadiance += directUnshadowed;
                    }
                }
            }
        }
    } else if (enableReSTIR && isPrimary) {
        // Direct lighting disabled or non-diffuse surface; clear reservoir
        ReservoirDI emptyR;
        emptyR.lightIdx = 0u;
        emptyR.uvX = 0.0;
        emptyR.uvY = 0.0;
        emptyR.wSum = 0.0;
        emptyR.M = 0.0;
        emptyR.W = 0.0;
        emptyR.targetPdf = 0.0;
        emptyR.pad = 0u;
        currentReservoirs[prd.pad] = emptyR;
    }

    // 3. BSDF Sampling for Next Direction
    vec3 nextDirection;
    vec3 throughputFactor;

    if (transmission > 0.01 || mat.type == 2u) {
        if (!enableRefraction) {
            nextDirection = sampleCosineHemisphere(hitNormal, prd.seed);
            throughputFactor = baseColor.rgb;
        } else {
            float refractionRatio = frontFace ? (1.0 / iorVal) : iorVal;
            vec3 unitDir = normalize(gl_WorldRayDirectionEXT);
            float cosTheta = min(dot(-unitDir, hitNormal), 1.0);
            float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

            bool cannotRefract = refractionRatio * sinTheta > 1.0;
            float reflectProb = fresnelSchlick(cosTheta, refractionRatio);

            if (cannotRefract || reflectProb > randFloat(prd.seed)) {
                nextDirection = reflect(unitDir, hitNormal);
                prd.nextOrigin = hitPoint + hitNormal * EPSILON;
            } else {
                nextDirection = refract(unitDir, hitNormal, refractionRatio);
                prd.nextOrigin = hitPoint - hitNormal * EPSILON;
            }
            prd.radiance = accumRadiance;
            prd.packedThroughputRG = packHalf2x16(baseColor.rg);
            uint flags = 1u | 2u; // hit = true, isDelta = true
            prd.packedThroughputB_Flags = (packHalf2x16(vec2(baseColor.b, 0.0)) & 0xFFFFu) | (flags << 16u);
            prd.packedNextDir = packOct32(normalize(nextDirection));
            prd.lastBsdfPdf = 1.0;
            prd.pad = 0u;
            return;
        }
    } else {
        float xi = randFloat(prd.seed);

        bool sampleSpecular = false;
        if (xi < clearcoatProb) {
            vec3 Hc = sampleGGX(clearcoatNormal, clearcoatAlpha, prd.seed);
            vec3 L = reflect(-V, Hc);
            float NcDotL = dot(clearcoatNormal, L);

            if (NcDotL > 0.0) {
                float NcDotV = clamp(dot(clearcoatNormal, V), 0.001, 1.0);
                float NcDotH = clamp(dot(clearcoatNormal, Hc), 0.001, 1.0);
                float VDotHc = clamp(dot(V, Hc), 0.0, 1.0);

                float VisC = visibilitySmithGGXCorrelated(NcDotL, NcDotV, clearcoatAlpha);
                float Fc = fresnelSchlick(VDotHc, 1.5) * clearcoat;

                vec3 specWeight = vec3((4.0 * VisC * Fc * NcDotL * VDotHc) / max(NcDotH * clearcoatProb, 1e-4));
                throughputFactor = clamp(specWeight, vec3(0.0), vec3(10.0));
                nextDirection = L;
                sampleSpecular = true;
            }
        } else if (xi < clearcoatProb + baseSpecProb) {
            vec3 H = sampleGGX(hitNormal, alphaRoughness, prd.seed);
            vec3 L = reflect(-V, H);
            float NdotL = dot(hitNormal, L);

            if (NdotL > 0.0) {
                float NdotV = clamp(dot(hitNormal, V), 0.001, 1.0);
                float NdotH = clamp(dot(hitNormal, H), 0.0, 1.0);
                float VdotH = clamp(dot(V, H), 0.0, 1.0);

                float Vis = visibilitySmithGGXCorrelated(NdotL, NdotV, alphaRoughness);
                vec3 F = fresnelSchlickVec(VdotH, F0);
                float Fc = fresnelSchlick(VdotH, 1.5) * clearcoat;

                vec3 specWeight = (4.0 * Vis * F * NdotL * VdotH * (1.0 - Fc)) / max(NdotH * baseSpecProb, 1e-4);
                throughputFactor = clamp(specWeight, vec3(0.0), vec3(10.0));
                nextDirection = L;
                sampleSpecular = true;
            }
        }

        if (!sampleSpecular) {
            nextDirection = sampleCosineHemisphere(hitNormal, prd.seed);
            vec3 H_diff = normalize(V + nextDirection);
            float VdotH_diff = clamp(dot(V, H_diff), 0.0, 1.0);
            vec3 F_diff = fresnelSchlickVec(VdotH_diff, F0);
            float Fc = fresnelSchlick(VdotH_diff, 1.5) * clearcoat;
            throughputFactor = (1.0 - Fc) * (vec3(1.0) - F_diff) * diffuseColor / max(1.0 - clearcoatProb - baseSpecProb, 1e-4);
        }
    }

    prd.radiance = accumRadiance;
    prd.nextOrigin = hitPoint + hitNormal * EPSILON;
    prd.packedNextDir = packOct32(normalize(nextDirection));
    prd.packedThroughputRG = packHalf2x16(throughputFactor.rg);
    uint flags = 1u; // hit = true, isDelta = false
    prd.packedThroughputB_Flags = (packHalf2x16(vec2(throughputFactor.b, 0.0)) & 0xFFFFu) | (flags << 16u);
    prd.lastBsdfPdf = evalBSDFPdf(V, normalize(nextDirection), hitNormal, clearcoatNormal, alphaRoughness, clearcoatAlpha, clearcoatProb, baseSpecProb, clearcoat);
    prd.pad = 0u;
}
