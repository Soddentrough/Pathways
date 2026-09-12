#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_ray_query : enable
#extension GL_EXT_nonuniform_qualifier : enable

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692
#define INV_PI 0.31830988618379067154
#define EPSILON 0.001

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
    vec4 position; // xyz: pos/corner, w: type
    vec4 emission; // rgb: color, w: area
    vec4 u;        // xyz: edge1, w: spot inner cos
    vec4 v;        // xyz: edge2, w: spot outer cos
    vec4 normal;   // xyz: normal/dir, w: padding
    vec4 sampling; // x: q, y: aliasIdx, z: pdf, w: flux
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
layout(binding = 8) uniform sampler2D sceneTextures[512];

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
    float fractionalSpp;
    uint totalCompositeSpp;
    uint numOpaqueTriangles;
    uint pad2;
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
            uint geomIdx = rayQueryGetIntersectionGeometryIndexEXT(rq, false);
            uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            uint triIdx = (geomIdx == 0u) ? primIdx : (primIdx + pc.numOpaqueTriangles);
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
                if (mat.albedoTex > 0u && mat.albedoTex <= 512u) {
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
        uint geomIdx = rayQueryGetIntersectionGeometryIndexEXT(rq, true);
        uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        uint triIdx = (geomIdx == 0u) ? primIdx : (primIdx + pc.numOpaqueTriangles);
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


void main() {
    bool prevIsDelta = ((prd.packedThroughputB_Flags >> 16u) & 2u) != 0u;
    vec3 accumRadiance = vec3(0.0);
    vec3 accumDiffuse = vec3(0.0);
    vec3 accumSpecular = vec3(0.0);

    uint primID = (gl_GeometryIndexEXT == 0u) ? gl_PrimitiveID : (gl_PrimitiveID + pc.numOpaqueTriangles);
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
    if (mat.albedoTex > 0u && mat.albedoTex <= 512u) {
        baseColor *= texture(sceneTextures[nonuniformEXT(mat.albedoTex - 1u)], hitUv);
    }
    if ((mat.alphaMode == 1u && baseColor.a < mat.alphaCutoff) ||
        (mat.alphaMode == 2u && randFloat(prd.seed) > baseColor.a)) {
        // Transparent / masked pixel - passthrough ray
        prd.diffuseRadiance = vec3(0.0);
        prd.specularRadiance = vec3(0.0);
        prd.packedAlbedoRG = 0u;
        prd.packedAlbedoB_Roughness = 0u;
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
    if (mat.normalTex > 0u && mat.normalTex <= 512u) {
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
    if (mat.mrTex > 0u && mat.mrTex <= 512u) {
        vec4 mrSample = texture(sceneTextures[nonuniformEXT(mat.mrTex - 1u)], hitUv);
        roughness *= mrSample.g;
        metallic *= mrSample.b;
    }
    roughness = clamp(roughness, 0.04, 1.0);
    float alphaRoughness = roughness * roughness;

    float transmission = mat.transmission;
    if (mat.transmissionTex > 0u && mat.transmissionTex <= 512u) {
        transmission *= texture(sceneTextures[nonuniformEXT(mat.transmissionTex - 1u)], hitUv).r;
    }

    vec3 emissive = mat.emissive.rgb;
    if (mat.emissiveTex > 0u && mat.emissiveTex <= 512u) {
        emissive *= texture(sceneTextures[nonuniformEXT(mat.emissiveTex - 1u)], hitUv).rgb;
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
        accumSpecular = emissive * misWeight;
        if (mat.type == 3u /* Emissive */) {
            bool isPrimary = ((prd.packedThroughputB_Flags >> 16u) & 4u) != 0u;
            if (isPrimary) {
                int currentPx = int(prd.pad % pc.tileWidth);
                int currentPy = int(prd.pad / pc.tileWidth);
                ivec2 baseCoord = ivec2(currentPx, currentPy);
                imageStore(uNormalDepthImage, baseCoord, vec4(geomNormal, length(hitPoint - ubo.position.xyz)));
            }
            vec3 specOut = accumRadiance * transmittance;
            if (!isPrimary) {
                const float MAX_INDIRECT_LUMINANCE = 35.0;
                float eLum = dot(specOut, vec3(0.2126, 0.7152, 0.0722));
                if (eLum > MAX_INDIRECT_LUMINANCE) {
                    specOut *= (MAX_INDIRECT_LUMINANCE / eLum);
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
    if (mat.specularTex > 0u && mat.specularTex <= 512u) {
        specFactor *= texture(sceneTextures[nonuniformEXT(mat.specularTex - 1u)], hitUv).a;
    }
    vec3 dielectricF0 = vec3(clamp(f0Dielectric * specFactor, 0.0, 1.0));
    vec3 F0 = mix(dielectricF0, baseColor.rgb, metallic);
    vec3 diffuseColor = baseColor.rgb * (1.0 - metallic);
    vec3 V = -gl_WorldRayDirectionEXT;

    // Clearcoat
    float clearcoat = mat.clearcoat;
    if (mat.clearcoatTex > 0u && mat.clearcoatTex <= 512u) {
        clearcoat *= texture(sceneTextures[nonuniformEXT(mat.clearcoatTex - 1u)], hitUv).r;
    }
    float clearcoatRoughness = mat.clearcoatRoughness;
    if (mat.clearcoatRoughnessTex > 0u && mat.clearcoatRoughnessTex <= 512u) {
        clearcoatRoughness *= texture(sceneTextures[nonuniformEXT(mat.clearcoatRoughnessTex - 1u)], hitUv).g;
    }
    clearcoatRoughness = clamp(clearcoatRoughness, 0.001, 1.0);
    float clearcoatAlpha = clearcoatRoughness * clearcoatRoughness;
    vec3 clearcoatNormal = geomNormal;
    if (mat.clearcoatNormalTex > 0u && mat.clearcoatNormalTex <= 512u) {
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

    // 2. Direct Lighting (Analytical Lights with Uniform NEE and MIS)
    bool isPrimary = ((prd.packedThroughputB_Flags >> 16u) & 4u) != 0u;
    float hitDepth = length(hitPoint - ubo.position.xyz);

    if (enableDirect && pc.numLights > 0u && mat.type != 3u && transmission < 0.1 && mat.type != 2u) {
        // Standard Uniform Light Picking NEE with MIS
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
        } else if (uint(light.position.w) == 2u /* DIRECTIONAL */) {
            lightDir = normalize(light.normal.xyz);
            lightDist = 10000.0;
            lightPdf = 1.0 / float(pc.numLights);
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

                vec3 directDiffuse = lightEmission * diffBRDF * NdotL * misWeightLight / lightPdf;
                vec3 directSpecular = lightEmission * specBRDF * NdotL * misWeightLight / lightPdf;
                vec3 directUnshadowed = directDiffuse + directSpecular;
                if (enableShadowDenoiser && isPrimary) {
                    int currentPx = int(prd.pad % pc.tileWidth);
                    int currentPy = int(prd.pad / pc.tileWidth);
                    ivec2 baseCoord = ivec2(currentPx, currentPy);
                    float rawShadow = inShadow ? 0.0 : 1.0;
                    imageStore(uDirectLightImage, baseCoord, vec4(directUnshadowed, rawShadow));
                    imageStore(uNormalDepthImage, baseCoord, vec4(hitNormal, hitDepth));
                } else {
                    accumRadiance += directUnshadowed;
                    accumDiffuse += directDiffuse;
                    accumSpecular += directSpecular;
                }
            }
        }
    }

    accumRadiance *= transmittance;
    accumDiffuse *= transmittance;
    accumSpecular *= transmittance;

    // 3. BSDF Sampling for Next Direction
    vec3 nextDirection;
    vec3 throughputFactor;
    bool sampleSpecular = false;

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
            if (isPrimary) {
                int currentPx = int(prd.pad % pc.tileWidth);
                int currentPy = int(prd.pad / pc.tileWidth);
                ivec2 baseCoord = ivec2(currentPx, currentPy);
                imageStore(uNormalDepthImage, baseCoord, vec4(hitNormal, hitDepth));
            }
            prd.diffuseRadiance = vec3(0.0);
            prd.specularRadiance = accumRadiance;
            if (isPrimary) {
                prd.packedAlbedoRG = packHalf2x16(baseColor.rg);
                prd.packedAlbedoB_Roughness = packHalf2x16(vec2(baseColor.b, roughness));
            } else {
                prd.packedAlbedoRG = 0u;
                prd.packedAlbedoB_Roughness = 0u;
            }
            prd.packedThroughputRG = packHalf2x16(baseColor.rg * transmittance.rg);
            uint flags = 1u | 2u | 8u; // hit = true, isDelta = true, isSpecular = true
            prd.packedThroughputB_Flags = (packHalf2x16(vec2(baseColor.b * transmittance.b, 0.0)) & 0xFFFFu) | (flags << 16u);
            prd.packedNextDir = packOct32(normalize(nextDirection));
            prd.lastBsdfPdf = 1.0;
            prd.pad = 0u;
            return;
        }
    } else {
        float xi = randFloat(prd.seed);

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

    if (isPrimary) {
        int currentPx = int(prd.pad % pc.tileWidth);
        int currentPy = int(prd.pad / pc.tileWidth);
        ivec2 baseCoord = ivec2(currentPx, currentPy);
        imageStore(uNormalDepthImage, baseCoord, vec4(hitNormal, hitDepth));
        prd.packedAlbedoRG = packHalf2x16(baseColor.rg);
        prd.packedAlbedoB_Roughness = packHalf2x16(vec2(baseColor.b, roughness));
    } else {
        prd.packedAlbedoRG = 0u;
        prd.packedAlbedoB_Roughness = 0u;
    }

    if (!isPrimary) {
        const float MAX_INDIRECT_LUMINANCE = 35.0;
        float dLum = dot(accumDiffuse, vec3(0.2126, 0.7152, 0.0722));
        if (dLum > MAX_INDIRECT_LUMINANCE) {
            accumDiffuse *= (MAX_INDIRECT_LUMINANCE / dLum);
        }
        float sLum = dot(accumSpecular, vec3(0.2126, 0.7152, 0.0722));
        if (sLum > MAX_INDIRECT_LUMINANCE) {
            accumSpecular *= (MAX_INDIRECT_LUMINANCE / sLum);
        }
    }

    prd.diffuseRadiance = accumDiffuse;
    prd.specularRadiance = accumSpecular;
    prd.nextOrigin = hitPoint + hitNormal * EPSILON;
    prd.packedNextDir = packOct32(normalize(nextDirection));
    prd.packedThroughputRG = packHalf2x16(throughputFactor.rg * transmittance.rg);
    uint flags = 1u | (sampleSpecular ? 8u : 0u); // hit = true, isDelta = false, bit 3 = isSpecular
    prd.packedThroughputB_Flags = (packHalf2x16(vec2(throughputFactor.b * transmittance.b, 0.0)) & 0xFFFFu) | (flags << 16u);
    prd.lastBsdfPdf = evalBSDFPdf(V, normalize(nextDirection), hitNormal, clearcoatNormal, alphaRoughness, clearcoatAlpha, clearcoatProb, baseSpecProb, clearcoat);
    prd.pad = 0u;
}
