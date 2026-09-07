#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : enable

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692
#define INV_PI 0.31830988618379067154
#define EPSILON 0.001

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
    float lastBsdfPdf;
    bool hasShadowRay;
    bool hit;
    bool isDelta;
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

layout(binding = 8) uniform sampler2D sceneTextures[64];

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
    return r0 + (1.0 - r0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

vec3 fresnelSchlickVec(float cosTheta, vec3 F0) {
    return F0 + (vec3(1.0) - F0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
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
    prd.hit = true;
    prd.hasShadowRay = false;
    prd.radiance = vec3(0.0);
    prd.shadowLightRad = vec3(0.0);

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

    // Alpha masking
    vec4 baseColor = mat.albedo;
    if (mat.albedoTex > 0u && mat.albedoTex <= 64u) {
        baseColor *= texture(sceneTextures[nonuniformEXT(mat.albedoTex - 1u)], hitUv);
    }
    if (mat.alphaMode == 1u && baseColor.a < mat.alphaCutoff) {
        // Transparent / masked pixel - passthrough ray
        prd.radiance = vec3(0.0);
        prd.throughputMod = vec3(1.0);
        prd.nextOrigin = hitPoint + gl_WorldRayDirectionEXT * EPSILON;
        prd.nextDirection = gl_WorldRayDirectionEXT;
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

    // 1. Emissive contribution with MIS
    if (length(emissive) > 1e-3) {
        float misWeight = 1.0;
        if (!prd.isDelta && pc.numLights > 0u && enableDirect) {
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
        prd.radiance = emissive * misWeight;
        if (mat.type == 3u /* Emissive */) {
            prd.throughputMod = vec3(0.0);
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
    vec3 clearcoatNormal = hitNormal;

    float clearcoatProb = (clearcoat > 0.001 && enableSpecular) ? (clearcoat * 0.25) : 0.0;
    float baseSpecProb = enableSpecular ? clamp(mix(0.04, 1.0, metallic), 0.05, 0.95) * (1.0 - clearcoatProb) : 0.0;

    // 2. Direct Lighting (Analytical Lights with MIS)
    if (enableDirect && pc.numLights > 0u && mat.type != 3u && transmission < 0.1 && mat.type != 2u) {
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
            vec3 H = normalize(V + lightDir);
            float NdotV = clamp(dot(hitNormal, V), 0.001, 1.0);
            float NdotH = clamp(dot(hitNormal, H), 0.0, 1.0);
            float VdotH = clamp(dot(V, H), 0.0, 1.0);

            float D = distributionGGX(NdotH, alphaRoughness);
            float Vis = visibilitySmithGGXCorrelated(NdotL, NdotV, alphaRoughness);
            vec3 F = fresnelSchlickVec(VdotH, F0);

            vec3 specBRDF = enableSpecular ? (D * Vis * F) : vec3(0.0);
            vec3 diffBRDF = (vec3(1.0) - F) * diffuseColor * INV_PI;

            vec3 brdf = diffBRDF + specBRDF;

            // Multiple Importance Sampling (MIS) balance heuristic for NEE
            float misWeightLight = 1.0;
            if (light.position.w == 0.0 /* Area Light */) {
                float bsdfPdf = evalBSDFPdf(V, lightDir, hitNormal, clearcoatNormal, alphaRoughness, clearcoatAlpha, clearcoatProb, baseSpecProb, clearcoat);
                misWeightLight = lightPdf / (lightPdf + bsdfPdf);
            }

            prd.shadowOrigin = hitPoint + hitNormal * EPSILON;
            prd.shadowDir = lightDir;
            prd.shadowDist = lightDist - EPSILON * 2.0;
            prd.shadowLightRad = lightEmission * brdf * NdotL * misWeightLight / lightPdf;
            prd.hasShadowRay = true;
        }
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
            prd.nextDirection = normalize(nextDirection);
            prd.throughputMod = baseColor.rgb;
            prd.isDelta = true;
            prd.lastBsdfPdf = 1.0;
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
            } else {
                nextDirection = sampleCosineHemisphere(hitNormal, prd.seed);
                vec3 H_diff = normalize(V + nextDirection);
                vec3 F_diff = fresnelSchlickVec(clamp(dot(V, H_diff), 0.0, 1.0), F0);
                throughputFactor = (vec3(1.0) - F_diff) * diffuseColor / max(1.0 - clearcoatProb - baseSpecProb, 1e-4);
            }
        } else if (xi < clearcoatProb + baseSpecProb) {
            vec3 H = sampleGGX(hitNormal, alphaRoughness, prd.seed);
            vec3 L = reflect(-V, H);
            float NdotL = dot(hitNormal, L);

            if (NdotL > 0.0) {
                float NdotV = clamp(dot(hitNormal, V), 0.001, 1.0);
                float NdotH = clamp(dot(hitNormal, H), 0.001, 1.0);
                float VdotH = clamp(dot(V, H), 0.0, 1.0);

                float Vis = visibilitySmithGGXCorrelated(NdotL, NdotV, alphaRoughness);
                vec3 F = fresnelSchlickVec(VdotH, F0);
                float Fc = fresnelSchlick(VdotH, 1.5) * clearcoat;

                vec3 specWeight = (4.0 * Vis * F * NdotL * VdotH * (1.0 - Fc)) / max(NdotH * baseSpecProb, 1e-4);
                throughputFactor = clamp(specWeight, vec3(0.0), vec3(10.0));
                nextDirection = L;
            } else {
                nextDirection = sampleCosineHemisphere(hitNormal, prd.seed);
                vec3 H_diff = normalize(V + nextDirection);
                vec3 F_diff = fresnelSchlickVec(clamp(dot(V, H_diff), 0.0, 1.0), F0);
                float Fc = fresnelSchlick(clamp(dot(V, H_diff), 0.0, 1.0), 1.5) * clearcoat;
                throughputFactor = (1.0 - Fc) * (vec3(1.0) - F_diff) * diffuseColor / max(1.0 - clearcoatProb - baseSpecProb, 1e-4);
            }
        } else {
            nextDirection = sampleCosineHemisphere(hitNormal, prd.seed);
            vec3 H_diff = normalize(V + nextDirection);
            vec3 F_diff = fresnelSchlickVec(clamp(dot(V, H_diff), 0.0, 1.0), F0);
            float Fc = fresnelSchlick(clamp(dot(V, H_diff), 0.0, 1.0), 1.5) * clearcoat;
            throughputFactor = (1.0 - Fc) * (vec3(1.0) - F_diff) * diffuseColor / max(1.0 - clearcoatProb - baseSpecProb, 1e-4);
        }
    }

    prd.nextOrigin = hitPoint + hitNormal * EPSILON;
    prd.nextDirection = normalize(nextDirection);
    prd.throughputMod = throughputFactor;
    prd.isDelta = false;
    prd.lastBsdfPdf = evalBSDFPdf(V, prd.nextDirection, hitNormal, clearcoatNormal, alphaRoughness, clearcoatAlpha, clearcoatProb, baseSpecProb, clearcoat);
}
