#ifndef WAVEFRONT_RESTIR_GLSL
#define WAVEFRONT_RESTIR_GLSL

// -----------------------------------------------------------------------------
// Reservoir GPU Memory Layout (32 Bytes Aligned)
// -----------------------------------------------------------------------------
struct ReservoirDI {
    uint lightIdx;
    float uvX;
    float uvY;
    float wSum;
    float M;
    float W;
    float targetPdf;
    uint pad; // Packed geometry: lower 16 bits = oct normal, upper 16 bits = half depth
};

layout(std430, binding = 20) buffer CurrentReservoirBuffer {
    ReservoirDI currentReservoirs[];
};

layout(std430, binding = 21) readonly buffer HistoryReservoirBuffer {
    ReservoirDI historyReservoirs[];
};

// -----------------------------------------------------------------------------
// Geometric Normal & Depth Packing
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Candidate Target Function Evaluation
// -----------------------------------------------------------------------------
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

    if (uint(l.position.w) == 0u /* AREA QUAD */) {
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
    } else if (uint(l.position.w) == 1u /* SPOT */) {
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

// -----------------------------------------------------------------------------
// Full ReSTIR DI Core (Initial Candidates + Temporal + Spatial Resampling)
// -----------------------------------------------------------------------------
void evalReSTIR_DI(
    ivec2 pixelCoord,
    uint pixelIndex,
    vec3 hitPoint,
    vec3 hitNormal,
    float hitDepth,
    vec3 V,
    vec3 diffuseColor,
    vec3 F0,
    float alphaRoughness,
    bool enableSpecular,
    inout uint seed,
    out ReservoirDI outR
) {
    ReservoirDI R;
    R.lightIdx = 0u;
    R.uvX = 0.0;
    R.uvY = 0.0;
    R.wSum = 0.0;
    R.M = 0.0;
    R.W = 0.0;
    R.targetPdf = 0.0;
    R.pad = 0u;

    // Fast-path: When scene has <= 1 light, selection probability is trivially 1.0.
    // Avoid redundant candidate loops and spatial gathers.
    if (pc.numLights <= 1u) {
        R.lightIdx = 0u;
        R.uvX = randFloat(seed);
        R.uvY = randFloat(seed);
        R.M = 1.0;
        float dummyPdf = 1.0;
        R.targetPdf = evalLightCandidate(
            0u, vec2(R.uvX, R.uvY), hitPoint, hitNormal, V, diffuseColor, F0, alphaRoughness, enableSpecular, dummyPdf
        );
        R.wSum = (dummyPdf > 0.0) ? (R.targetPdf / dummyPdf) : 0.0;
        R.W = (R.targetPdf > 0.0 && dummyPdf > 0.0) ? (1.0 / dummyPdf) : 0.0;
        R.pad = packGeom(hitNormal, hitDepth);
        currentReservoirs[pixelIndex] = R;
        outR = R;
        return;
    }

    // 1. Initial Candidate Generation (M_init = 4 candidates with Chao's WRS)
    const uint M_init = 4u;
    for (uint c = 0u; c < M_init; ++c) {
        uint candIdx = uint(randFloat(seed) * float(pc.numLights)) % pc.numLights;
        vec2 cUv = randVec2(seed);
        float cLightPdf = 0.0;
        float p_hat = evalLightCandidate(
            candIdx, cUv, hitPoint, hitNormal, V, diffuseColor, F0, alphaRoughness, enableSpecular, cLightPdf
        );

        float w_i = (cLightPdf > 0.0) ? (p_hat / cLightPdf) : 0.0;
        R.wSum += w_i;
        R.M += 1.0;
        if (randFloat(seed) * R.wSum < w_i) {
            R.lightIdx = candIdx;
            R.uvX = cUv.x;
            R.uvY = cUv.y;
            R.targetPdf = p_hat;
        }
    }

    // 2. Temporal Resampling (Reproject Surface Hit to Frame t-1)
    vec4 prevClip = ubo.prevViewProj * vec4(hitPoint, 1.0);
    if (prevClip.w > 0.0) {
        vec2 prevNDC = prevClip.xy / prevClip.w;
        vec2 prevUV = prevNDC * 0.5 + 0.5;
        if (prevUV.x >= 0.0 && prevUV.x <= 1.0 && prevUV.y >= 0.0 && prevUV.y <= 1.0) {
            int prevPx = clamp(int(prevUV.x * float(pc.width)), 0, int(pc.width) - 1);
            int prevPy = clamp(int(prevUV.y * float(pc.height)), 0, int(pc.height) - 1);
            uint prevIdx = uint(prevPy * int(pc.width) + prevPx);

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
                    float historyM = min(R_prev.M, 24.0);
                    float dummyPdf;
                    float p_hat_current = evalLightCandidate(
                        R_prev.lightIdx, vec2(R_prev.uvX, R_prev.uvY), hitPoint, hitNormal, V,
                        diffuseColor, F0, alphaRoughness, enableSpecular, dummyPdf
                    );

                    float w_temporal = p_hat_current * R_prev.W * historyM;
                    R.wSum += w_temporal;
                    R.M += historyM;
                    if (randFloat(seed) * R.wSum < w_temporal) {
                        R.lightIdx = R_prev.lightIdx;
                        R.uvX = R_prev.uvX;
                        R.uvY = R_prev.uvY;
                        R.targetPdf = p_hat_current;
                    }
                }
            }
        }
    }

    // 3. Spatial Resampling (Edge-Stopping Bilateral Filtering)
    bool enableSpatial = (ubo.flags & (1u << 7)) != 0u;
    if (enableSpatial) {
        uint spatialSamples = (ubo.flags >> 8u) & 0xFu;
        if (spatialSamples == 0u) spatialSamples = 3u;
        spatialSamples = min(spatialSamples, 4u); // Cap at 4 for strict < 2.5ms budget
        float spatialRadius = float((ubo.flags >> 12u) & 0xFFu);
        if (spatialRadius < 1.0) spatialRadius = 8.0;

        for (uint i = 0u; i < spatialSamples; ++i) {
            float angle = randFloat(seed) * TWO_PI;
            float rad = sqrt(randFloat(seed)) * spatialRadius;
            ivec2 offset = ivec2(round(vec2(cos(angle), sin(angle)) * rad));
            if (offset == ivec2(0)) {
                offset = (randFloat(seed) > 0.5) ? ivec2(1, 0) : ivec2(0, 1);
            }

            ivec2 nbrCoord = clamp(pixelCoord + offset, ivec2(0), ivec2(int(pc.width) - 1, int(pc.height) - 1));
            uint nbrIdx = uint(nbrCoord.y * int(pc.width) + nbrCoord.x);

            ReservoirDI R_nbr = historyReservoirs[nbrIdx];
            if (R_nbr.pad != 0u && R_nbr.M > 0.0 && R_nbr.W > 0.0 && R_nbr.lightIdx < pc.numLights) {
                vec3 nbrNormal;
                float nbrDepth;
                unpackGeom(R_nbr.pad, nbrNormal, nbrDepth);

                float normDot = dot(hitNormal, nbrNormal);
                float depthDiff = abs(hitDepth - nbrDepth) / max(hitDepth, 1e-3);

                if (normDot > 0.90 && depthDiff < 0.10) {
                    float nbrM = min(R_nbr.M, 24.0);
                    float dummyPdf;
                    float p_hat_nbr = evalLightCandidate(
                        R_nbr.lightIdx, vec2(R_nbr.uvX, R_nbr.uvY), hitPoint, hitNormal, V,
                        diffuseColor, F0, alphaRoughness, enableSpecular, dummyPdf
                    );

                    float w_spatial = p_hat_nbr * R_nbr.W * nbrM;
                    R.wSum += w_spatial;
                    R.M += nbrM;
                    if (randFloat(seed) * R.wSum < w_spatial) {
                        R.lightIdx = R_nbr.lightIdx;
                        R.uvX = R_nbr.uvX;
                        R.uvY = R_nbr.uvY;
                        R.targetPdf = p_hat_nbr;
                    }
                }
            }
        }
    }

    // 4. Compute Final Unbiased Weight W with M-capping
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
    currentReservoirs[pixelIndex] = R;
    outR = R;
}

#endif // WAVEFRONT_RESTIR_GLSL
