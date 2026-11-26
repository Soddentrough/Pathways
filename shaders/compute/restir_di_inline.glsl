#ifndef RESTIR_DI_INLINE_GLSL
#define RESTIR_DI_INLINE_GLSL

// =============================================================================
// Inline Bounce-0 ReSTIR DI Spatiotemporal Reuse + Single-Pass Inline Shadow
// =============================================================================
// Requires before inclusion:
//   - restir_common.glsl (with RESTIR_INCLUDE_SAMPLER_HELPERS)
//   - bsdf_standard.glsl, mis_common.glsl, wavefront_shadow_inline.glsl
//   - ubo, pc, lights[], uMotionVectorImage, uPrevNormalDepthImage,
//     uMlSpecAlbedoImage, currentReservoirs[], historyReservoirs[]
// =============================================================================

void restirDiRunInlineReuse(
    uint pixelIndex,
    bool enableShadows,
    bool manyLightCheckVis,
    float metallic,
    float roughness,
    float alphaRoughness,
    float diffTrans,
    vec3 hitPoint,
    vec3 hitNormal,
    float hitT,
    vec3 V,
    f16vec3 diffuseColor,
    f16vec3 F0,
    inout DiReservoir r,
    inout DiSample sel,
    inout uint seed,
    out uint outEffectiveM,
    out bool outHasCamMotion,
    out bool outWinnerVisVerified
) {
    outHasCamMotion = false;
    outWinnerVisVerified = false;

    if (manyLightCheckVis && enableShadows && sel.has && sel.target > 1e-6) {
        vec3 visOrig = sel.isTrans ? offsetRayOrigin(hitPoint, -hitNormal) : offsetRayOrigin(hitPoint, hitNormal);
        bool freshBlocked = traceShadowRayInline(visOrig, sel.dir, sel.dist,
                                                 (ubo.flags & (1u << 5)) != 0u, pc.numOpaqueTriangles);
        outWinnerVisVerified = true;
        if (freshBlocked) {
            uint baseM = restirGetM(r.idxM);
            r.wSum = 0.0;
            r.targetCosBits = 0u;
            r.idxM = restirPackIdxM(0u, baseM);
            r.flags = restirPackFlags(0u, false, false, hitNormal);
            sel.has = false;
            sel.target = 0.0;
        }
    }

    bool isV4DiCapture = (pc.captureMlData & ML_CAPTURE_UPWAYS_V4_BIT) != 0u;
    bool isMlDenoiserDi = (pc.captureMlData & ML_CAPTURE_ML_DENOISER_MASK) != 0u;
    ivec2 px = ivec2(int(pixelIndex % pc.width), int(pixelIndex / pc.width));
    bool wroteHistory = false;

    if (ubo.frameIndex > 0u && (ubo.flags & (1u << 23)) == 0u) {
        vec4 mvFull = imageLoad(uMotionVectorImage, px);
        vec2 mvPx = mvFull.xy * vec2(float(pc.width), float(pc.height));
        bool hasCamMotion = dot(mvPx, mvPx) > 0.25;
        outHasCamMotion = hasCamMotion;

        float effEnvIntensity = (pc.hasEnvMap == 1u) ? pc.envMapIntensity : 0.0;
        uint effNumLights = restirEffectiveLightCount(pc.numLights);
        bool allowMultiLightStaticDi = (effNumLights > 4u);
        bool manyLightScene = (pc.numLights > 4096u);
        bool allowTemporalDiHistory = isV4DiCapture || hasCamMotion || allowMultiLightStaticDi;
        bool allowSpatialDi = (effNumLights > 1u || hasCamMotion) &&
                              !(metallic > 0.5 && roughness < 0.25) &&
                              !(isMlDenoiserDi && !hasCamMotion);

        if (!allowTemporalDiHistory) {
            DiReservoir hOut = r;
            hOut.flags &= DI_FLAG_HISTORY_MASK;
            currentReservoirs[pixelIndex] = hOut;
            wroteHistory = true;
        }

        if (allowTemporalDiHistory || allowSpatialDi) {
            float currViewZ = max(dot(hitPoint - ubo.position.xyz, -ubo.viewInverse[2].xyz), 1e-4);
            float expectedPrevDepth = max(currViewZ + mvFull.z, 1e-4);
            float expectedPrevHitT = hitT * (expectedPrevDepth / currViewZ);
            ivec2 prevCoord = ivec2(round(vec2(px) - mvPx));

            uint restirSeed = seed ^ 0x9E3779B9u;
            pcg_hash(restirSeed);

            uint tapHash = (pixelIndex * 0x9E3779B9u) ^ (pc.frameIndex * 0x85EBCA6Bu);
            if (isV4DiCapture) {
                tapHash ^= 0xA3C59AC3u;
            }
            pcg_hash(tapHash);
            uint rot = (tapHash >> 30u) & 3u;
            uint numSpatialTaps = manyLightScene ? 1u : 2u;
            uint tapOff = (tapHash >> 24u) % 24u;
            uint tempMCap = min(pc.restirMCap, hasCamMotion ? 6u : (isV4DiCapture ? 20u : (manyLightScene ? ((pc.captureMlData == 0u) ? 8u : 4u) : 4u)));
            uint spatMCap = isV4DiCapture
                ? min(pc.restirMCap, hasCamMotion ? 6u : 12u)
                : (hasCamMotion ? min(pc.restirMCap, 6u) : min(pc.restirMCap, manyLightScene ? ((pc.captureMlData == 0u) ? 12u : 8u) : 16u));
            float spatDiDepthGate = isV4DiCapture ? 0.06 : (manyLightScene ? 0.04 : RESTIR_TAP_DEPTH_REL_GATE);

            int startTap = allowTemporalDiHistory ? -1 : 0;
            int endTap = allowSpatialDi ? int(numSpatialTaps) : 0;
            [[dont_unroll]]
            for (int tapPass = startTap; tapPass < endTap; ++tapPass) {
                bool isTemp = (tapPass < 0);
                ivec2 tapCoord = prevCoord;
                if (!isTemp) {
                    uint tap = uint(tapPass);
                    uint tapIdx = isV4DiCapture
                        ? (((tapHash >> 28u) & 2u) + tap)
                        : (hasCamMotion
                            ? (8u + ((tapOff + tap * 7u) % 24u))
                            : (manyLightScene
                                ? (((tapHash >> 27u) + tap) & 7u)
                                : (((tapHash >> 24u) + tap * 7u) & 31u)));
                    ivec2 base = RESTIR_TAP_OFFSETS[tapIdx];
                    if ((rot & 1u) != 0u) base = ivec2(-base.y, base.x);
                    if ((rot & 2u) != 0u) base = -base;
                    tapCoord += base;
                }
                if (tapCoord.x >= 0 && tapCoord.y >= 0 &&
                    tapCoord.x < int(pc.width) && tapCoord.y < int(pc.height)) {
                    float prevTapDepth = imageLoad(uPrevNormalDepthImage, tapCoord).w;
                    float relDepthDiff = abs(expectedPrevDepth - prevTapDepth) / max(max(expectedPrevDepth, prevTapDepth), 1e-4);
                    float depthGate = isTemp ? 0.06 : spatDiDepthGate;
                    if (relDepthDiff <= depthGate) {
                        DiReservoir src = historyReservoirs[uint(tapCoord.y) * pc.width + uint(tapCoord.x)];
                        uint ageNew = isTemp ? min(restirGetAge(src.flags) + 1u, RESTIR_MAX_AGE) : restirGetAge(src.flags);
                        uint tapMCap = isTemp ? tempMCap : spatMCap;
                        bool clampW = isTemp ? (hasCamMotion || manyLightScene) : true;
                        if (restirDiApplyTap(r, sel, src, pc.numLights, effEnvIntensity, hitPoint, hitNormal, hitT, expectedPrevHitT,
                                             V, diffuseColor, F0, alphaRoughness, diffTrans,
                                             tapMCap, ageNew, false, clampW, restirSeed)) {
                            outWinnerVisVerified = false;
                        }
                    }
                }
                if (isTemp) {
                    DiReservoir hOut = r;
                    hOut.flags &= DI_FLAG_HISTORY_MASK;
                    currentReservoirs[pixelIndex] = hOut;
                    wroteHistory = true;
                }
            }
        }
    }

    if (!wroteHistory) {
        DiReservoir hOut = r;
        hOut.flags &= DI_FLAG_HISTORY_MASK;
        currentReservoirs[pixelIndex] = hOut;
    }

    if (isV4DiCapture && pc.sppSampleIndex == 0u) {
        float restirMNorm = float(min(restirGetM(r.idxM), 64u)) / 64.0;
        vec4 prevSa = imageLoad(uMlSpecAlbedoImage, px);
        imageStore(uMlSpecAlbedoImage, px, vec4(prevSa.rgb, restirMNorm));
    }

    outEffectiveM = max(restirGetM(r.idxM), 1u);
}

#endif // RESTIR_DI_INLINE_GLSL
