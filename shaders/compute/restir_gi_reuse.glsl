#ifndef RESTIR_GI_REUSE_GLSL
#define RESTIR_GI_REUSE_GLSL

// =============================================================================
// ReSTIR PT Indirect Spatiotemporal Reuse & Decoupled Visibility Replay
// =============================================================================
// Requires before inclusion:
//   - restir_common.glsl, bsdf_standard.glsl
//   - CameraUBO `ubo`
//   - uAccumImage, uMotionVectorImage, uPrevNormalDepthImage,
//     uMlDiffuseImage, uMlSpecularImage
//   - giReservoirs[], giHistoryReservoirs[]
// =============================================================================

StdBsdf giReconstructBsdf(X1Context ctx, uint captureMlData) {
    vec3 n1 = x1ContextNormal(ctx);
    vec3 wo = x1ContextWo(ctx);
    vec3 baseColor = x1ContextBaseColor(ctx);
    float metallic = x1ContextMetallic(ctx);
    float f0Scalar = x1ContextF0(ctx);
    float roughness = x1ContextRoughness(ctx);
    vec3 diffuseColor = baseColor * (1.0 - metallic);
    vec3 F0 = mix(vec3(f0Scalar), baseColor, metallic);
    StdBsdf b = makeStdBsdf(diffuseColor, F0, roughness * roughness, 0.0, metallic, dot(n1, wo), STD_BSDF_FULL);
    return b;
}

StdBsdf giReconstructTargetBsdf(X1Context ctx, uint captureMlData) {
    vec3 n1 = x1ContextNormal(ctx);
    vec3 wo = x1ContextWo(ctx);
    vec3 baseColor = x1ContextBaseColor(ctx);
    float metallic = x1ContextMetallic(ctx);
    float f0Scalar = x1ContextF0(ctx);
    float origRoughness = x1ContextRoughness(ctx);
    float targetRoughness = max(origRoughness, 0.35);
    vec3 diffuseColor = baseColor * (1.0 - metallic);
    vec3 F0 = mix(vec3(f0Scalar), baseColor, metallic);
    StdBsdf b = makeStdBsdf(diffuseColor, F0, targetRoughness * targetRoughness, 0.0, metallic, dot(n1, wo), STD_BSDF_FULL);
    return b;
}

float giTargetAtBsdf(StdBsdf giBsdf, vec3 n1, vec3 wo, vec3 x1, vec3 x2, vec3 radiance) {
    vec3 dv = x2 - x1;
    float dist = length(dv);
    if (dist < GI_TAP_MIN_DIST) return 0.0;
    vec3 wi = dv / dist;
    if (dot(n1, wi) <= GI_TAP_MIN_COS) return 0.0;
    vec3 L2 = max(radiance, vec3(0.0));
    if (dot(L2, vec3(0.2126, 0.7152, 0.0722)) <= 1e-6) return 0.0;
    float unusedPdf = 0.0;
    vec3 fCos = stdBsdfEval(giBsdf, n1, wo, wi, unusedPdf);
    return max(dot(fCos * L2, vec3(0.2126, 0.7152, 0.0722)), 0.0);
}

struct GiTapEval {
    uint mTap;
    float visWFactor;
    vec3 dirToX2;
    vec3 radiance;
};

bool giApplyTapEx(
    inout GiReservoir r,
    in GiReservoir tap,
    StdBsdf giBsdf,
    vec3 x1,
    vec3 n1,
    vec3 wo,
    float roughness,
    uint mCap,
    uint ageNew,
    bool verifyVisInline,
    uint numOpaqueTriangles,
    float wSrcMax,
    float wFactorMax,
    inout uint seed,
    out GiTapEval outEval
) {
    outEval.mTap = 0u;
    outEval.visWFactor = 0.0;
    outEval.dirToX2 = vec3(0.0, 1.0, 0.0);
    outEval.radiance = vec3(0.0);

    if (!giIsValid(tap) || giGetM(tap) == 0u) return false;
    if (giGetAge(tap) >= GI_MAX_AGE) return false;
    if (dot(n1, giGetX1Normal(tap)) < RESTIR_TAP_NORMAL_MIN_DOT) return false;

    uint mSrc = giGetM(tap);
    uint mTap = min(mSrc, mCap);
    if (mTap == 0u) return false;
    outEval.mTap = mTap;

    float pTap = giGetTarget(tap);
    float tapWSum = giGetWSum(tap);
    float weight = 0.0;
    float pHat = 0.0;
    vec3 x2c = vec3(0.0);
    vec3 n2c = vec3(0.0, 1.0, 0.0);
    vec3 radC = vec3(0.0);
    float dNew = 0.0;
    float cosRcNew = 0.0;
    vec3 dirNew = vec3(0.0, 1.0, 0.0);
    float wFactor = 0.0;

    if (pTap > 0.0 && tapWSum > 0.0 && giHasSurface(tap)) {
        x2c = giGetX2(tap);
        n2c = giGetN2(tap);
        radC = giGetRadiance(tap);
        vec3 toNew = x2c - x1;
        dNew = length(toNew);
        if (dNew >= GI_TAP_MIN_DIST) {
            dirNew = toNew / dNew;
            float cos1 = dot(n1, dirNew);
            float rawCosRc = dot(n2c, -dirNew);
            // Signed front-face Jacobian gate: strictly reject thin-wall back-side leaks
            if (cos1 > GI_TAP_MIN_COS && rawCosRc > 1e-4) {
                float dSrc = max(giGetDSrc(tap), 1e-3);
                float distRatio = dSrc / max(dNew, 1e-3);
                bool validFootprint = (distRatio >= 0.2 && distRatio <= 5.0) && (roughness >= 0.20);
                if (validFootprint) {
                    pHat = giTargetAtBsdf(giBsdf, n1, wo, x1, x2c, radC);
                    if (pHat > 0.0) {
                        cosRcNew = clamp(rawCosRc, 0.0, 1.0);
                        float cosRcSrc = giGetCosSrc(tap);
                        float jac = evalGIJacobian(dSrc, dNew, cosRcSrc, cosRcNew);
                        float W_src = tapWSum / max(float(mSrc) * pTap, 1e-6);
                        if (wSrcMax > 0.0) {
                            W_src = min(W_src, wSrcMax);
                        }
                        wFactor = W_src * jac;
                        if (wFactorMax > 0.0) {
                            wFactor = min(wFactor, wFactorMax);
                        }
                        weight = wFactor * float(mTap) * pHat;
                        if (isnan(weight) || isinf(weight) || !(weight > 0.0)) {
                            weight = 0.0;
                        }
#if ENABLE_INLINE_SHADOWS
                        if (verifyVisInline && weight > 0.0) {
                            vec3 visOrig = offsetRayOrigin(x1, n1);
                            vec3 visTarget = offsetRayOrigin(x2c, n2c);
                            vec3 visTo = visTarget - visOrig;
                            float visDist = length(visTo);
                            if (visDist <= 2.0 * EPSILON ||
                                traceShadowRayInline(visOrig, visTo / visDist, visDist, (ubo.flags & (1u << 5)) != 0u, numOpaqueTriangles)) {
                                weight = 0.0;
                            }
                        }
#endif
                        if (weight > 0.0) {
                            outEval.visWFactor = wFactor;
                            outEval.dirToX2 = dirNew;
                            outEval.radiance = radC;
                        }
                    }
                }
            }
        }
    }

    float wSelf = (giIsValid(r) && giHasSurface(r) && giGetTarget(r) > 0.0) ? giGetWSum(r) : 0.0;
    uint mSelf = giIsValid(r) ? giGetM(r) : 0u;
    uint combinedM = mSelf + mTap;
    float combined = wSelf + weight;

    if (weight > 0.0 && (randFloat(seed) * combined < weight)) {
        r = makeGiReservoir(true, x2c, n2c, n1, combined, combinedM, pHat, radC, dNew, cosRcNew, ageNew);
        return true;
    }
    if (giIsValid(r) && giHasSurface(r) && giGetTarget(r) > 0.0) {
        r = makeGiReservoir(true, giGetX2(r), giGetN2(r), n1, combined, combinedM, giGetTarget(r),
                            giGetRadiance(r), giGetDSrc(r), giGetCosSrc(r), giGetAge(r));
    } else {
        r = makeGiReservoir(false, vec3(0.0), vec3(0.0, 1.0, 0.0), n1, 0.0, combinedM, 0.0,
                            vec3(0.0), 0.0, 0.0, giIsValid(r) ? giGetAge(r) : ageNew);
    }
    return false;
}

bool giApplyTap(
    inout GiReservoir r,
    in GiReservoir tap,
    StdBsdf giBsdf,
    vec3 x1,
    vec3 n1,
    vec3 wo,
    float roughness,
    uint mCap,
    uint ageNew,
    bool verifyVisInline,
    uint numOpaqueTriangles,
    float wSrcMax,
    float wFactorMax,
    inout uint seed
) {
    GiTapEval unusedEval;
    return giApplyTapEx(
        r, tap, giBsdf, x1, n1, wo, roughness, mCap, ageNew, verifyVisInline,
        numOpaqueTriangles, wSrcMax, wFactorMax, seed, unusedEval
    );
}

void giAccumulateRbSample(
    StdBsdf giBsdf,
    vec3 giN1,
    vec3 giWo,
    bool mergeMatteSpec,
    float sampleClamp,
    uint mWeight,
    float wFactor,
    vec3 wi,
    vec3 L2,
    inout vec3 rbDiffSum,
    inout vec3 rbSpecSum
) {
    if (mWeight == 0u || wFactor <= 0.0) return;
    float unusedPdf = 0.0;
    vec3 specFCos = vec3(0.0);
    vec3 fCos = stdBsdfEvalSplit(giBsdf, giN1, giWo, wi, unusedPdf, specFCos);
    vec3 scale = L2 * wFactor;
    vec3 diff = mergeMatteSpec ? (fCos * scale) : (max(fCos - specFCos, vec3(0.0)) * scale);
    vec3 spec = mergeMatteSpec ? vec3(0.0) : (specFCos * scale);
    if (sampleClamp > 0.0) {
        vec3 tot = diff + spec;
        float rawLum = dot(tot, vec3(0.2126, 0.7152, 0.0722));
        if (rawLum > sampleClamp) {
            float s = sampleClamp / rawLum;
            diff *= s;
            spec *= s;
        }
    }
    float mF = float(mWeight);
    rbDiffSum += diff * mF;
    rbSpecSum += spec * mF;
}

void restirGiRunReuse(
    uint pixelIndex,
    uint width,
    uint height,
    uint frameIndex,
    uint sppSampleIndex,
    uint restirMCap,
    uint captureMlData,
    float indirectClamp,
    bool useInlineVis,
    uint numOpaqueTriangles,
    X1Context giCtx,
    vec3 giX1,
    bool hasSelfCandidate,
    vec3 selfX2,
    vec3 selfN2,
    float selfHitT,
    float cosRcSelf,
    vec3 giL2,
    inout uint seed,
    out bool outHasGiShadowRay,
    out PackedShadowRay outGiShadowRay
) {
    outHasGiShadowRay = false;
#if ENABLE_INLINE_SHADOWS
    bool inlineVisActive = useInlineVis && ((ubo.flags & (1u << 4)) != 0u);
#else
    bool inlineVisActive = false;
#endif

    vec3 giN1 = x1ContextNormal(giCtx);
    vec3 giWo = x1ContextWo(giCtx);
    float giMetallic = x1ContextMetallic(giCtx);
    float giRoughness = x1ContextRoughness(giCtx);
    float giQSelf = x1ContextQSelf(giCtx);
    StdBsdf giBsdf = giReconstructBsdf(giCtx, captureMlData);
    StdBsdf giTargetBsdf = giReconstructTargetBsdf(giCtx, captureMlData);

    GiReservoir giR = makeGiReservoir(false, vec3(0.0), vec3(0.0, 1.0, 0.0), giN1, 0.0, 1u, 0.0, vec3(0.0), 0.0, 0.0, 0u);
    bool giWinnerSelf = false;
    if (hasSelfCandidate) {
        float giPSelf = giTargetAtBsdf(giTargetBsdf, giN1, giWo, giX1, selfX2, giL2);
        if (giPSelf > 0.0 && giQSelf > 1e-8) {
            float giWSelf = giPSelf / giQSelf;
            giR = makeGiReservoir(true, selfX2, selfN2, giN1, giWSelf, 1u, giPSelf, giL2, selfHitT, cosRcSelf, 0u);
            giWinnerSelf = true;
        }
    }

    ivec2 giPix = ivec2(int(pixelIndex % width), int(pixelIndex / width));
    vec4 mvData = vec4(0.0);
    ivec2 giPrev = giPix;
    float expectedPrevDepth = 1.0;
    bool hasGiMotion = false;
    bool canReuseHistory = (ubo.frameIndex > 0u) && ((ubo.flags & (1u << 23)) == 0u);
    bool isV4Gi = (captureMlData & ML_CAPTURE_UPWAYS_V4_BIT) != 0u;
    bool isMlGi = (captureMlData & ML_CAPTURE_ML_DENOISER_MASK) != 0u;
    bool useRbGi = (!isMlGi) && inlineVisActive;
    bool mergeMatteSpecRb = (!isMlGi) && (giRoughness >= 0.70 && giMetallic < 0.05);
    vec3 rbDiffSum = vec3(0.0);
    vec3 rbSpecSum = vec3(0.0);
    uint rbMTotal = 1u;

    GiReservoir giTempR = giR;
    bool giTempWinnerSelf = giWinnerSelf;
    GiTapEval tempEval;
    tempEval.mTap = 0u;
    tempEval.visWFactor = 0.0;
    tempEval.dirToX2 = vec3(0.0, 1.0, 0.0);
    tempEval.radiance = vec3(0.0);

    if (canReuseHistory) {
        mvData = imageLoad(uMotionVectorImage, giPix);
        vec2 giMvPx = mvData.xy * vec2(float(width), float(height));
        hasGiMotion = dot(giMvPx, giMvPx) > 0.25;
        float selfDepth = max(dot(giX1 - ubo.position.xyz, -ubo.viewInverse[2].xyz), 0.0);
        expectedPrevDepth = max(selfDepth + mvData.z, 1e-4);
        giPrev = ivec2(round(vec2(giPix) - giMvPx));

        bool allowTemporalGiHistory = isMlGi || hasGiMotion;
        if (allowTemporalGiHistory && abs(mvData.x) < 1.5 && abs(mvData.y) < 1.5 &&
            giPrev.x >= 0 && giPrev.y >= 0 && giPrev.x < int(width) && giPrev.y < int(height)) {
            float prevTapDepth = imageLoad(uPrevNormalDepthImage, giPrev).w;
            float relDepthDiff = abs(expectedPrevDepth - prevTapDepth) / max(max(expectedPrevDepth, prevTapDepth), 1e-4);
            float tempDepthGate = hasGiMotion ? 0.03 : 0.05;
            if (relDepthDiff <= tempDepthGate) {
                GiReservoir giTap = giHistoryReservoirs[uint(giPrev.y) * width + uint(giPrev.x)];
                uint tempMCap = min(restirMCap, hasGiMotion ? 4u : (isMlGi ? 8u : 3u));
                uint giAgeNew = min(giGetAge(giTap) + 1u, GI_MAX_AGE);
                bool inlineVisTemp = inlineVisActive;
                if (giApplyTapEx(giTempR, giTap, giTargetBsdf, giX1, giN1, giWo, giRoughness,
                                 tempMCap, giAgeNew, inlineVisTemp, numOpaqueTriangles,
                                 hasGiMotion ? 22.0 : 0.0, hasGiMotion ? 28.0 : 0.0, seed, tempEval)) {
                    giTempWinnerSelf = false;
                }
            }
        }
    }

    // Save pure TEMPORAL reservoir to history (never feed spatial reuse back into temporal history)
    if (sppSampleIndex == 0u) {
        giReservoirs[pixelIndex] = giIsValid(giTempR) ? giTempR : invalidGiReservoir();
    }

    if (useRbGi && giWinnerSelf) {
        vec3 selfTo = selfX2 - giX1;
        float selfDist = length(selfTo);
        if (selfDist > 2.0 * EPSILON) {
            float wSelf = 1.0 / giQSelf;
            if (indirectClamp > 0.0) {
                wSelf = min(wSelf, hasGiMotion ? 24.0 : GI_WEIGHT_CLAMP_MAX);
            }
            giAccumulateRbSample(
                giBsdf, giN1, giWo, mergeMatteSpecRb, indirectClamp, 1u, wSelf,
                selfTo / selfDist, giL2, rbDiffSum, rbSpecSum
            );
        }
    }
    if (useRbGi && hasGiMotion && tempEval.mTap > 0u) {
        rbMTotal += tempEval.mTap;
        float effTempClamp = (indirectClamp > 0.0) ? min(indirectClamp, 10.0) : 0.0;
        giAccumulateRbSample(
            giBsdf, giN1, giWo, mergeMatteSpecRb, effTempClamp, tempEval.mTap,
            min(tempEval.visWFactor, 24.0), tempEval.dirToX2, tempEval.radiance, rbDiffSum, rbSpecSum
        );
    }

    bool useTempForGiShade = isMlGi || hasGiMotion;
    GiReservoir giShadeR = useTempForGiShade ? giTempR : giR;
    bool giShadeWinnerSelf = useTempForGiShade ? giTempWinnerSelf : giWinnerSelf;

    if (canReuseHistory) {
        uint tapHash = (pixelIndex * 0x9E3779B9u) ^ (frameIndex * 0x85EBCA6Bu);
        if (isMlGi) {
            tapHash ^= 0xA3C59AC3u;
        }
        pcg_hash(tapHash);
        uint rot = (tapHash >> 30u) & 3u;
        uint numSpatialTaps = isMlGi
            ? 1u
            : (hasGiMotion ? 1u : ((numOpaqueTriangles > 2000000u) ? 1u : 8u));
        uint tapOff = (tapHash >> 24u) % 24u;
        uint spatialMCap = isMlGi
            ? min(restirMCap, hasGiMotion ? 4u : 8u)
            : (hasGiMotion ? min(restirMCap, 4u) : 1u);
        float spatDepthGate = isMlGi ? 0.03 : RESTIR_TAP_DEPTH_REL_GATE;
        float spatNormalGate = isMlGi ? 0.95 : RESTIR_TAP_NORMAL_MIN_DOT;

        for (uint tap = 0u; tap < numSpatialTaps; ++tap) {
            uint tapIdx = isMlGi
                ? (((tapHash >> 27u) + tap) & 7u)
                : (hasGiMotion
                    ? (8u + ((tapOff + tap * 7u) % 24u))
                    : (((tapHash >> 24u) + tap * 3u) & 31u));
            ivec2 base = RESTIR_TAP_OFFSETS[tapIdx];
            if ((rot & 1u) != 0u) base = ivec2(-base.y, base.x);
            if ((rot & 2u) != 0u) base = -base;
            ivec2 offset = base;
            ivec2 giCoord = giPrev + offset;
            if (giCoord.x < 0 || giCoord.y < 0 || giCoord.x >= int(width) || giCoord.y >= int(height)) {
                continue;
            }
            float prevTapDepth = imageLoad(uPrevNormalDepthImage, giCoord).w;
            float relDepthDiff = abs(expectedPrevDepth - prevTapDepth) / max(max(expectedPrevDepth, prevTapDepth), 1e-4);
            if (relDepthDiff > spatDepthGate) continue;

            GiReservoir giTap = giHistoryReservoirs[uint(giCoord.y) * width + uint(giCoord.x)];
            if (dot(giN1, giGetX1Normal(giTap)) < spatNormalGate) continue;
            GiTapEval spatEval;
            if (giApplyTapEx(giShadeR, giTap, giTargetBsdf, giX1, giN1, giWo, giRoughness,
                             spatialMCap, giGetAge(giTap), inlineVisActive, numOpaqueTriangles,
                             hasGiMotion ? 22.0 : 28.0, hasGiMotion ? 28.0 : 32.0, seed, spatEval)) {
                giShadeWinnerSelf = false;
            }
            if (useRbGi && spatEval.mTap > 0u) {
                rbMTotal += spatEval.mTap;
                float effSpatClamp = (hasGiMotion && indirectClamp > 0.0) ? min(indirectClamp, 10.0) : indirectClamp;
                float maxW = (indirectClamp > 0.0) ? (hasGiMotion ? 24.0 : GI_WEIGHT_CLAMP_MAX) : 1e30;
                giAccumulateRbSample(
                    giBsdf, giN1, giWo, mergeMatteSpecRb, effSpatClamp, spatEval.mTap,
                    min(spatEval.visWFactor, maxW), spatEval.dirToX2, spatEval.radiance, rbDiffSum, rbSpecSum
                );
            }
        }
    }

    if (useRbGi) {
        float invDenom = ((ubo.spp > 0u) ? (1.0 / float(ubo.spp)) : 1.0) / float(max(rbMTotal, 1u));
        vec3 diffContrib = rbDiffSum * invDenom;
        vec3 specContrib = rbSpecSum * invDenom;
        vec3 rawTotal = diffContrib + specContrib;
        float effGiClamp = (hasGiMotion && rbMTotal > 1u && indirectClamp > 0.0)
            ? min(indirectClamp, 10.0)
            : indirectClamp;
        vec3 totalContrib = (effGiClamp > 0.0) ? clampIndirectRadiance(rawTotal, effGiClamp) : rawTotal;
        float rawLum = dot(rawTotal, vec3(0.2126, 0.7152, 0.0722));
        float clampedScale = (rawLum > 1e-8) ? (dot(totalContrib, vec3(0.2126, 0.7152, 0.0722)) / rawLum) : 1.0;
        diffContrib *= clampedScale;
        specContrib *= clampedScale;
        if (dot(totalContrib, totalContrib) > 1e-12) {
            if (captureMlData == 0u || (captureMlData & 2u) != 0u) {
                vec4 prevAcc = imageLoad(uAccumImage, giPix);
                imageStore(uAccumImage, giPix, vec4(addFp16Stochastic(prevAcc.rgb, totalContrib, seed), prevAcc.a));
            }
            wfAddSplitIndirect(giPix, diffContrib, specContrib, captureMlData, seed);
            if (captureMlData != 0u) {
                if (dot(diffContrib, diffContrib) > 1e-12) {
                    vec4 prevDiff = imageLoad(uMlDiffuseImage, giPix);
                    imageStore(uMlDiffuseImage, giPix, vec4(addFp16Stochastic(prevDiff.rgb, diffContrib, seed), prevDiff.a));
                }
                if (dot(specContrib, specContrib) > 1e-12) {
                    vec4 prevSpec = imageLoad(uMlSpecularImage, giPix);
                    imageStore(uMlSpecularImage, giPix, vec4(addFp16Stochastic(prevSpec.rgb, specContrib, seed), prevSpec.a));
                }
            }
        }
    } else if (giIsValid(giShadeR) && giHasSurface(giShadeR) && giGetM(giShadeR) > 0u && giGetTarget(giShadeR) > 0.0) {
        {
            vec3 giX2w = giShadeWinnerSelf ? selfX2 : giGetX2(giShadeR);
            vec3 giToW = giX2w - giX1;
            float giDistW = length(giToW);
            if (giDistW > 2.0 * EPSILON) {
                vec3 giDirW = giToW / giDistW;
                float unusedPdfW = 0.0;
                vec3 specFCosW = vec3(0.0);
                vec3 fCosW = stdBsdfEvalSplit(giBsdf, giN1, giWo, giDirW, unusedPdfW, specFCosW);
                vec3 giL2w = giGetRadiance(giShadeR);
                float giTargetW = giGetTarget(giShadeR);
                float giW = giGetWSum(giShadeR) / max(float(giGetM(giShadeR)) * giTargetW, 1e-8);
                if (indirectClamp > 0.0) {
                    giW = min(giW, hasGiMotion ? 24.0 : GI_WEIGHT_CLAMP_MAX);
                }
                float invSpp = (ubo.spp > 0u) ? (1.0 / float(ubo.spp)) : 1.0;
                vec3 scaleW = giL2w * (giW * invSpp);

                bool isV4 = (captureMlData & ML_CAPTURE_UPWAYS_V4_BIT) != 0u;
                bool isMlGiOut = (captureMlData & ML_CAPTURE_ML_DENOISER_MASK) != 0u;
                bool mergeMatteSpec = (!isMlGiOut) && (giRoughness >= 0.70 && giMetallic < 0.05);
                vec3 diffContrib = mergeMatteSpec ? (fCosW * scaleW) : (max(fCosW - specFCosW, vec3(0.0)) * scaleW);
                vec3 specContrib = mergeMatteSpec ? vec3(0.0) : (specFCosW * scaleW);
                vec3 rawTotal = diffContrib + specContrib;
                float effGiClamp = (hasGiMotion && !giShadeWinnerSelf && indirectClamp > 0.0)
                    ? min(indirectClamp, 10.0)
                    : indirectClamp;
                vec3 totalContrib = (effGiClamp > 0.0) ? clampIndirectRadiance(rawTotal, effGiClamp) : rawTotal;
                float rawLum = dot(rawTotal, vec3(0.2126, 0.7152, 0.0722));
                float clampedScale = (rawLum > 1e-8) ? (dot(totalContrib, vec3(0.2126, 0.7152, 0.0722)) / rawLum) : 1.0;
                diffContrib *= clampedScale;
                specContrib *= clampedScale;
                if (dot(totalContrib, totalContrib) > 1e-12) {
                    if (giShadeWinnerSelf || inlineVisActive || ((ubo.flags & (1u << 4)) == 0u)) {
                        if (captureMlData == 0u || (captureMlData & 2u) != 0u) {
                            vec4 prevAcc = imageLoad(uAccumImage, giPix);
                            imageStore(uAccumImage, giPix, vec4(addFp16Stochastic(prevAcc.rgb, totalContrib, seed), prevAcc.a));
                        }
                        wfAddSplitIndirect(giPix, diffContrib, specContrib, captureMlData, seed);
                        if (captureMlData != 0u) {
                            if (dot(diffContrib, diffContrib) > 1e-12) {
                                vec3 outDiff = isV4
                                    ? demodulateMlSafe(diffContrib, imageLoad(uMlAlbedoRoughnessImage, giPix).rgb)
                                    : diffContrib;
                                vec4 prevDiff = imageLoad(uMlDiffuseImage, giPix);
                                imageStore(uMlDiffuseImage, giPix, vec4(addFp16Stochastic(prevDiff.rgb, outDiff, seed), prevDiff.a));
                            }
                            if (dot(specContrib, specContrib) > 1e-12) {
                                vec3 outSpec = isV4
                                    ? demodulateMlSafe(specContrib, imageLoad(uMlSpecAlbedoImage, giPix).rgb)
                                    : specContrib;
                                vec4 prevSpec = imageLoad(uMlSpecularImage, giPix);
                                imageStore(uMlSpecularImage, giPix, vec4(addFp16Stochastic(prevSpec.rgb, outSpec, seed), prevSpec.a));
                            }
                        }
                    } else {
                        vec3 giVisOrig = offsetRayOrigin(giX1, giN1);
                        vec3 giVisTarget = offsetRayOrigin(giX2w, giGetN2(giShadeR));
                        vec3 giVisTo = giVisTarget - giVisOrig;
                        float giVisDist = length(giVisTo);
                        if (giVisDist > 2.0 * EPSILON) {
                            float diffLum = dot(diffContrib, vec3(0.2126, 0.7152, 0.0722));
                            float specLum = dot(specContrib, vec3(0.2126, 0.7152, 0.0722));
                            float specFrac = clamp(specLum / max(diffLum + specLum, 1e-8), 0.0, 1.0);
                            outHasGiShadowRay = true;
                            outGiShadowRay.originDist = vec4(giVisOrig, giVisDist);
                            outGiShadowRay.dirPixelRad = uvec4(
                                packOct32(giVisTo / giVisDist),
                                pixelIndex,
                                packHalf2x16(totalContrib.rg),
                                packHalf2x16(vec2(totalContrib.b, specFrac))
                            );
                        }
                    }
                }
            }
        }
    }
}

#ifdef RESTIR_INCLUDE_SAMPLER_HELPERS
void accumulateDiRbSample(
    StdBsdf bsdf,
    vec3 hitPoint,
    vec3 hitNormal,
    vec3 V,
    uint numLights,
    uint captureMlData,
    bool canContinue,
    bool diRestirMisActive,
    DiSample s,
    float weightOverPHat,
    uint mTap,
    float maxTapLum,
    inout vec3 rbDiDiffSum,
    inout vec3 rbDiSpecSum
) {
    if (!s.has || weightOverPHat <= 0.0 || s.target <= 1e-6) return;
    float neeBsdfPdf = 0.0;
    vec3 neeSpecFCos;
    vec3 neeFCos = stdBsdfEvalSplit(bsdf, hitNormal, V, s.dir, neeBsdfPdf, neeSpecFCos);
    if (dot(neeFCos, vec3(1.0)) <= 1e-7) return;

    float misWeight = 1.0;
    Light selLight = lights[s.lightIdx];
    uint selType = uint(selLight.position.w);
    if ((selType == 0u || selType == 3u) && lightUsesMis(s.lightIdx)) {
        if (!canContinue || misReferenceMode() ||
            (ubo.flags & UBO_FLAG_INDIRECT_LIGHT) == 0u ||
            (ubo.flags & UBO_FLAG_DBG_NO_NEE_AREA) != 0u) {
            misWeight = neeMisWeightScaled(s.lightIdx, 0.0, neeBsdfPdf, canContinue, 1.0);
        } else {
            float selectPdf = s.selectPdf;
            if (selectPdf < 0.0) {
                selectPdf = misLightSelectPdf(s.lightIdx, hitPoint, numLights);
            }
            float effCos = max(s.cosLight, 1e-4);
            float effDist = max(s.dist, 1e-4);
            float lightPdfEff = ((effDist * effDist) / (selLight.emission.w * effCos)) * selectPdf;
            float diMisScale = diRestirMisActive ? restirDiMisLightScale(numLights, captureMlData) : 1.0;
            misWeight = neeMisWeightScaled(s.lightIdx, lightPdfEff, neeBsdfPdf, canContinue, diMisScale);
        }
    }

    vec3 neeScale = s.emission * (misWeight * weightOverPHat);
    vec3 diffContrib = max(neeFCos - neeSpecFCos, vec3(0.0)) * neeScale;
    vec3 specContrib = neeSpecFCos * neeScale;
    if (maxTapLum > 0.0 && mTap > 0u) {
        float tapLum = dot(diffContrib + specContrib, vec3(0.2126, 0.7152, 0.0722));
        float lumCap = maxTapLum * float(mTap);
        if (tapLum > lumCap) {
            float sc = lumCap / tapLum;
            diffContrib *= sc;
            specContrib *= sc;
        }
    }
    rbDiDiffSum += diffContrib;
    rbDiSpecSum += specContrib;
}
#endif

#endif // RESTIR_GI_REUSE_GLSL
