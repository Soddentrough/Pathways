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
    if (captureMlData != 0u && roughness >= 0.70 && metallic < 0.05) {
        b.specProb = 0.0;
    }
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
    inout uint seed
) {
    if (!giIsValid(tap) || giGetM(tap) == 0u) return false;
    if (giGetAge(tap) >= GI_MAX_AGE) return false;
    if (dot(n1, giGetX1Normal(tap)) < RESTIR_TAP_NORMAL_MIN_DOT) return false;

    uint mSrc = giGetM(tap);
    uint mTap = min(mSrc, mCap);
    if (mTap == 0u) return false;

    float pTap = giGetTarget(tap);
    float tapWSum = giGetWSum(tap);
    float weight = 0.0;
    float pHat = 0.0;
    vec3 x2c = vec3(0.0);
    vec3 n2c = vec3(0.0, 1.0, 0.0);
    vec3 radC = vec3(0.0);
    float dNew = 0.0;
    float cosRcNew = 0.0;

    if (pTap > 0.0 && tapWSum > 0.0 && giHasSurface(tap)) {
        x2c = giGetX2(tap);
        n2c = giGetN2(tap);
        radC = giGetRadiance(tap);
        vec3 toNew = x2c - x1;
        dNew = length(toNew);
        if (dNew >= GI_TAP_MIN_DIST) {
            vec3 dirNew = toNew / dNew;
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
                        float targetRatio = pHat / max(pTap, 1e-6);
                        weight = (tapWSum / float(mSrc)) * float(mTap) * targetRatio * jac;
                        if (isnan(weight) || isinf(weight) || !(weight > 0.0)) {
                            weight = 0.0;
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

void restirGiRunReuse(
    uint pixelIndex,
    uint width,
    uint height,
    uint frameIndex,
    uint sppSampleIndex,
    uint restirMCap,
    uint captureMlData,
    float indirectClamp,
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

    vec3 giN1 = x1ContextNormal(giCtx);
    vec3 giWo = x1ContextWo(giCtx);
    float giMetallic = x1ContextMetallic(giCtx);
    float giRoughness = x1ContextRoughness(giCtx);
    float giQSelf = x1ContextQSelf(giCtx);
    StdBsdf giBsdf = giReconstructBsdf(giCtx, captureMlData);

    GiReservoir giR = makeGiReservoir(false, vec3(0.0), vec3(0.0, 1.0, 0.0), giN1, 0.0, 1u, 0.0, vec3(0.0), 0.0, 0.0, 0u);
    bool giWinnerSelf = false;
    if (hasSelfCandidate) {
        float giPSelf = giTargetAtBsdf(giBsdf, giN1, giWo, giX1, selfX2, giL2);
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
    bool canReuseHistory = (ubo.frameIndex > 0u) && ((ubo.flags & (1u << 23)) == 0u);

    if (canReuseHistory) {
        mvData = imageLoad(uMotionVectorImage, giPix);
        float selfDepth = max(dot(giX1 - ubo.position.xyz, -ubo.viewInverse[2].xyz), 0.0);
        expectedPrevDepth = max(selfDepth + mvData.z, 1e-4);
        giPrev = ivec2(round(vec2(giPix) - mvData.xy * vec2(float(width), float(height))));

        if (abs(mvData.x) < 1.5 && abs(mvData.y) < 1.5 &&
            giPrev.x >= 0 && giPrev.y >= 0 && giPrev.x < int(width) && giPrev.y < int(height)) {
            float prevTapDepth = imageLoad(uPrevNormalDepthImage, giPrev).w;
            float relDepthDiff = abs(expectedPrevDepth - prevTapDepth) / max(max(expectedPrevDepth, prevTapDepth), 1e-4);
            if (relDepthDiff <= RESTIR_TAP_DEPTH_REL_GATE) {
                GiReservoir giTap = giHistoryReservoirs[uint(giPrev.y) * width + uint(giPrev.x)];
                uint tempMCap = min(restirMCap, 20u);
                uint giAgeNew = min(giGetAge(giTap) + 1u, GI_MAX_AGE);
                if (giApplyTap(giR, giTap, giBsdf, giX1, giN1, giWo, giRoughness, tempMCap, giAgeNew, seed)) {
                    giWinnerSelf = false;
                }
            }
        }
    }

    // Save pure TEMPORAL reservoir to history (never feed spatial reuse back into temporal history)
    if (sppSampleIndex == 0u) {
        giReservoirs[pixelIndex] = giIsValid(giR) ? giR : invalidGiReservoir();
    }

    GiReservoir giShadeR = giR;
    bool giShadeWinnerSelf = giWinnerSelf;

    if (canReuseHistory && abs(mvData.x) < 1.5 && abs(mvData.y) < 1.5) {
        uint tapHash = (pixelIndex * 0x9E3779B9u) ^ (frameIndex * 0x85EBCA6Bu);
        uint rot = (tapHash >> 30u) & 3u;
        uint spatialMCap = min(restirMCap, 8u);

        for (uint tap = 0u; tap < 4u; ++tap) {
            ivec2 base = RESTIR_TAP_OFFSETS[tap];
            for (uint rr = 0u; rr < rot; ++rr) {
                base = ivec2(-base.y, base.x);
            }
            ivec2 giCoord = giPrev + base;
            if (giCoord.x < 0 || giCoord.y < 0 || giCoord.x >= int(width) || giCoord.y >= int(height)) {
                continue;
            }
            float prevTapDepth = imageLoad(uPrevNormalDepthImage, giCoord).w;
            float relDepthDiff = abs(expectedPrevDepth - prevTapDepth) / max(max(expectedPrevDepth, prevTapDepth), 1e-4);
            if (relDepthDiff > RESTIR_TAP_DEPTH_REL_GATE) continue;

            GiReservoir giTap = giHistoryReservoirs[uint(giCoord.y) * width + uint(giCoord.x)];
            if (giApplyTap(giShadeR, giTap, giBsdf, giX1, giN1, giWo, giRoughness, spatialMCap, giGetAge(giTap), seed)) {
                giShadeWinnerSelf = false;
            }
        }
    }

    if (giIsValid(giShadeR) && giHasSurface(giShadeR) && giGetM(giShadeR) > 0u && giGetTarget(giShadeR) > 0.0) {
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
                    giW = min(giW, GI_WEIGHT_CLAMP_MAX);
                }
                float invSpp = (ubo.spp > 0u) ? (1.0 / float(ubo.spp)) : 1.0;
                vec3 scaleW = giL2w * (giW * invSpp);

                bool mergeMatteSpec = (giRoughness >= 0.70 && giMetallic < 0.05);
                vec3 diffContrib = mergeMatteSpec ? (fCosW * scaleW) : (max(fCosW - specFCosW, vec3(0.0)) * scaleW);
                vec3 specContrib = mergeMatteSpec ? vec3(0.0) : (specFCosW * scaleW);
                if (indirectClamp > 0.0) {
                    diffContrib = clampIndirectRadiance(diffContrib, indirectClamp);
                    specContrib = clampIndirectRadiance(specContrib, indirectClamp);
                }
                vec3 totalContrib = diffContrib + specContrib;
                if (dot(totalContrib, totalContrib) > 1e-12) {
                    if (giShadeWinnerSelf) {
                        if (captureMlData == 0u || (captureMlData & 2u) != 0u) {
                            vec4 prevAcc = imageLoad(uAccumImage, giPix);
                            imageStore(uAccumImage, giPix, vec4(addFp16Stochastic(prevAcc.rgb, totalContrib, seed), prevAcc.a));
                        }
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
                    } else {
                        float diffLum = dot(diffContrib, vec3(0.2126, 0.7152, 0.0722));
                        float specLum = dot(specContrib, vec3(0.2126, 0.7152, 0.0722));
                        float specFrac = clamp(specLum / max(diffLum + specLum, 1e-8), 0.0, 1.0);
                        outHasGiShadowRay = true;
                        outGiShadowRay.originDist = vec4(giX1, giDistW);
                        outGiShadowRay.dirPixelRad = uvec4(
                            packOct32(giDirW),
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

#endif // RESTIR_GI_REUSE_GLSL
