#ifndef NRC_RECORD_HELPER_GLSL
#define NRC_RECORD_HELPER_GLSL

// Shared helper to record a 128-byte self-training Bellman record (Müller et al. 2021)
// at the current shading vertex after continuation BSDF sampling.
// Caches outgoing indirect radiance L_ind(x_1, V) directly so Bounce-0
// and Bounce-1 direct NEE remain 100% shadow-accurate and unbiased.
void nrcEmitSelfTrainingRecord(
    vec3 hitPoint,
    vec3 hitNormal,
    vec3 V,
    uint pixelIndex,
    vec3 diffuseAlbedo,
    float nrcRoughness,
    bool validBounce,
    vec3 nextOrigin,
    vec3 nextDirection,
    vec3 nrcBsdfWeight,
    uint nextPdfCode,
    inout uint seed
) {
    if (!validBounce) {
        return;
    }

    vec3 clampedBsdfW = min(nrcBsdfWeight, vec3(2.0));
    vec3 localRad = vec3(0.0);
    vec3 tailWeight = vec3(0.0);
    float hasTail = 0.0;
    vec3 tailPos = hitPoint;
    float tailRough = 1.0;
    vec3 tailN = hitNormal;
    vec3 tailAlbedo = vec3(0.0);

    bool hasNonOpaque = (ubo.flags & (1u << 5)) != 0u;
    rayQueryEXT tailRq;
    rayQueryInitializeEXT(
        tailRq, topLevelAS,
        gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
        hasNonOpaque ? RAY_MASK_OPAQUE : RAY_MASK_ALL,
        nextOrigin, EPSILON, nextDirection, 10000.0
    );
    rayQueryProceedEXT(tailRq);

    float bestT = 10000.0;
    bool hitTri = (rayQueryGetIntersectionTypeEXT(tailRq, true) == gl_RayQueryCommittedIntersectionTriangleEXT);
    if (hitTri) {
        bestT = rayQueryGetIntersectionTEXT(tailRq, true);
    }

    bool hitOpaqueSurface = false;
    ShadeMaterial tMat;
    uint tType = 0u;
    vec3 tAlb = vec3(0.0);
    vec3 tEmit = vec3(0.0);
    float tMisW = 1.0;

    if (hitTri) {
        uint instIdx = rayQueryGetIntersectionInstanceCustomIndexEXT(tailRq, true);
        uint geomIdx = rayQueryGetIntersectionGeometryIndexEXT(tailRq, true);
        uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(tailRq, true);
        vec2 bary = rayQueryGetIntersectionBarycentricsEXT(tailRq, true);
        mat4x3 tailW2o = rayQueryGetIntersectionWorldToObjectEXT(tailRq, true);

        InstanceGPU inst = instances[instIdx];
        uint triIdx = inst.firstTriangle + ((geomIdx == 0u) ? primIdx : (primIdx + inst.numOpaqueTriangles));
        Triangle tri = triangles[triIdx];
        uint tMatId = tri.materialId + inst.materialOffset;
        tMat = shadeMaterials[tMatId];
        tType = smGetTypeFlags(tMat);

        vec3 objN = getTriangleNormal(tri, bary);
        vec3 rawTriN = normalize(transpose(mat3(tailW2o)) * objN);
        bool hitFrontFace = (dot(nextDirection, rawTriN) < 0.0);
        tailN = hitFrontFace ? rawTriN : -rawTriN;
        vec2 tUv = getTriangleUV(tri, bary);

        tAlb = tMat.albedo.rgb;
        uint tAlbTex = smGetAlbedoTex(tMat);
        if (tAlbTex > 0u && tAlbTex <= 512u) {
            tAlb *= SAMPLE_SCENE_TEXTURE_LOD(tAlbTex, tUv, 4.0).rgb;
        }
        tEmit = tMat.emissive_spec.rgb;
        uint tEmTex = smGetEmissiveTex(tMat);
        if (tEmTex > 0u && tEmTex <= 512u && (tType & MATERIAL_FLAG_PROCEDURAL_HOLO) == 0u) {
            tEmit *= SAMPLE_SCENE_TEXTURE_LOD(tEmTex, tUv, 4.0).rgb;
        }
        // Balance-heuristic MIS partition on front-facing emitter hits along x_1 -> x_2,
        // matching NEE's weight at x_1 so close-range 1/d^2 emitter variance is eliminated
        // while back-face fixture cavities do not smear across the MLP.
        if (tri.padding0 == TRI_LIGHT_NONE) {
            tMisW = 1.0;
        } else if (!hitFrontFace || tri.padding0 == TRI_LIGHT_NEE_ONLY) {
            tMisW = 0.0;
        } else {
            tMisW = emitterHitMisWeight(0u, triIdx, instIdx, nextOrigin, nextDirection, bestT, nextPdfCode, pc.numLights);
        }
        hitOpaqueSurface = true;
    }

    if (hitOpaqueSurface) {
        float tRough = clamp(tMat.pbrParams.x, 0.04, 1.0);
        float tMet = clamp(tMat.pbrParams.y, 0.0, 1.0);
        if ((tType & 0xFFu) == 1u) {
            tMet = max(tMet, 1.0);
        }
        bool isDeltaOrTransTail = ((tType & MATERIAL_TYPE_MASK) == 2u) || smIsTransmissive(tMat) || (tMet >= 0.9 && tRough < 0.20);
        if (!isDeltaOrTransTail) {
            float nrcCap = min(pc.indirectClamp, 24.0);
            if (length(tEmit) > 1e-3 && tMisW > 0.0) {
                localRad += clampedBsdfW * clampIndirectRadiance(tEmit * tMisW, nrcCap);
            }

            if ((tType & MATERIAL_TYPE_MASK) != 3u) {
                tailPos = nextOrigin + bestT * nextDirection;
                float tIor = tMat.pbrParams.z > 0.0 ? tMat.pbrParams.z : 1.5;
                float tF0Dielec = pow((tIor - 1.0) / (tIor + 1.0), 2.0);
                vec3 tDiff = tAlb * (1.0 - tMet);
                vec3 tF0 = mix(vec3(clamp(tF0Dielec * tMat.emissive_spec.w, 0.0, 1.0)), tAlb, tMet);

            if (((ubo.flags & UBO_FLAG_DIRECT_LIGHT) != 0u) && pc.numLights > 0u) {
                uint lIdx = 0u;
                float lSelPdf = 1.0;
                if ((ubo.flags & UBO_FLAG_LIGHT_TREE) != 0u && pc.numLights > 1u) {
                    SAMPLE_LIGHT_TREE(pc.numLights, tailPos, tailN, seed, lIdx, lSelPdf);
                } else {
                    SAMPLE_LIGHT_ALIAS(pc.numLights, seed, lIdx, lSelPdf);
                }

                Light L = lights[lIdx];
                vec3 lDir = vec3(0.0, 1.0, 0.0);
                float lDist = 0.0;
                float lPdf = 1.0;
                vec3 lEm = L.emission.rgb;
                bool lValid = false;

                uint lType = uint(L.position.w);
                if (lType == 0u || lType == 3u) {
                    vec2 uv = randVec2(seed);
                    if (lType == 3u) {
                        float r1 = sqrt(uv.x);
                        uv = vec2(1.0 - r1, uv.y * r1);
                    }
                    vec3 lPt = L.position.xyz + uv.x * L.u.xyz + uv.y * L.v.xyz;
                    vec3 toL = lPt - tailPos;
                    lDist = length(toL);
                    lDir = toL / max(lDist, 1e-4);
                    float cosL = dot(-lDir, L.normal.xyz);
                    if (cosL > 0.0) {
                        lPdf = (lDist * lDist) / max(L.emission.w * cosL, 1e-12);
                        lValid = true;
                    }
                } else if (lType == 1u) {
                    vec3 toL = L.position.xyz - tailPos;
                    lDist = length(toL);
                    lDir = toL / max(lDist, 1e-4);
                    float cosSpot = dot(-lDir, L.normal.xyz);
                    if (cosSpot >= L.v.w) {
                        float spotFactor = clamp((cosSpot - L.v.w) / max(L.u.w - L.v.w, 1e-4), 0.0, 1.0);
                        lEm *= spotFactor / max(lDist * lDist, 1e-4);
                        lValid = true;
                    }
                } else if (lType == 2u) {
                    lDir = normalize(L.normal.xyz);
                    lDist = 10000.0;
                    lValid = true;
                }

                if (lValid && dot(tailN, lDir) > 0.0) {
                    StdBsdf tailBsdf = makeStdBsdf(
                        tDiff, tF0, tRough * tRough, 0.0, tMet,
                        clamp(dot(tailN, -nextDirection), 0.0, 1.0),
                        (tMet > 0.5) ? STD_BSDF_SIMPLE_METAL : STD_BSDF_SIMPLE_DIFF
                    );
                    float tailBsdfPdf = 0.0;
                    vec3 tailFCos = stdBsdfEval(tailBsdf, tailN, -nextDirection, lDir, tailBsdfPdf);
                    float lPdfEff = max(lPdf * lSelPdf, 1e-20);
                    // At x_2 (tailPos), use MIS-weighted NEE consistent with finite-horizon
                    // self-training so near-field 1/d^2 area-light spikes cannot blow up the cache.
                    float lMis = (lType == 0u || lType == 3u)
                        ? neeMisWeight(lIdx, lPdfEff, tailBsdfPdf, true)
                        : 1.0;
                    vec3 tailDirect = clampIndirectRadiance(tailFCos * lEm * (lMis / lPdfEff), nrcCap);
                    if (dot(tailDirect, tailDirect) > 1e-10) {
                        bool vis = true;
#if ENABLE_INLINE_SHADOWS
                        if ((ubo.flags & (1u << 4)) != 0u) {
                            vis = !traceShadowRayInline(
                                offsetRayOrigin(tailPos, tailN), lDir, lDist,
                                hasNonOpaque, pc.numOpaqueTriangles
                            );
                        }
#endif
                        if (vis) {
                            localRad += clampedBsdfW * tailDirect;
                        }
                    }
                }
            }

                tailRough = max(tRough, 0.20);
                tailAlbedo = max(mix(tDiff, tF0, tMet), vec3(0.04));
                float horizonScale = (pc.maxBounces == 4u) ? 0.84 : 0.90;
                tailWeight = clamp(clampedBsdfW * horizonScale, vec3(0.0), vec3(0.85));
                hasTail = 1.0;
            }
        }
    } else {
        vec3 tailSky;
        if (pc.hasEnvMap == 1u) {
            tailSky = textureLod(environmentMap, directionToEquirectangular(nextDirection), 2.0).rgb * pc.envMapIntensity;
        } else {
            float st = 0.5 * (nextDirection.y + 1.0);
            tailSky = mix(vec3(0.02, 0.02, 0.04), vec3(0.05, 0.07, 0.1), st);
        }
        localRad += clampedBsdfW * clampIndirectRadiance(tailSky, min(pc.indirectClamp, 24.0));
    }

    uvec4 trainBallot = subgroupBallot(true);
    uint trainWaveCount = subgroupBallotBitCount(trainBallot);
    uint trainWaveBase = 0u;
    if (subgroupElect()) {
        trainWaveBase = atomicAdd(nrcCounters.trainCount, trainWaveCount);
    }
    trainWaveBase = subgroupBroadcastFirst(trainWaveBase);
    uint trainIdx = trainWaveBase + subgroupBallotExclusiveBitCount(trainBallot);
    uint maxTrainRecords = max(16384u, (pc.width * pc.height) >> 4u);
    if (trainIdx >= maxTrainRecords) {
        return;
    }

    vec3 normLocal = clampIndirectRadiance(localRad, min(pc.indirectClamp, 24.0));
    nrcTrainRecords[trainIdx].pos_roughness = vec4(hitPoint, nrcRoughness);
    nrcTrainRecords[trainIdx].normal_flags = vec4(hitNormal, 1.0);
    nrcTrainRecords[trainIdx].dir_pixelIndex = vec4(V, uintBitsToFloat(pixelIndex));
    nrcTrainRecords[trainIdx].albedo_pad = vec4(diffuseAlbedo, 0.0);
    nrcTrainRecords[trainIdx].throughput = vec4(tailWeight, hasTail);
    nrcTrainRecords[trainIdx].targetRadiance = vec4(normLocal, 0.0);
    nrcTrainRecords[trainIdx].tailPos_roughness = vec4(tailPos, tailRough);
    nrcTrainRecords[trainIdx].tailPacked = vec4(
        uintBitsToFloat(nrcPackUnitVec3(tailN)),
        uintBitsToFloat(nrcPackUnitVec3(-nextDirection)),
        uintBitsToFloat(packUnorm4x8(vec4(clamp(tailAlbedo, 0.0, 1.0), 0.0))),
        0.0
    );
}

#endif // NRC_RECORD_HELPER_GLSL
