#ifndef SHADING_CORE_GLSL
#define SHADING_CORE_GLSL

// =============================================================================
// Shared Wavefront Traversal & NEE Core (Phase 4)
// =============================================================================
// Requires before inclusion:
//   - wavefront_common.glsl
//   - CameraUBO `ubo` (with `.flags`)
//   - topLevelAS, triangles[], spheres[], instances[], shadeMaterials[],
//     DECLARE_SCENE_TEXTURES_BINDING(8)
// Optional (when SHADING_CORE_ENABLE_NEE is defined):
//   - lights[], lightNodes[], bsdf_standard.glsl, mis_common.glsl
// =============================================================================

#ifndef ENABLE_ALPHA_MASK
#define ENABLE_ALPHA_MASK 1
#endif

#ifndef SHADING_CORE_SKIP_SCENE_HIT
void traceClosestSceneHit(
    vec3 origin,
    vec3 dir,
    float tMin,
    float tMax,
    uint numSpheres,
    out float closestT,
    out int closestTri,
    out int closestSphere,
    out vec2 closestBary,
    out vec3 closestSphereNorm,
    out uint hitInstIdx,
    out uint hitInstMatOffset,
    out mat4x3 hitO2w,
    out mat4x3 hitW2o
) {
    closestT = tMax;
    closestTri = -1;
    closestSphere = -1;
    closestBary = vec2(0.0);
    closestSphereNorm = vec3(0.0, 1.0, 0.0);
    hitInstIdx = 0u;
    hitInstMatOffset = 0u;
    hitO2w = mat4x3(1.0);
    hitW2o = mat4x3(1.0);

    rayQueryEXT rq;
#if ENABLE_ALPHA_MASK
    bool hasAlphaMask = (ubo.flags & (1u << 10)) != 0u;
    if (!hasAlphaMask) {
        rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT, RAY_MASK_ALL, origin, tMin, dir, closestT);
        rayQueryProceedEXT(rq);
    } else {
        rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsNoneEXT, RAY_MASK_ALL, origin, tMin, dir, closestT);
        while (rayQueryProceedEXT(rq)) {
            if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
                uint cInstIdx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false);
                uint cGeomIdx = rayQueryGetIntersectionGeometryIndexEXT(rq, false);
                uint cPrimIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
                InstanceGPU cInst = instances[cInstIdx];
                uint cTriIdx = cInst.firstTriangle + ((cGeomIdx == 0u) ? cPrimIdx : (cPrimIdx + cInst.numOpaqueTriangles));
                uint cMatId = triangles[cTriIdx].materialId + cInst.materialOffset;
                ShadeMaterial sm = shadeMaterials[cMatId];
                if (smGetAlphaMode(sm) == 1u /* ALPHA_MODE_MASK */) {
                    vec2 cBary = rayQueryGetIntersectionBarycentricsEXT(rq, false);
                    float cw = 1.0 - cBary.x - cBary.y;
                    vec2 cUv = cw * vec2(triangles[cTriIdx].uv0x, triangles[cTriIdx].uv0y)
                             + cBary.x * vec2(triangles[cTriIdx].uv1x, triangles[cTriIdx].uv1y)
                             + cBary.y * vec2(triangles[cTriIdx].uv2x, triangles[cTriIdx].uv2y);
                    float cAlpha = sm.albedo.a;
                    uint albTex = smGetAlbedoTex(sm);
                    if (albTex > 0u && albTex <= 512u) {
                        cAlpha *= SAMPLE_SCENE_TEXTURE(albTex, cUv).a;
                    }
                    if (cAlpha < smGetAlphaCutoff(sm)) {
                        continue;
                    }
                }
                rayQueryConfirmIntersectionEXT(rq);
            }
        }
    }
#else
    rayQueryInitializeEXT(rq, topLevelAS, gl_RayFlagsOpaqueEXT, RAY_MASK_ALL, origin, tMin, dir, closestT);
    rayQueryProceedEXT(rq);
#endif

    if (rayQueryGetIntersectionTypeEXT(rq, true) == gl_RayQueryCommittedIntersectionTriangleEXT) {
        closestT = rayQueryGetIntersectionTEXT(rq, true);
        hitInstIdx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, true);
        uint geomIdx = rayQueryGetIntersectionGeometryIndexEXT(rq, true);
        uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
        InstanceGPU inst = instances[hitInstIdx];
        hitInstMatOffset = inst.materialOffset;
        closestTri = int(inst.firstTriangle + ((geomIdx == 0u) ? primIdx : (primIdx + inst.numOpaqueTriangles)));
        closestBary = rayQueryGetIntersectionBarycentricsEXT(rq, true);
        hitO2w = rayQueryGetIntersectionObjectToWorldEXT(rq, true);
        hitW2o = rayQueryGetIntersectionWorldToObjectEXT(rq, true);
    }

    for (uint i = 0u; i < numSpheres; ++i) {
        float spT;
        vec3 spNorm;
        if (intersectSphere(origin, dir, spheres[i], tMin, closestT, spT, spNorm)) {
            closestT = spT;
            closestSphere = int(i);
            closestTri = -1;
            closestSphereNorm = spNorm;
        }
    }
}
#endif // SHADING_CORE_SKIP_SCENE_HIT

#ifdef SHADING_CORE_ENABLE_NEE
bool evalStandardDirectNEE(
    StdBsdf bsdf,
    vec3 hitPoint,
    vec3 hitNormal,
    vec3 V,
    uint numLights,
    uint numCandidates,
    bool canContinue,
    inout uint seed,
    out vec3 outShadowOrigin,
    out vec3 outLightDir,
    out float outLightDist,
    out vec3 outDiffContrib,
    out vec3 outSpecContrib
) {
    outShadowOrigin = vec3(0.0);
    outLightDir = vec3(0.0);
    outLightDist = 0.0;
    outDiffContrib = vec3(0.0);
    outSpecContrib = vec3(0.0);
    if (numLights == 0u) return false;

    bool useLightTree = ((ubo.flags & UBO_FLAG_LIGHT_TREE) != 0u) && (numLights > 1u);
    uint M = max(numCandidates, 1u);
    float wSum = 0.0;
    float selTarget = 0.0;
    vec3 selDiffScaled = vec3(0.0);
    vec3 selSpecScaled = vec3(0.0);
    vec3 selDir = vec3(0.0);
    float selDist = 0.0;
    vec3 selOrigin = vec3(0.0);

    for (uint c = 0u; c < M; ++c) {
        uint lightIdx = 0u;
        float lightPickPdf = 1.0;
        if (useLightTree) {
            SAMPLE_LIGHT_TREE(numLights, hitPoint, hitNormal, seed, lightIdx, lightPickPdf);
        } else {
            SAMPLE_LIGHT_ALIAS(numLights, seed, lightIdx, lightPickPdf);
        }
        Light light = lights[lightIdx];
        uint lightType = uint(light.position.w + 0.5);

        vec3 lightDir = vec3(0.0, 1.0, 0.0);
        float lightDist = 0.0;
        vec3 candEmission = light.emission.rgb;
        float candPdfSolidAngle = 1.0;
        bool validCand = false;

        if (lightType == 2u /* DIRECTIONAL */) {
            lightDir = normalize(light.normal.xyz);
            lightDist = 10000.0;
            candPdfSolidAngle = 1.0;
            validCand = true;
        } else if (lightType == 1u /* SPOT */) {
            vec3 toLight = light.position.xyz - hitPoint;
            float distSq = max(dot(toLight, toLight), 1e-4);
            lightDist = sqrt(distSq);
            lightDir = toLight / lightDist;
            float thetaCos = dot(-lightDir, light.normal.xyz);
            float innerCos = light.u.w;
            float outerCos = light.v.w;
            if (thetaCos >= outerCos) {
                float spotAtten = clamp((thetaCos - outerCos) / max(innerCos - outerCos, 1e-4), 0.0, 1.0);
                candEmission *= spotAtten / distSq;
                candPdfSolidAngle = 1.0;
                validCand = true;
            }
        } else /* AREA QUAD or AREA TRIANGLE */ {
            vec2 ruv = randVec2(seed);
            if (lightType == 3u /* AREA TRIANGLE */) {
                float r1 = sqrt(ruv.x);
                ruv = vec2(1.0 - r1, ruv.y * r1);
            }
            vec3 lightPos = light.position.xyz + light.u.xyz * ruv.x + light.v.xyz * ruv.y;
            vec3 toLight = lightPos - hitPoint;
            float distSq = max(dot(toLight, toLight), 1e-4);
            lightDist = sqrt(distSq);
            lightDir = toLight / lightDist;
            float cosLight = dot(light.normal.xyz, -lightDir);
            if (cosLight > 0.0 && light.emission.w > 0.0) {
                candPdfSolidAngle = distSq / max(light.emission.w * cosLight, 1e-12);
                validCand = true;
            }
        }

        if (!validCand) continue;

        float bsdfPdf = 0.0;
        vec3 specFCos = vec3(0.0);
        vec3 totalFCos = stdBsdfEvalSplit(bsdf, hitNormal, V, lightDir, bsdfPdf, specFCos);
        if (dot(totalFCos, totalFCos) <= 1e-14) continue;

        float sourcePdf = max(candPdfSolidAngle * lightPickPdf, 1e-20);
        float misW = (lightType == 0u || lightType == 3u)
            ? neeMisWeight(lightIdx, sourcePdf, bsdfPdf, canContinue)
            : 1.0;
        if (misW <= 0.0) continue;

        vec3 candDiff = max(totalFCos - specFCos, vec3(0.0)) * candEmission * misW;
        vec3 candSpec = specFCos * candEmission * misW;
        float pHat = max(dot(candDiff + candSpec, vec3(0.2126, 0.7152, 0.0722)), 0.0);
        if (pHat <= 0.0) continue;

        float risW = pHat / sourcePdf;
        wSum += risW;
        if (randFloat(seed) * wSum <= risW || selTarget <= 0.0) {
            selTarget = pHat;
            selDiffScaled = candDiff;
            selSpecScaled = candSpec;
            selDir = lightDir;
            selDist = lightDist;
            vec3 nOff = (dot(hitNormal, lightDir) >= 0.0) ? hitNormal : -hitNormal;
            selOrigin = offsetRayOrigin(hitPoint, nOff);
        }
    }

    if (selTarget <= 0.0 || wSum <= 0.0) return false;
    float risScale = wSum / (float(M) * selTarget);
    outShadowOrigin = selOrigin;
    outLightDir = selDir;
    outLightDist = selDist;
    outDiffContrib = selDiffScaled * risScale;
    outSpecContrib = selSpecScaled * risScale;
    return true;
}
#endif // SHADING_CORE_ENABLE_NEE

#endif // SHADING_CORE_GLSL
