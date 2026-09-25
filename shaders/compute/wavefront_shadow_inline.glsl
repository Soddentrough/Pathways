// Hybrid Inline Shadow Ray Evaluation for Primary Hit Points (Bounce 0)
// Evaluates direct shadow occlusion via hardware rayQueryEXT directly on-chip.

bool traceShadowRayInline(vec3 origin, vec3 dir, float maxDist, bool hasNonOpaque, uint numSpheres, uint numOpaqueTriangles) {
    bool occluded = false;

    
    bool enableCaustics = (ubo.flags & (1u << 8)) != 0u && (ubo.flags & (1u << 6)) != 0u; // approximate light check
    rayQueryEXT rq;
    rayQueryInitializeEXT(rq, topLevelAS,
                          gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                          0xFF, origin, EPSILON, dir, maxDist - EPSILON * 2.0);
    while (rayQueryProceedEXT(rq)) {
        if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
            uint instIdx = rayQueryGetIntersectionInstanceCustomIndexEXT(rq, false);
            uint geomIdx = rayQueryGetIntersectionGeometryIndexEXT(rq, false);
            uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
            InstanceGPU inst = instances[instIdx];
            uint triIdx = inst.firstTriangle + ((geomIdx == 0u) ? primIdx : (primIdx + inst.numOpaqueTriangles));
            uint matId = triangles[triIdx].materialId + inst.materialOffset;
            uint arch = materialArchetypes[matId];
            
            if (arch == 4u || (!enableCaustics && arch == 1u)) { // EMISSIVE or DIELECTRIC
                continue;
            }
            if (enableCaustics && arch == 1u) {
                if (materials[matId].thickness <= 0.001) {
                    continue;
                }
            }
            if (arch == 5u) { // ALPHAMASK
                Material mat = materials[matId];
                vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rq, false);
                Triangle ctri = triangles[triIdx];
                vec2 cuv = getTriangleUV(ctri, bary);
                float calpha = mat.albedo.a;
                if (mat.albedoTex > 0u && mat.albedoTex <= 512u) {
                    calpha *= SAMPLE_SCENE_TEXTURE(mat.albedoTex, cuv).a;
                }
                float cutoff = (mat.alphaMode == 1u) ? mat.alphaCutoff : 0.5;
                if (calpha < cutoff) {
                    continue;
                }
            }
            rayQueryConfirmIntersectionEXT(rq);
            rayQueryTerminateEXT(rq);
        }
    }
    occluded = (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT);


    if (!occluded && numSpheres > 0u) {
        bool enableCaustics = (ubo.flags & (1u << 8)) != 0u;
        for (uint i = 0; i < numSpheres; ++i) {
            uint sMatId = spheres[i].materialId;
            Material sMat = materials[sMatId];
            bool isDielectric = (sMat.type == 2u || sMat.transmission > 0.05);
            if (sMat.type == 3u || (!enableCaustics && isDielectric)) continue;
            if (enableCaustics && isDielectric && sMat.thickness <= 0.001) continue;
            float spT;
            vec3 spNorm;
            if (intersectSphere(origin, dir, spheres[i], EPSILON, maxDist - EPSILON * 2.0, spT, spNorm)) {
                occluded = true;
                break;
            }
        }
    }

    return occluded;
}
