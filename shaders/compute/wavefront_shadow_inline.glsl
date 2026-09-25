// Hybrid Inline Shadow Ray Evaluation for Primary Hit Points (Bounce 0)
// Evaluates direct shadow occlusion via hardware rayQueryEXT directly on-chip.

bool traceShadowRayInline(vec3 origin, vec3 dir, float maxDist, bool hasNonOpaque, uint numSpheres, uint numOpaqueTriangles) {
    bool occluded = false;

    if (!hasNonOpaque) {
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topLevelAS,
                              gl_RayFlagsOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                              0xFF, origin, EPSILON, dir, maxDist - EPSILON * 2.0);
        rayQueryProceedEXT(rq);
        occluded = (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT);
    } else {
        // Pass 1: 100% Fixed-Function Hardware Traversal against Opaque Geometry
        // gl_RayFlagsCullNoOpaqueEXT culls all non-opaque geometry, ensuring zero shader interruptions
        rayQueryEXT rqOpaque;
        rayQueryInitializeEXT(rqOpaque, topLevelAS,
                              gl_RayFlagsCullNoOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                              0xFF, origin, EPSILON, dir, maxDist - EPSILON * 2.0);
        rayQueryProceedEXT(rqOpaque);
        occluded = (rayQueryGetIntersectionTypeEXT(rqOpaque, true) != gl_RayQueryCommittedIntersectionNoneEXT);

        // Pass 2: Only rays that clear solid architecture test alpha-tested / non-opaque geometry
        if (!occluded) {
            bool enableCaustics = (ubo.flags & (1u << 8)) != 0u;
            rayQueryEXT rqAlpha;
            rayQueryInitializeEXT(rqAlpha, topLevelAS,
                                  gl_RayFlagsCullOpaqueEXT | gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                                  0xFF, origin, EPSILON, dir, maxDist - EPSILON * 2.0);
            while (rayQueryProceedEXT(rqAlpha)) {
                if (rayQueryGetIntersectionTypeEXT(rqAlpha, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
                    uint instIdx = rayQueryGetIntersectionInstanceCustomIndexEXT(rqAlpha, false);
                    uint geomIdx = rayQueryGetIntersectionGeometryIndexEXT(rqAlpha, false);
                    uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rqAlpha, false);
                    InstanceGPU inst = instances[instIdx];
                    uint triIdx = inst.firstTriangle + ((geomIdx == 0u) ? primIdx : (primIdx + inst.numOpaqueTriangles));
                    uint matId = triangles[triIdx].materialId + inst.materialOffset;
                    uint arch = materialArchetypes[matId];
                    if (arch == MATERIAL_ARCHETYPE_EMISSIVE || (!enableCaustics && arch == MATERIAL_ARCHETYPE_DIELECTRIC)) {
                        continue;
                    }
                    if (enableCaustics && arch == MATERIAL_ARCHETYPE_DIELECTRIC) {
                        if (materials[matId].thickness <= 0.001) {
                            continue;
                        }
                    }
                    if (materials[matId].alphaMode != 0u /* ALPHA_MODE_OPAQUE */) {
                        Material mat = materials[matId];
                        vec2 bary = rayQueryGetIntersectionBarycentricsEXT(rqAlpha, false);
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
                    rayQueryConfirmIntersectionEXT(rqAlpha);
                    rayQueryTerminateEXT(rqAlpha);
                }
            }
            if (rayQueryGetIntersectionTypeEXT(rqAlpha, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
                occluded = true;
            }
        }
    }

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
