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
        rayQueryEXT rq;
        rayQueryInitializeEXT(rq, topLevelAS,
                              gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsSkipClosestHitShaderEXT,
                              0xFF, origin, EPSILON, dir, maxDist - EPSILON * 2.0);
        while (rayQueryProceedEXT(rq)) {
            if (rayQueryGetIntersectionTypeEXT(rq, false) == gl_RayQueryCandidateIntersectionTriangleEXT) {
                uint geomIdx = rayQueryGetIntersectionGeometryIndexEXT(rq, false);
                uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, false);
                uint triIdx = (geomIdx == 0u) ? primIdx : (primIdx + numOpaqueTriangles);
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
                        calpha *= texture(sceneTextures[mat.albedoTex - 1u], cuv).a;
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
        if (rayQueryGetIntersectionTypeEXT(rq, true) != gl_RayQueryCommittedIntersectionNoneEXT) {
            uint geomIdx = rayQueryGetIntersectionGeometryIndexEXT(rq, true);
            uint primIdx = rayQueryGetIntersectionPrimitiveIndexEXT(rq, true);
            uint triIdx = (geomIdx == 0u) ? primIdx : (primIdx + numOpaqueTriangles);
            Material m = materials[triangles[triIdx].materialId];
            if (m.type != 3u && m.type != 2u && m.transmission <= 0.05) {
                occluded = true;
            }
        }
    }

    if (!occluded && numSpheres > 0u) {
        for (uint i = 0; i < numSpheres; ++i) {
            if (materials[spheres[i].materialId].type == 3u ||
                materials[spheres[i].materialId].type == 2u ||
                materials[spheres[i].materialId].transmission > 0.05) continue;
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
