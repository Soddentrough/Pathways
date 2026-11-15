#ifndef MIS_COMMON_GLSL
#define MIS_COMMON_GLSL

// =============================================================================
// Consistent NEE <-> BSDF-sampling multiple importance sampling (plan P1a).
// =============================================================================
// The balance heuristic is unbiased only when, for every emitter point y seen
// from a vertex x, the NEE weight and the BSDF-hit weight are computed from
// the SAME pair of densities and therefore sum to exactly one:
//
//   w_light(y) = p_l(y) / (p_l(y) + p_b(y)),  w_bsdf(y) = p_b(y) / (p_l(y) + p_b(y))
//
//   p_l = lightSelectPdf(i, x) * d^2 / (A_i * cosL)    (solid-angle measure)
//   p_b = quantized mixture pdf of the BSDF sampler at x
//
// The BSDF side learns p_b through the ray state (12-bit log2 code, see
// packRayState); NEE quantizes its p_b with the same codec so both sides use
// bit-identical values. RIS on the NEE side does not change this: the RIS
// estimator is unbiased for ∫ f·w_light for any target function.
//
// Emitter triangles carry their light index in Triangle.padding0
// (src/scene/LightMapping.hpp):
//   TRI_LIGHT_NONE      no light covers it -> BSDF hit is the only estimator
//   lightIdx + 1        balance heuristic against light lightIdx
//   TRI_LIGHT_NEE_ONLY  NEE owns it (light without LIGHT_FLAG_MIS) -> weight 0
//
// Known approximation: the BSDF side evaluates p_l from the epsilon-offset ray
// origin, NEE from the surface point (relative error ~ EPSILON / distance).
//
// Requires before inclusion: `lights[]`, `lightNodes[]`, `triangles[]` storage
// buffers and the camera UBO `ubo` (for flags).
// =============================================================================

#define TRI_LIGHT_NONE        0u
#define TRI_LIGHT_NEE_ONLY    0xFFFFFFFFu
#define TRI_LIGHT_INSTANCED_BIT 0x80000000u // low 31 bits: quad index within the prototype
#define LIGHT_FLAG_MIS        1u
#define UBO_FLAG_DIRECT_LIGHT (1u << 0)
#define UBO_FLAG_INDIRECT_LIGHT (1u << 1)
#define UBO_FLAG_LIGHT_TREE   (1u << 7)
#define UBO_FLAG_REFERENCE    (1u << 25)
// Diagnostics only (PATHWAYS_DEBUG_STRATEGY env var, Engine.cpp): suppress one
// estimator so per-strategy energy / variance can be measured. Biased by design.
#define UBO_FLAG_DBG_NO_BSDF_EMIT (1u << 28) // drop emission found by BSDF-sampled rays (pdfCode != 0)
#define UBO_FLAG_DBG_NO_NEE_AREA  (1u << 29) // drop NEE to area lights

// ---- BSDF pdf codec (shared by NEE and the ray state) -----------------------
// code 0 is reserved for "no MIS at the previous vertex" (camera ray, delta
// lobe, or a shader that performs no NEE). Codes 1..4095 cover
// pdf in [2^-20, 2^44) with 1/64-octave (~1.1 %) steps.
uint encodeBsdfPdf(float pdf) {
    if (!(pdf > 0.0)) return 1u;
    float c = round((log2(pdf) + 20.0) * 64.0);
    return uint(clamp(c, 1.0, 4095.0));
}

float decodeBsdfPdf(uint code) {
    return exp2(float(code) * (1.0 / 64.0) - 20.0);
}

float quantizeBsdfPdf(float pdf) {
    return decodeBsdfPdf(encodeBsdfPdf(pdf));
}

bool misReferenceMode() {
    return (ubo.flags & UBO_FLAG_REFERENCE) != 0u;
}

bool lightUsesMis(uint lightIdx) {
    return (floatBitsToUint(lights[lightIdx].pad0.x) & LIGHT_FLAG_MIS) != 0u;
}

// ---- Light selection pdf -----------------------------------------------------
// Replays the SAMPLE_LIGHT_TREE descent along the light's stored path
// (pad0.y = branch bits, bit d set = right child; pad0.z = depth), using the
// exact same importance, fallback and clamping as the sampler.
float lightTreeSelectPdf(uint lightIdx, vec3 x) {
    uint bits = floatBitsToUint(lights[lightIdx].pad0.y);
    uint depth = floatBitsToUint(lights[lightIdx].pad0.z);
    if (depth > 32u) return 0.0;
    uint node = 0u;
    float p = 1.0;
    for (uint d = 0u; d < depth; ++d) {
        LightTreeNode n = lightNodes[node];
        if (n.children.z == 1u) return 0.0;
        float impL = evalLightTreeNodeImportance(x, vec3(0.0), lightNodes[n.children.x]);
        float impR = evalLightTreeNodeImportance(x, vec3(0.0), lightNodes[n.children.y]);
        float tot = impL + impR;
        if (tot <= 1e-12) {
            impL = 0.5;
            tot = 1.0;
        }
        float pL = clamp(impL / tot, 0.001, 0.999);
        bool right = ((bits >> d) & 1u) != 0u;
        p *= right ? (1.0 - pL) : pL;
        node = right ? n.children.y : n.children.x;
    }
    LightTreeNode leaf = lightNodes[node];
    if (leaf.children.z != 1u || leaf.children.x != lightIdx) return 0.0;
    return max(p, 1e-20);
}

float misLightSelectPdf(uint lightIdx, vec3 x, uint numLights) {
    if (numLights <= 1u) return 1.0;
    if ((ubo.flags & UBO_FLAG_LIGHT_TREE) != 0u) {
        return lightTreeSelectPdf(lightIdx, x);
    }
    return max(lights[lightIdx].sampling.z, 1e-20);
}

// ---- NEE side ------------------------------------------------------------------
// lightPdfSolidAngle: the full NEE pdf of the sample (selection x area->solid
// angle), bsdfPdf: unquantized mixture pdf of the BSDF sampler at x.
// canContinue: whether this vertex can spawn a continuation ray at all
// (bounce + 1 < maxBounces). Both strategies must cover the same set of path
// lengths: on the last vertex BSDF sampling cannot reach a hittable emitter
// (and the BSDF-only reference cannot either), so NEE to MIS lights would add
// paths one segment longer than the budget -> weight 0. NEE-only lights keep
// weight 1 (they are NEE'd at every vertex in the reference as well).
// With indirect lighting disabled there is no BSDF strategy -> weight 1.
// Deterministic effective light-sample multiplier for Bounce-0 ReSTIR DI vs Bounce-1 BSDF emitter hits.
// Depends solely on uniform frame/scene parameters so NEE and BSDF-hit weights sum to 1.0 pointwise.
float restirDiMisLightScale(uint numLights) {
    float freshM = float(max(min(numLights, 8u), 1u));
    bool hasHistory = (ubo.frameIndex > 0u) && ((ubo.flags & (1u << 23)) == 0u);
    if (!hasHistory) return freshM;
    return min(freshM * 3.0, 40.0);
}

#ifndef ML_CAPTURE_ML_DENOISER_MASK
#define ML_CAPTURE_ML_DENOISER_MASK 12u
#endif

uint restirDiFreshM(uint numLights, uint captureMlData) {
    if ((captureMlData & ML_CAPTURE_ML_DENOISER_MASK) != 0u) return max(min(numLights, 8u), 1u);
    if (numLights <= 1u) return 1u;
    if (numLights > 4096u) return (captureMlData == 0u) ? 6u : 4u;
    return min(numLights, 16u);
}

float restirDiMisLightScale(uint numLights, uint captureMlData) {
    if ((captureMlData & ML_CAPTURE_ML_DENOISER_MASK) != 0u) return float(max(min(numLights, 8u), 1u));
    float freshM = float(restirDiFreshM(numLights, captureMlData));
    bool hasHistory = (ubo.frameIndex > 0u) && ((ubo.flags & (1u << 23)) == 0u);
    if (!hasHistory || numLights <= 4u) return freshM;
    if (captureMlData != 0u || numLights > 4096u) {
        return min(freshM * 2.5, 24.0);
    }
    return min(freshM * 4.0, 64.0);
}

uint restirSecondaryFreshM(uint bounce, uint numLights, bool isSpecularPath, uint captureMlData) {
    if (numLights <= 1u) return 1u;
    bool rawRestirMode = (captureMlData == 0u);
    if (numLights > 4096u) {
        if (bounce == 1u) {
            return isSpecularPath ? 2u : (rawRestirMode ? 2u : 1u);
        }
        return 1u;
    }
    if (bounce == 1u) {
        return min(numLights, isSpecularPath ? (rawRestirMode ? 32u : 16u) : (rawRestirMode ? 16u : 8u));
    }
    if (bounce == 2u) {
        return min(numLights, isSpecularPath ? 16u : (rawRestirMode ? 8u : 4u));
    }
    return min(numLights, rawRestirMode ? 4u : 2u);
}

float restirSecondaryMisLightScale(uint spawnBounce, uint numLights, uint captureMlData, uint enableNrc) {
    if (captureMlData != 0u || enableNrc != 0u || numLights <= 1u) return 1.0;
    if ((ubo.flags & (1u << 9)) == 0u || (ubo.flags & (1u << 10)) != 0u) return 1.0;
    if (spawnBounce == 1u) {
        bool hasHistory = (ubo.frameIndex > 0u) && ((ubo.flags & (1u << 23)) == 0u);
        float freshM1 = float(restirSecondaryFreshM(1u, numLights, false, captureMlData));
        return (freshM1 <= 1.0) ? 1.0 : (hasHistory ? min(freshM1 * 2.5, 40.0) : freshM1);
    }
    return float(restirSecondaryFreshM(spawnBounce, numLights, false, captureMlData));
}

float neeMisWeightScaled(uint lightIdx, float lightPdfSolidAngle, float bsdfPdf, bool canContinue, float lightSampleScale) {
    if ((ubo.flags & UBO_FLAG_DBG_NO_NEE_AREA) != 0u) return 0.0;
    if (!lightUsesMis(lightIdx)) return 1.0;
    if ((ubo.flags & UBO_FLAG_INDIRECT_LIGHT) == 0u) return 1.0;
    if (!canContinue) return 0.0;
    if (misReferenceMode()) return 0.0;
    float pl = lightPdfSolidAngle * max(lightSampleScale, 1.0);
    float pb = quantizeBsdfPdf(bsdfPdf);
    return pl / (pl + pb);
}

float neeMisWeight(uint lightIdx, float lightPdfSolidAngle, float bsdfPdf, bool canContinue) {
    return neeMisWeightScaled(lightIdx, lightPdfSolidAngle, bsdfPdf, canContinue, 1.0);
}

// ---- BSDF-hit side ------------------------------------------------------------
// Weight of emission found by a BSDF-sampled ray (origin, dir, hitT) whose
// previous vertex encoded its sampling pdf as pdfCode.
float emitterHitMisWeightScaled(uint hitType, uint primId, uint instId, vec3 origin, vec3 dir, float hitT,
                                uint pdfCode, uint numLights, float lightSampleScale) {
    if (hitType != 0u || pdfCode == 0u) return 1.0;
    if ((ubo.flags & UBO_FLAG_DBG_NO_BSDF_EMIT) != 0u) return 0.0;
    if ((ubo.flags & UBO_FLAG_DIRECT_LIGHT) == 0u || numLights == 0u) return 1.0;
    uint m = triangles[primId].padding0;
    if (m == TRI_LIGHT_NONE) return 1.0;
    if (m == TRI_LIGHT_NEE_ONLY) return 0.0;
    uint li;
    if ((m & TRI_LIGHT_INSTANCED_BIT) != 0u) {
        // Emissive-geometry light (EmissiveGeometryLights.hpp): one light per emitting quad per
        // instance. InstanceGPU::flags holds (firstLightOfInstance + 1); the low 31 bits of the
        // triangle tag are the quad's index within its prototype.
        uint base = instances[instId].flags;
        if (base == 0u) return 1.0;
        li = base - 1u + (m & ~TRI_LIGHT_INSTANCED_BIT);
        if (li >= numLights) return 1.0;
    } else {
        li = m - 1u;
        if (li >= numLights) return 1.0;
    }
    Light L = lights[li];
    float cosL = dot(-dir, L.normal.xyz);
    if (cosL <= 0.0) return 1.0;       // back side: NEE never samples it
    if (misReferenceMode()) return 1.0;
    float pl = misLightSelectPdf(li, origin, numLights) * (hitT * hitT) / max(L.emission.w * cosL, 1e-12);
    pl *= max(lightSampleScale, 1.0);
    float pb = decodeBsdfPdf(pdfCode);
    return pb / (pb + pl);
}

float emitterHitMisWeight(uint hitType, uint primId, uint instId, vec3 origin, vec3 dir, float hitT,
                          uint pdfCode, uint numLights) {
    return emitterHitMisWeightScaled(hitType, primId, instId, origin, dir, hitT, pdfCode, numLights, 1.0);
}

#endif // MIS_COMMON_GLSL
