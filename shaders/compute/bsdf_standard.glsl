#ifndef BSDF_STANDARD_GLSL
#define BSDF_STANDARD_GLSL

// =============================================================================
// STANDARD archetype BSDF (glTF metal-rough + diffuse transmission), plan P1a.
// =============================================================================
// One definition shared by NEE (evaluation) and the path continuation
// (sampling weight), so both strategies integrate the same function and use
// the same pdf for MIS.
//
//   f = spec_GGX(F(VdotH), D, Vis)                         (reflection only)
//     + (1 - F(VdotH)) * diffuseColor / pi * (1 - dt)     (reflection)
//     + (1 - F(VdotH)) * diffuseColor / pi * dt           (transmission, H from V and -L)
//
// Sampling is a one-sample mixture: spec lobe (GGX VNDF) with probability
// specProb, otherwise the cosine lobe (transmitted with probability dt).
// The continuation weight is f·|cos| / pdf_mixture (not lobe-separate), and
// pdf_mixture is what the next vertex uses for MIS.
//
// Modes:
//   STD_BSDF_FULL         the full model above
//   STD_BSDF_SIMPLE_DIFF  streamlined secondary bounce, non-metal: Lambert
//                         only (no Fresnel, no spec lobe)
//   STD_BSDF_SIMPLE_METAL streamlined secondary bounce, metal: GGX only
// The streamlined modes are a deliberate (biased w.r.t. the full model)
// approximation, but they are internally consistent: NEE and sampling agree.
// =============================================================================

#define STD_BSDF_FULL         0u
#define STD_BSDF_SIMPLE_DIFF  1u
#define STD_BSDF_SIMPLE_METAL 2u

// Below this the GGX sampler and evaluator lose agreement in fp32; surfaces
// that smooth are handled as delta mirrors by the shaders instead.
#define STD_BSDF_MIN_ALPHA    1e-3

struct StdBsdf {
    vec3 diffuseColor;  // base * (1 - metallic)
    vec3 F0;
    float alpha;        // GGX alpha (roughness^2), >= STD_BSDF_MIN_ALPHA
    float diffTrans;    // diffuse transmission fraction
    float specProb;     // probability of sampling the base spec lobe (given "not clearcoat")
    uint mode;
    bool specEnabled;   // UBO flag bit 2; false removes spec + clearcoat lobes
    // Optional layers (COMPLEX archetype, STD_BSDF_FULL only). Zero = absent.
    vec3 Nc;            // clearcoat normal
    float clearcoat;    // clearcoat factor
    float ccAlpha;      // clearcoat GGX alpha
    float ccProb;       // probability of sampling the clearcoat lobe
    vec3 sheenColor;
    float sheenRoughness;
    float sheenMax;     // max(sheenColor), base attenuation
};

float stdSpecSelectProb(vec3 F0, float NdotV, float metallic) {
    vec3 Fv = fresnelSchlickVec(NdotV, F0);
    return mix(clamp(dot(Fv, vec3(0.3333333)), 0.05, 0.95), 1.0, metallic);
}

StdBsdf makeStdBsdf(vec3 diffuseColor, vec3 F0, float alpha, float diffTrans, float metallic, float NdotV, uint mode) {
    StdBsdf b;
    b.diffuseColor = diffuseColor;
    b.F0 = F0;
    b.alpha = max(alpha, STD_BSDF_MIN_ALPHA);
    // Shaders only sample the transmission lobe above 0.001; snap below that
    // so the evaluated mixture pdf matches the sampler exactly.
    b.diffTrans = (diffTrans > 0.001) ? diffTrans : 0.0;
    b.mode = mode;
    b.specEnabled = true;
    if (mode == STD_BSDF_SIMPLE_DIFF) {
        b.specProb = 0.0;
    } else if (mode == STD_BSDF_SIMPLE_METAL) {
        b.specProb = 1.0;
    } else {
        b.specProb = stdSpecSelectProb(F0, max(NdotV, 1e-4), metallic);
    }
    b.Nc = vec3(0.0, 0.0, 1.0);
    b.clearcoat = 0.0;
    b.ccAlpha = 1.0;
    b.ccProb = 0.0;
    b.sheenColor = vec3(0.0);
    b.sheenRoughness = 1.0;
    b.sheenMax = 0.0;
    return b;
}

// COMPLEX archetype layers (KHR_materials_clearcoat / _sheen), FULL mode only.
void stdBsdfSetLayers(inout StdBsdf b, bool specEnabled, vec3 Nc, float clearcoat, float ccAlpha,
                      vec3 sheenColor, float sheenRoughness) {
    b.specEnabled = specEnabled;
    if (!specEnabled) {
        b.specProb = 0.0;
    }
    if (specEnabled && clearcoat > 0.001) {
        b.Nc = Nc;
        b.clearcoat = clearcoat;
        b.ccAlpha = max(ccAlpha, STD_BSDF_MIN_ALPHA);
        b.ccProb = clearcoat * 0.25;
    }
    float sMax = clamp(max(sheenColor.r, max(sheenColor.g, sheenColor.b)), 0.0, 1.0);
    if (sMax > 1e-4) {
        b.sheenColor = sheenColor;
        b.sheenRoughness = sheenRoughness;
        b.sheenMax = sMax;
    }
}

// Solid-angle pdf of a reflected direction under GGX VNDF sampling:
// D_V(H) / (4 VdotH) = G1(V) D(H) / (4 NdotV) = D / (2 (NdotV + sqrt(a2 + (1-a2) NdotV^2)))
float ggxVndfReflectPdf(float NdotV, float D, float alpha) {
    float a2 = max(alpha * alpha, 1e-6);
    return D / max(2.0 * (NdotV + sqrt(a2 + (1.0 - a2) * NdotV * NdotV)), 1e-12);
}

// Turquin's analytical multiple-scattering GGX energy compensation (RTXPT BxDF.hlsli)
vec3 multiScatterSpecularApprox(float alpha, float NdotV, vec3 F0) {
    float a2 = alpha * alpha;
    float ems = mix(0.2 * alpha, 0.32 * alpha + 1.94 * a2, NdotV);
    return vec3(1.0) + F0 * ems;
}

// Returns f(V, L) * |N·L| and the mixture sampling pdf of L (solid angle);
// specFCos receives the glossy part (base spec, clearcoat, sheen) of the
// returned value.
//
// Layers (FULL mode only, set by stdBsdfSetLayers):
//   sheen:     f = base * (1 - sheenMax) + sheenColor * D_charlie * V_neubelt
//   clearcoat: f = f * (1 - Fc) + Dc * Visc * Fc,  Fc = Schlick(VdotH, ior 1.5) * cc
//              (requires L and V above the clearcoat normal Nc)
// Mixture pdf: ccProb * pdf_cc(Nc) + (1 - ccProb) * [specProb * pdf_spec
//              + (1 - specProb) * pdf_cos]. Sheen is not sampled (covered by the
//              cosine and GGX lobes, both of which have full upper-hemisphere support).
vec3 stdBsdfEvalSplit(StdBsdf b, vec3 N, vec3 V, vec3 L, out float pdf, out vec3 specFCos) {
    pdf = 0.0;
    specFCos = vec3(0.0);
    float NdotV = max(dot(N, V), 1e-4);
    float NdotLraw = dot(N, L);
    float baseSel = 1.0 - b.ccProb;

    if (NdotLraw > 0.0) {
        float NdotL = NdotLraw;
        float diffSel = 1.0 - b.specProb;
        float diffPdf = (1.0 - b.diffTrans) * NdotL * INV_PI;
        if (b.mode == STD_BSDF_SIMPLE_DIFF) {
            pdf = diffPdf;
            return b.diffuseColor * (INV_PI * (1.0 - b.diffTrans) * NdotL);
        }
        vec3 H = normalize(V + L);
        float NdotH = clamp(dot(N, H), 0.0, 1.0);
        float VdotH = clamp(dot(V, H), 0.0, 1.0);
        vec3 F = fresnelSchlickVec(VdotH, b.F0);
        float D = distributionGGX(NdotH, b.alpha);
        float Vis = visibilitySmithGGXCorrelated(NdotL, NdotV, b.alpha);
        vec3 ms = multiScatterSpecularApprox(b.alpha, NdotV, b.F0);
        vec3 spec = b.specEnabled ? (D * Vis * NdotL) * F * ms : vec3(0.0);
        float specPdf = ggxVndfReflectPdf(NdotV, D, b.alpha);
        if (b.mode == STD_BSDF_SIMPLE_METAL) {
            pdf = specPdf;
            specFCos = spec;
            return spec;
        }
        vec3 diff = (vec3(1.0) - F) * b.diffuseColor * (INV_PI * (1.0 - b.diffTrans) * NdotL);
        pdf = baseSel * (b.specProb * specPdf + diffSel * diffPdf);

        if (b.sheenMax > 0.0) {
            float Ds = evalSheenCharlie(NdotH, b.sheenRoughness);
            float Vs = 1.0 / (4.0 * max(NdotL + NdotV - NdotL * NdotV, 1e-4));
            float keep = 1.0 - b.sheenMax;
            diff *= keep;
            spec = spec * keep + b.sheenColor * (Ds * Vs * NdotL);
        }

        if (b.ccProb > 0.0) {
            float NcDotV = dot(b.Nc, V);
            float NcDotL = dot(b.Nc, L);
            if (NcDotV > 0.0 && NcDotL > 0.0) {
                float NcDotH = clamp(dot(b.Nc, H), 0.0, 1.0);
                float Dc = distributionGGX(NcDotH, b.ccAlpha);
                float Visc = visibilitySmithGGXCorrelated(NcDotL, NcDotV, b.ccAlpha);
                float Fc = fresnelSchlick(VdotH, 1.5) * b.clearcoat;
                diff *= (1.0 - Fc);
                spec = spec * (1.0 - Fc) + vec3(Dc * Visc * Fc * NdotL);
                pdf += b.ccProb * ggxVndfReflectPdf(NcDotV, Dc, b.ccAlpha);
            }
        }
        specFCos = spec;
        return spec + diff;
    }

    if (NdotLraw < 0.0 && b.diffTrans > 0.001 && b.mode != STD_BSDF_SIMPLE_METAL) {
        float NdotL = -NdotLraw;
        float diffPdf = b.diffTrans * NdotL * INV_PI;
        if (b.mode == STD_BSDF_SIMPLE_DIFF) {
            pdf = diffPdf;
            return b.diffuseColor * (INV_PI * b.diffTrans * NdotL);
        }
        vec3 H = normalize(V - L);
        float VdotH = clamp(dot(V, H), 0.0, 1.0);
        vec3 F = fresnelSchlickVec(VdotH, b.F0);
        pdf = baseSel * (1.0 - b.specProb) * diffPdf;
        return (vec3(1.0) - F) * b.diffuseColor * (INV_PI * b.diffTrans * NdotL);
    }
    return vec3(0.0);
}

vec3 stdBsdfEval(StdBsdf b, vec3 N, vec3 V, vec3 L, out float pdf) {
    vec3 unusedSpec;
    return stdBsdfEvalSplit(b, N, V, L, pdf, unusedSpec);
}

// GGX VNDF half-vector sample around N (isotropic), world space.
vec3 sampleGgxVndfHalfVector(vec3 N, vec3 V, float alpha, inout uint seed) {
    vec3 b1, b2;
    buildOrthonormalBasis(N, b1, b2);
    vec3 Vl = vec3(dot(V, b1), dot(V, b2), max(dot(V, N), 1e-4));
    Vl = normalize(Vl);
    vec3 Hl = sampleVNDF_GGX(Vl, alpha, alpha, seed);
    return normalize(b1 * Hl.x + b2 * Hl.y + N * Hl.z);
}

#endif // BSDF_STANDARD_GLSL
