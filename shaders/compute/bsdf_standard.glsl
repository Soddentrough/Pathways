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
    vec3 anisoT;
    vec3 anisoB;
    float alphaX;
    float alphaY;
    float iridescence;
    float iridescenceIor;
    float iridescenceThickness;
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
    float diffLum = dot(max(diffuseColor, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
    if (mode == STD_BSDF_SIMPLE_DIFF) {
        b.specProb = 0.0;
    } else if (mode == STD_BSDF_SIMPLE_METAL || (diffLum <= 1e-6 && b.diffTrans <= 0.0)) {
        b.specProb = 1.0;
    } else {
        vec3 Fv = fresnelSchlickVec(max(NdotV, 1e-4), F0);
        float fLum = dot(max(Fv, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
        float dLum = diffLum * (1.0 - clamp(fLum, 0.0, 1.0));
        b.specProb = mix(clamp(fLum / max(fLum + dLum, 1e-6), 0.05, 0.95), 1.0, metallic);
    }
    b.Nc = vec3(0.0, 0.0, 1.0);
    b.clearcoat = 0.0;
    b.ccAlpha = 1.0;
    b.ccProb = 0.0;
    b.sheenColor = vec3(0.0);
    b.sheenRoughness = 1.0;
    b.sheenMax = 0.0;
    b.anisoT = vec3(1.0, 0.0, 0.0);
    b.anisoB = vec3(0.0, 1.0, 0.0);
    b.alphaX = b.alpha;
    b.alphaY = b.alpha;
    b.iridescence = 0.0;
    b.iridescenceIor = 1.3;
    b.iridescenceThickness = 0.0;
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

void stdBsdfSetAnisoIridescence(inout StdBsdf b, vec3 anisoT, vec3 anisoB, float alphaX, float alphaY,
                                float iridescence, float iridIor, float iridThickness) {
    b.anisoT = anisoT;
    b.anisoB = anisoB;
    b.alphaX = max(alphaX, STD_BSDF_MIN_ALPHA);
    b.alphaY = max(alphaY, STD_BSDF_MIN_ALPHA);
    b.iridescence = clamp(iridescence, 0.0, 1.0);
    b.iridescenceIor = max(iridIor, 1.0);
    b.iridescenceThickness = max(iridThickness, 0.0);
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
        if (b.iridescence > 0.001) {
            vec3 iridF = evalThinFilmIridescence(VdotH, b.iridescenceIor, b.iridescenceThickness);
            F = mix(F, iridF, b.iridescence);
        }
        float D;
        float Vis;
        float specPdf;
        if (abs(b.alphaX - b.alphaY) > 1e-5) {
            float TdotH = dot(b.anisoT, H);
            float BdotH = dot(b.anisoB, H);
            D = distributionAnisotropicGGX(TdotH, BdotH, NdotH, b.alphaX, b.alphaY);
            float Vt = dot(V, b.anisoT);
            float Vb = dot(V, b.anisoB);
            float Lt = dot(L, b.anisoT);
            float Lb = dot(L, b.anisoB);
            float lambdaV = sqrt(max(b.alphaX * b.alphaX * Vt * Vt + b.alphaY * b.alphaY * Vb * Vb + NdotV * NdotV, 1e-12));
            float lambdaL = sqrt(max(b.alphaX * b.alphaX * Lt * Lt + b.alphaY * b.alphaY * Lb * Lb + NdotL * NdotL, 1e-12));
            Vis = 0.5 / max(NdotL * lambdaV + NdotV * lambdaL, 1e-6);
            specPdf = D / max(2.0 * (NdotV + lambdaV), 1e-12);
        } else {
            D = distributionGGX(NdotH, b.alpha);
            Vis = visibilitySmithGGXCorrelated(NdotL, NdotV, b.alpha);
            specPdf = ggxVndfReflectPdf(NdotV, D, b.alpha);
        }
        vec3 ms = multiScatterSpecularApprox(b.alpha, NdotV, b.F0);
        vec3 spec = b.specEnabled ? (D * Vis * NdotL) * F * ms : vec3(0.0);
        if (b.mode == STD_BSDF_SIMPLE_METAL) {
            pdf = specPdf;
            specFCos = spec;
            return spec;
        }
        vec3 diff = (vec3(1.0) - F) * b.diffuseColor * (INV_PI * (1.0 - b.diffTrans) * NdotL);
        pdf = baseSel * (b.specProb * specPdf + diffSel * diffPdf);

        if (b.sheenMax > 0.0) {
            float Ds = evalSheenCharlie(NdotH, b.sheenRoughness);
            // Neubelt & Pettineo / glTF KHR_materials_sheen silhouette clamp (0.1) prevents grazing fireflies.
            float Vs = 1.0 / (4.0 * max(NdotL + NdotV - NdotL * NdotV, 0.1));
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
                spec = spec * (1.0 - Fc) + vec3(Dc * Visc * Fc * NcDotL);
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
        if (b.iridescence > 0.001) {
            vec3 iridF = evalThinFilmIridescence(VdotH, b.iridescenceIor, b.iridescenceThickness);
            F = mix(F, iridF, b.iridescence);
        }
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
vec3 sampleGgxVndfHalfVectorVec2(vec3 N, vec3 V, float alpha, vec2 u2) {
    vec3 b1, b2;
    buildOrthonormalBasis(N, b1, b2);
    vec3 Vl = vec3(dot(V, b1), dot(V, b2), max(dot(V, N), 1e-4));
    Vl = normalize(Vl);
    vec3 Hl = sampleVNDF_GGX_Vec2(Vl, alpha, alpha, u2);
    return normalize(b1 * Hl.x + b2 * Hl.y + N * Hl.z);
}

vec3 sampleGgxVndfHalfVector(vec3 N, vec3 V, float alpha, inout uint seed) {
    return sampleGgxVndfHalfVectorVec2(N, V, alpha, randVec2(seed));
}

// Anisotropic-capable GGX VNDF half-vector sample matching stdBsdfEvalSplit.
vec3 sampleStdGgxVndfHalfVectorVec2(StdBsdf b, vec3 N, vec3 V, vec2 u2) {
    vec3 b1, b2;
    float ax = b.alpha;
    float ay = b.alpha;
    if (abs(b.alphaX - b.alphaY) > 1e-5) {
        b1 = b.anisoT;
        b2 = b.anisoB;
        ax = b.alphaX;
        ay = b.alphaY;
    } else {
        buildOrthonormalBasis(N, b1, b2);
    }
    vec3 Vl = normalize(vec3(dot(V, b1), dot(V, b2), max(dot(V, N), 1e-4)));
    vec3 Hl = sampleVNDF_GGX_Vec2(Vl, ax, ay, u2);
    return normalize(b1 * Hl.x + b2 * Hl.y + N * Hl.z);
}

vec3 sampleStdGgxVndfHalfVector(StdBsdf b, vec3 N, vec3 V, inout uint seed) {
    return sampleStdGgxVndfHalfVectorVec2(b, N, V, randVec2(seed));
}

// Unified mixture BSDF bounce sampler matching stdBsdfEvalSplit PDF exactly.
bool sampleStdBsdfBounce(
    StdBsdf b,
    vec3 N,
    vec3 V,
    float uLobe,
    float uTrans,
    vec2 uDir,
    out vec3 outDir,
    out vec3 outWeight,
    out float outPdf,
    out bool outSampledSpec,
    out bool outTransmitted
) {
    outDir = vec3(0.0);
    outWeight = vec3(0.0);
    outPdf = 0.0;
    outSampledSpec = false;
    outTransmitted = false;

    if (b.mode == STD_BSDF_SIMPLE_DIFF) {
        if (b.diffTrans > 0.0 && uTrans < b.diffTrans) {
            outDir = sampleCosineHemisphereVec2(-N, uDir);
            outTransmitted = true;
            if (dot(-N, outDir) <= 0.0) return false;
        } else {
            outDir = sampleCosineHemisphereVec2(N, uDir);
            if (dot(N, outDir) <= 0.0) return false;
        }
    } else if (b.mode == STD_BSDF_SIMPLE_METAL) {
        vec3 H = sampleStdGgxVndfHalfVectorVec2(b, N, V, uDir);
        outDir = reflect(-V, H);
        outSampledSpec = true;
        if (dot(N, outDir) <= 0.0) return false;
    } else {
        if (b.ccProb > 0.0 && uLobe < b.ccProb) {
            vec3 Hc = sampleGgxVndfHalfVectorVec2(b.Nc, V, b.ccAlpha, uDir);
            outDir = reflect(-V, Hc);
            outSampledSpec = true;
            if (dot(N, outDir) <= 0.0 || dot(b.Nc, outDir) <= 0.0) return false;
        } else {
            float uBase = (uLobe - b.ccProb) / max(1.0 - b.ccProb, 1e-6);
            if (uBase < b.specProb) {
                vec3 H = sampleStdGgxVndfHalfVectorVec2(b, N, V, uDir);
                outDir = reflect(-V, H);
                outSampledSpec = true;
                if (dot(N, outDir) <= 0.0) return false;
            } else if (b.diffTrans > 0.0 && uTrans < b.diffTrans) {
                outDir = sampleCosineHemisphereVec2(-N, uDir);
                outTransmitted = true;
                if (dot(-N, outDir) <= 0.0) return false;
            } else {
                outDir = sampleCosineHemisphereVec2(N, uDir);
                if (dot(N, outDir) <= 0.0) return false;
            }
        }
    }

    vec3 fCos = stdBsdfEval(b, N, V, outDir, outPdf);
    if (outPdf <= 1e-7) return false;
    outWeight = fCos / outPdf;
    return true;
}

#endif // BSDF_STANDARD_GLSL
