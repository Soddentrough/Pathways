#ifndef WAVEFRONT_RESTIR_GI_GLSL
#define WAVEFRONT_RESTIR_GI_GLSL

// -----------------------------------------------------------------------------
// ReSTIR GI Reservoir GPU Layout (32 Bytes Cache-Aligned)
// -----------------------------------------------------------------------------
struct ReservoirGI {
    uint packedDir;        // Oct-encoded secondary ray direction (omega_i)
    float hitDist;         // Distance to secondary hit point (x_s)
    uint packedRadRG;      // FP16 RG incoming radiance from secondary hit
    uint packedRadB_normS; // Lower 16-bit: FP16 B radiance; Upper 16-bit: Oct-encoded secondary normal
    float wSum;            // Sum of candidate weights
    float M;               // Effective sample count
    float W;               // Final unbiased contribution weight
    uint primaryGeom;      // Primary surface: Lower 16-bit oct normal, Upper 16-bit half depth
};

// -----------------------------------------------------------------------------
// Raw Secondary Sample Buffer (32 Bytes Cache-Aligned, 128-bit Vector Coalesced)
// Populated at Bounce 0 (primary geom/albedo) and Bounce 1 (secondary hit L_o, x_s, n_s)
// -----------------------------------------------------------------------------
layout(std430, binding = 22) buffer CurrentGIReservoirBuffer {
    ReservoirGI currentGIReservoirs[];
};

layout(std430, binding = 23) readonly buffer HistoryGIReservoirBuffer {
    ReservoirGI historyGIReservoirs[];
};

// Split Raw Sample Buffer (16 Bytes per sample, Coalesced 128-bit Vector Writes)
// Primary samples: [0 .. pc.width * pc.height - 1]
// Secondary samples: [pc.width * pc.height .. 2 * pc.width * pc.height - 1]
layout(std430, binding = 24) buffer RawGISampleBuffer {
    uvec4 rawGISamples[];
};

// -----------------------------------------------------------------------------
// Geometric Normal & Depth Packing
// -----------------------------------------------------------------------------
#ifndef PACK_GEOM_DEFINED
#define PACK_GEOM_DEFINED
uint packGeom(vec3 norm, float depth) {
    vec2 oct = octEncode(norm);
    uint n16 = packSnorm4x8(vec4(oct, 0.0, 0.0)) & 0xFFFFu;
    uint d16 = packHalf2x16(vec2(0.0, depth));
    return d16 | n16;
}

void unpackGeom(uint p, out vec3 norm, out float depth) {
    depth = unpackHalf2x16(p).y;
    vec2 oct = unpackSnorm4x8(p & 0xFFFFu).xy;
    norm = octDecode(oct);
}
#endif

// -----------------------------------------------------------------------------
// Radiance and Normal Packing Utilities
// -----------------------------------------------------------------------------
void packRadianceAndNorm(vec3 rad, vec3 normS, out uint outRG, out uint outB_norm) {
    outRG = packHalf2x16(rad.rg);
    uint b16 = packHalf2x16(vec2(rad.b, 0.0));
    vec2 oct = octEncode(normS);
    uint n16 = packSnorm4x8(vec4(oct, 0.0, 0.0));
    outB_norm = (n16 << 16u) | (b16 & 0xFFFFu);
}

void unpackRadianceAndNorm(uint inRG, uint inB_norm, out vec3 outRad, out vec3 outNormS) {
    outRad.rg = unpackHalf2x16(inRG);
    outRad.b = unpackHalf2x16(inB_norm & 0xFFFFu).x;
    vec2 oct = unpackSnorm4x8(inB_norm >> 16u).xy;
    outNormS = octDecode(oct);
}

// -----------------------------------------------------------------------------
// Vector Coalesced Store Helpers (Single 128-bit write instruction on RDNA 4)
// -----------------------------------------------------------------------------
void storePrimaryGISample(uint pixelIndex, uint geom, uint albRG, uint albB, uint frameId) {
    rawGISamples[pixelIndex] = uvec4(geom, albRG, albB, frameId);
}

void storeSecondaryGISample(uint pixelIndex, uint screenPixels, vec3 rayDir, float hitDist, vec3 rad, vec3 normS) {
    uint rg = 0u;
    uint b_n = 0u;
    if (any(greaterThan(rad, vec3(1e-5)))) {
        packRadianceAndNorm(rad, normS, rg, b_n);
    }
    rawGISamples[screenPixels + pixelIndex] = uvec4(packOct32(rayDir), floatBitsToUint(hitDist), rg, b_n);
}

void storeSecondaryGISamplePackedDir(uint pixelIndex, uint screenPixels, uint packedDir, float hitDist, vec3 rad, vec3 normS) {
    uint rg = 0u;
    uint b_n = 0u;
    if (any(greaterThan(rad, vec3(1e-5)))) {
        packRadianceAndNorm(rad, normS, rg, b_n);
    }
    rawGISamples[screenPixels + pixelIndex] = uvec4(packedDir, floatBitsToUint(hitDist), rg, b_n);
}

// -----------------------------------------------------------------------------
// GI Target Function Evaluation
// p_hat = luminance( L_o(x_s) * f_r(x, w_i -> w_o) * cos(theta_x) )
// -----------------------------------------------------------------------------
float evalGITarget(vec3 radiance, vec3 albedo, vec3 hitN, vec3 sampleDir) {
    float nDotL = max(dot(hitN, sampleDir), 0.0);
    if (nDotL <= 0.0) return 0.0;
    float lum = dot(radiance * albedo, vec3(0.2126, 0.7152, 0.0722));
    return lum * (nDotL * INV_PI);
}

// -----------------------------------------------------------------------------
// Jacobian Determinant for Spatial & Temporal Sample Shifts
// Corrects for the change in solid angle when reusing secondary hit x_s from
// original receiver x_orig to new receiver x_new:
// J = (||x_s - x_orig||^2 / ||x_s - x_new||^2) * (cos(theta_s_new) / cos(theta_s_orig))
// -----------------------------------------------------------------------------
float evalGIJacobian(
    vec3 x_new,
    vec3 x_orig,
    vec3 w_orig,
    float dist_orig,
    vec3 n_s,
    out vec3 w_new,
    out float dist_new
) {
    // If secondary hit is environment / sky (infinite distance), direction is preserved, J = 1.0
    if (dist_orig >= 1e4) {
        w_new = w_orig;
        dist_new = dist_orig;
        return 1.0;
    }

    vec3 x_s = x_orig + w_orig * dist_orig;
    vec3 to_s_new = x_s - x_new;
    dist_new = length(to_s_new);
    if (dist_new < 1e-4) {
        w_new = w_orig;
        return 0.0;
    }
    float invDist = 1.0 / dist_new;
    w_new = to_s_new * invDist;

    float cos_s_orig = abs(dot(n_s, -w_orig));
    float cos_s_new  = abs(dot(n_s, -w_new));

    float distRatio = dist_orig * invDist;
    float J = (distRatio * distRatio) * (cos_s_new / max(cos_s_orig, 1e-3));

    // Clamp Jacobian to prevent extreme variance spikes near grazing or contact silhouettes
    return clamp(J, 0.05, 10.0);
}

#endif // WAVEFRONT_RESTIR_GI_GLSL
