#ifndef PROCEDURAL_NOISE_GLSL
#define PROCEDURAL_NOISE_GLSL

// =============================================================================
// Pathways Procedural Noise & Terrain Shading Module
// Optimized for AMD RDNA 4 (gfx1201) Wave32 Execution
// Continuous 3D noise functions with analytical gradients and zero memory lookups
// =============================================================================

// Pure ALU 3D integer bit-permutation hash (zero memory / LDS lookups)
vec3 hash33(vec3 p) {
    uvec3 u = uvec3(floatBitsToInt(p));
    u = (u ^ (u.yzx >> 15u)) * 0x45d9f3bu;
    u = (u ^ (u.zxy >> 15u)) * 0x45d9f3bu;
    u = u ^ (u >> 16u);
    return vec3(u) * (2.0 / 4294967295.0) - 1.0;
}

// 3D Simplex noise with simultaneous analytical gradient
// Avoids costly finite difference approximations and saves ~15 VGPRs on RDNA 4
float simplex3D_grad(vec3 p, out vec3 grad) {
    const float F3 = 1.0 / 3.0;
    const float G3 = 1.0 / 6.0;

    // 1. Skew space to determine simplex origin
    vec3 s = floor(p + dot(p, vec3(F3)));
    vec3 x0 = p - s + dot(s, vec3(G3));

    // 2. Simplex traversal order (step-based branchless sort)
    vec3 e = step(x0.yzx, x0.xyz);
    vec3 i1 = e * (1.0 - e.zxy);
    vec3 i2 = 1.0 - e.zxy * (1.0 - e.xyz);

    vec3 x1 = x0 - i1 + G3;
    vec3 x2 = x0 - i2 + 2.0 * G3;
    vec3 x3 = x0 - 1.0 + 3.0 * G3;

    // 3. Radial distance kernels: w_i = max(0.6 - |x_i|^2, 0)
    vec4 w;
    w.x = max(0.6 - dot(x0, x0), 0.0);
    w.y = max(0.6 - dot(x1, x1), 0.0);
    w.z = max(0.6 - dot(x2, x2), 0.0);
    w.w = max(0.6 - dot(x3, x3), 0.0);

    vec4 w2 = w * w;
    vec4 w4 = w2 * w2;

    // 4. Hash pseudo-random gradient vectors at simplex vertices
    vec3 g0 = hash33(s);
    vec3 g1 = hash33(s + i1);
    vec3 g2 = hash33(s + i2);
    vec3 g3 = hash33(s + 1.0);

    vec4 gdotx = vec4(dot(g0, x0), dot(g1, x1), dot(g2, x2), dot(g3, x3));

    // 5. Analytical gradient: d(w^4 * (g.x))/dx = -8*x * w^3 * (g.x) + w^4 * g
    vec4 w3_gdotx = (w * w2) * gdotx * -8.0;
    grad = g0 * w4.x + g1 * w4.y + g2 * w4.z + g3 * w4.w +
           x0 * w3_gdotx.x + x1 * w3_gdotx.y + x2 * w3_gdotx.z + x3 * w3_gdotx.w;
    grad *= 32.0;

    return 32.0 * dot(w4, gdotx);
}

// Low-register 2D Voronoi (9-cell neighborhood) for cracks, rock fissures, and pebble patterns
void voronoi2D(vec2 p, out float f1, out float f2) {
    vec2 cell = floor(p);
    vec2 frac = fract(p);
    f1 = 8.0;
    f2 = 8.0;

    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            vec2 neighbor = vec2(float(i), float(j));
            vec2 pt = neighbor + 0.5 + 0.4 * hash33(vec3(cell + neighbor, 1.0)).xy;
            vec2 diff = pt - frac;
            float d = dot(diff, diff);
            if (d < f1) {
                f2 = f1;
                f1 = d;
            } else if (d < f2) {
                f2 = d;
            }
        }
    }
    f1 = sqrt(f1);
    f2 = sqrt(f2);
}

float voronoi2D(vec2 p) {
    float f1, f2;
    voronoi2D(p, f1, f2);
    return f1;
}

// Multi-scale Fractal Brownian Motion (fBM) with fixed orthogonal 3D domain rotation
// Breaks axis alignment and eliminates visible grid artifacts using minimal octaves
float fbm3D_grad(vec3 p, int octaves, out vec3 outGrad) {
    float value = 0.0;
    float amplitude = 0.5;
    float frequency = 1.0;
    outGrad = vec3(0.0);

    const mat3 rot = mat3(
        0.00,  0.80,  0.60,
       -0.80,  0.36, -0.48,
       -0.60, -0.48,  0.64
    );
    const mat3 rotT = transpose(rot);

    vec3 currentP = p;
    mat3 curRotT = mat3(1.0); // Identity for octave 0 (unrotated domain)

    for (int i = 0; i < octaves; ++i) {
        vec3 g;
        float n = simplex3D_grad(currentP * frequency, g);
        value += amplitude * n;
        outGrad += (amplitude * frequency) * (curRotT * g);
        currentP = rot * currentP;
        curRotT = curRotT * rotT;
        frequency *= 2.0;
        amplitude *= 0.5;
    }
    return value;
}

// Directional water wave spectrum (Gerstner/sinusoidal superposition)
// Accumulates 6 directional wave components across deep swell, cross chop, and capillary ripples
vec3 evaluateWaterWaves(vec3 pos, float time, out vec3 normalOffset) {
    const vec2 D[6] = vec2[6](
        vec2(0.7071, 0.7071),
        vec2(-0.6000, 0.8000),
        vec2(0.9138, -0.4061),
        vec2(-0.3162, -0.9487),
        vec2(0.8660, 0.5000),
        vec2(-0.5000, -0.8660)
    );
    // (spatial frequency k, temporal angular frequency w, amplitude A)
    const vec3 waveP[6] = vec3[6](
        vec3(0.7854, 0.9425, 0.0750), // lambda=8.0m, c=1.2m/s, amp=0.075 (deep swell)
        vec3(1.2566, 1.8849, 0.0900), // lambda=5.0m, c=1.5m/s, amp=0.090 (deep swell)
        vec3(3.1416, 6.5973, 0.0600), // lambda=2.0m, c=2.1m/s, amp=0.060 (cross chop)
        vec3(5.2360, 14.660, 0.0375), // lambda=1.2m, c=2.8m/s, amp=0.0375 (cross chop)
        vec3(10.472, 33.510, 0.0150), // lambda=0.6m, capillary, amp=0.015 (fine specular glints)
        vec3(15.708, 55.000, 0.0075)  // lambda=0.4m, capillary, amp=0.0075 (fine specular glints)
    );

    vec2 dH = vec2(0.0);
    for (int w = 0; w < 6; ++w) {
        float ph = dot(D[w], pos.xz) * waveP[w].x + time * waveP[w].y;
        dH += D[w] * (waveP[w].x * waveP[w].z * cos(ph));
    }
    normalOffset = vec3(-dH.x, 0.0, -dH.y);
    return normalize(vec3(-dH.x, 1.0, -dH.y));
}

vec3 evalProceduralWaterNormal(vec2 xzPos, float time) {
    vec3 dummyOffset;
    return evaluateWaterWaves(vec3(xzPos.x, 0.0, xzPos.y), time, dummyOffset);
}

#endif // PROCEDURAL_NOISE_GLSL
