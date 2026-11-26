#ifndef RGB9E5_GLSL
#define RGB9E5_GLSL

uint packRGB9E5(vec3 rgb) {
    vec3 v = mix(vec3(0.0), clamp(rgb, vec3(0.0), vec3(65408.0)), not(isnan(rgb)));
    float maxc = max(v.r, max(v.g, v.b));
    if (maxc <= 0.0) {
        return 0u;
    }
    int exp_shared = max(-16, int(floor(log2(max(maxc, 1e-30))))) + 16;
    float invDenom = exp2(float(24 - exp_shared));
    int maxm = int(floor(maxc * invDenom + 0.5));
    if (maxm >= 512) {
        exp_shared += 1;
        invDenom = exp2(float(24 - exp_shared));
    }
    uint rm = min(uint(floor(v.r * invDenom + 0.5)), 511u);
    uint gm = min(uint(floor(v.g * invDenom + 0.5)), 511u);
    uint bm = min(uint(floor(v.b * invDenom + 0.5)), 511u);
    if ((rm | gm | bm) == 0u) {
        return 0u;
    }
    return rm | (gm << 9u) | (bm << 18u) | (uint(clamp(exp_shared, 0, 31)) << 27u);
}

vec3 unpackRGB9E5(uint p) {
    int e = int((p >> 27u) & 31u) - 24;
    float scale = exp2(float(e));
    return vec3(float(p & 511u), float((p >> 9u) & 511u), float((p >> 18u) & 511u)) * scale;
}

#endif // RGB9E5_GLSL
