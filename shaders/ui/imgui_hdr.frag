#version 450 core

layout(location = 0) out vec4 fColor;
layout(set = 0, binding = 0) uniform texture2D _Texture;
layout(set = 1, binding = 0) uniform sampler _Sampler;
layout(location = 0) in struct { vec4 Color; vec2 UV; } In;

// 0 = SDR (sRGB nonlinear 8-bit UNORM swapchain)
// 1 = scRGB Linear (RGBA16F extended linear, 1.0 = 80 nits, Rec.709 primaries)
// 2 = HDR10 PQ (RGB10A2 UNORM, SMPTE ST.2084 PQ, Rec.2020 primaries)
layout(constant_id = 0) const uint kHdrMode = 0u;
layout(constant_id = 1) const float kPaperWhiteNits = 200.0;

vec3 srgbToLinear(vec3 srgb) {
    vec3 lo = srgb / 12.92;
    vec3 hi = pow((max(srgb, vec3(0.0)) + 0.055) / 1.055, vec3(2.4));
    return mix(lo, hi, greaterThan(srgb, vec3(0.04045)));
}

// ITU-R BT.2087 / BT.2407 Rec.709 -> Rec.2020 linear primary conversion (column-major)
const mat3 REC709_TO_REC2020 = mat3(
    0.6274040, 0.0690970, 0.0163916,
    0.3292820, 0.9195400, 0.0880132,
    0.0433136, 0.0113612, 0.8955950
);

vec3 linearToST2084(vec3 linNitsNormalized) {
    const float m1 = 0.1593017578125;
    const float m2 = 78.84375;
    const float c1 = 0.8359375;
    const float c2 = 18.8515625;
    const float c3 = 18.6875;
    vec3 ym = pow(clamp(linNitsNormalized, vec3(0.0), vec3(1.0)), vec3(m1));
    return pow((c1 + c2 * ym) / (1.0 + c3 * ym), vec3(m2));
}

void main() {
    vec4 srgbCol = In.Color * texture(sampler2D(_Texture, _Sampler), In.UV.st);
    if (kHdrMode == 0u) {
        fColor = srgbCol;
        return;
    }

    vec3 lin709 = srgbToLinear(clamp(srgbCol.rgb, vec3(0.0), vec3(1.0)));
    float paperWhite = clamp(kPaperWhiteNits, 80.0, 1000.0);

    if (kHdrMode == 1u) {
        fColor = vec4(lin709 * (paperWhite / 80.0), srgbCol.a);
        return;
    }

    vec3 lin2020 = max(REC709_TO_REC2020 * lin709, vec3(0.0));
    vec3 pq2020 = linearToST2084(lin2020 * (paperWhite / 10000.0));
    fColor = vec4(pq2020, srgbCol.a);
}
