#ifndef WAVEFRONT_COMMON_GLSL
#define WAVEFRONT_COMMON_GLSL

#extension GL_EXT_ray_query : enable
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_ballot : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable

#define PI 3.14159265358979323846
#define TWO_PI 6.28318530717958647692
#define INV_PI 0.31830988618379067154
#define EPSILON 0.0005

struct Vertex {
    vec4 position; // xyz: pos, w: u
    vec4 normal;   // xyz: norm, w: v
};

struct Triangle {
    Vertex v0;
    Vertex v1;
    Vertex v2;
    uint materialId;
    uint padding[3];
};

struct Sphere {
    vec4 centerRadius; // xyz: center, w: radius
    uint materialId;
    uint padding[3];
};

struct Material {
    vec4 albedo;
    vec4 emissive;
    float roughness;
    float metallic;
    float ior;
    float transmission;
    uint type; // 0: diffuse, 1: metallic, 2: dielectric, 3: emissive
    uint albedoTex;
    uint normalTex;
    uint padding;
};

struct Light {
    vec4 position; // xyz: pos/corner, w: type (0: area, 1: spot)
    vec4 emission; // rgb: color, w: area
    vec4 u;        // xyz: edge1, w: spot inner cos
    vec4 v;        // xyz: edge2, w: spot outer cos
    vec4 normal;   // xyz: normal/dir, w: padding
};

// 96-byte cache-line friendly ray payload
struct RayPayload {
    vec4 origin;     // xyz: origin, w: uintBitsToFloat(flags: active=1)
    vec4 direction;  // xyz: direction, w: uintBitsToFloat(bounce)
    vec4 throughput; // rgb: throughput, w: uintBitsToFloat(seed)
    vec4 radiance;   // rgb: accumulated radiance, w: uintBitsToFloat(pixelIndex)
    vec4 hitNormal;  // xyz: shading normal, w: uintBitsToFloat(materialId)
    vec4 hitExtra;   // xy: uv, z: hit t, w: uintBitsToFloat(frontFace | hitType)
};

struct VkDispatchIndirectCommand {
    uint x;
    uint y;
    uint z;
};

// PCG Random Number Generator
uint pcg_hash(inout uint seed) {
    seed = seed * 747796405u + 2891336453u;
    uint word = ((seed >> ((seed >> 28u) + 4u)) ^ seed) * 277803737u;
    return (word >> 22u) ^ word;
}

float randFloat(inout uint seed) {
    return float(pcg_hash(seed)) / 4294967296.0;
}

vec2 randVec2(inout uint seed) {
    return vec2(randFloat(seed), randFloat(seed));
}

vec3 randVec3(inout uint seed) {
    return vec3(randFloat(seed), randFloat(seed), randFloat(seed));
}

vec2 directionToEquirectangular(vec3 dir) {
    float phi = atan(dir.z, dir.x);
    float theta = acos(clamp(dir.y, -1.0, 1.0));
    return vec2((phi + PI) / TWO_PI, theta / PI);
}

// 2D Morton Z-curve mapping for an 8x4 Wave32 block
ivec2 getPixelCoordsMorton8x4(uint linearIdx, uint width, uint height) {
    uint tilesX = (width + 7u) / 8u;
    uint tileIdx = linearIdx / 32u;
    uint inTile = linearIdx % 32u;
    uint tileX = tileIdx % tilesX;
    uint tileY = tileIdx / tilesX;

    uint mx = ((inTile >> 0u) & 1u) | (((inTile >> 2u) & 1u) << 1u) | (((inTile >> 4u) & 1u) << 2u);
    uint my = ((inTile >> 1u) & 1u) | (((inTile >> 3u) & 1u) << 1u);

    return ivec2(int(tileX * 8u + mx), int(tileY * 4u + my));
}

// Cosine-weighted hemisphere sampling
vec3 sampleCosineHemisphere(vec3 normal, inout uint seed) {
    vec2 r = randVec2(seed);
    float phi = TWO_PI * r.x;
    float cosTheta = sqrt(r.y);
    float sinTheta = sqrt(max(0.0, 1.0 - r.y));

    vec3 tangent = abs(normal.x) > 0.1 ? vec3(0, 1, 0) : vec3(1, 0, 0);
    tangent = normalize(cross(normal, tangent));
    vec3 bitangent = cross(normal, tangent);

    return normalize(tangent * (cos(phi) * sinTheta) +
                     bitangent * (sin(phi) * sinTheta) +
                     normal * cosTheta);
}

// Schlick's approximation for Fresnel reflectance
float fresnelSchlick(float cosTheta, float refIdx) {
    float r0 = (1.0 - refIdx) / (1.0 + refIdx);
    r0 = r0 * r0;
    return r0 + (1.0 - r0) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
}

// Procedural sphere ray intersection
bool intersectSphere(vec3 origin, vec3 dir, Sphere sphere, float tMin, float tMax, out float outT, out vec3 outNormal) {
    vec3 oc = origin - sphere.centerRadius.xyz;
    float radius = sphere.centerRadius.w;
    float a = dot(dir, dir);
    float halfB = dot(oc, dir);
    float c = dot(oc, oc) - radius * radius;
    float discriminant = halfB * halfB - a * c;

    if (discriminant < 0.0) return false;
    float sqrtd = sqrt(discriminant);

    float root = (-halfB - sqrtd) / a;
    if (root < tMin || root > tMax) {
        root = (-halfB + sqrtd) / a;
        if (root < tMin || root > tMax) return false;
    }

    outT = root;
    outNormal = (origin + root * dir - sphere.centerRadius.xyz) / radius;
    return true;
}

#endif // WAVEFRONT_COMMON_GLSL
