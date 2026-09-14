#include "scene/GltfLoader.hpp"
#include "core/Logger.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <cassert>
#include <cstring>
#include <algorithm>

using namespace pathways;

namespace {

void assert_near(float a, float b, float eps = 0.005f, const char* msg = "") {
    if (std::abs(a - b) > eps) {
        std::cerr << "Assertion failed: " << a << " != " << b << " (eps: " << eps << ") " << msg << std::endl;
        std::exit(1);
    }
}

void check_true(bool cond, const char* msg = "") {
    if (!cond) {
        std::cerr << "Assertion failed: condition is false! " << msg << std::endl;
        std::exit(1);
    }
}

// Helper to base64 encode binary buffer for embedded glTF URI
std::string base64_encode(const uint8_t* data, size_t len) {
    static const char base64_chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "abcdefghijklmnopqrstuvwxyz"
        "0123456789+/";
    std::string ret;
    int i = 0;
    uint8_t char_array_3[3];
    uint8_t char_array_4[4];

    while (len--) {
        char_array_3[i++] = *(data++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for (i = 0; (i < 4); i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for (int j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

        for (int j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while ((i++ < 3))
            ret += '=';
    }
    return ret;
}

} // anonymous namespace

int main() {
    Logger::setLogLevel(LogLevel::Info);

    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: KHR_mesh_quantization Verification Suite      " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // -------------------------------------------------------------------------
    // Test 1: Quantized Signed/Unsigned Conversion Clamping & Spec Compliance
    // -------------------------------------------------------------------------
    std::cout << "[Step 1] Validating normalized integer conversion arithmetic..." << std::endl;
    {
        // Signed Short (component type 5122): -32768 to 32767
        // In glTF: normalized signed short must be clamped to max(c / 32767.0, -1.0)
        int16_t minShort = -32768;
        int16_t maxShort = 32767;
        int16_t zeroShort = 0;

        float fMinShort = std::clamp(minShort / 32767.0f, -1.0f, 1.0f);
        float fMaxShort = std::clamp(maxShort / 32767.0f, -1.0f, 1.0f);
        float fZeroShort = std::clamp(zeroShort / 32767.0f, -1.0f, 1.0f);

        check_true(fMinShort == -1.0f, "Signed short -32768 clamped to exactly -1.0f");
        check_true(fMaxShort == 1.0f, "Signed short 32767 converts to 1.0f");
        check_true(fZeroShort == 0.0f, "Signed short 0 converts to 0.0f");

        // Signed Byte (component type 5120): -128 to 127
        // In glTF: normalized signed byte must be clamped to max(c / 127.0, -1.0)
        int8_t minByte = -128;
        int8_t maxByte = 127;
        int8_t zeroByte = 0;

        float fMinByte = std::clamp(minByte / 127.0f, -1.0f, 1.0f);
        float fMaxByte = std::clamp(maxByte / 127.0f, -1.0f, 1.0f);
        float fZeroByte = std::clamp(zeroByte / 127.0f, -1.0f, 1.0f);

        check_true(fMinByte == -1.0f, "Signed byte -128 clamped to exactly -1.0f");
        check_true(fMaxByte == 1.0f, "Signed byte 127 converts to 1.0f");
        check_true(fZeroByte == 0.0f, "Signed byte 0 converts to 0.0f");

        // Unsigned Short (component type 5123): 0 to 65535
        uint16_t maxUshort = 65535;
        float fMaxUshort = maxUshort / 65535.0f;
        check_true(fMaxUshort == 1.0f, "Unsigned short 65535 converts to 1.0f");

        std::cout << "[PASS] Normalized arithmetic and boundary clamping verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // Test 2: In-Memory glTF 2.0 with KHR_mesh_quantization Extension
    // -------------------------------------------------------------------------
    std::cout << "[Step 2] Building synthetic KHR_mesh_quantization glTF asset..." << std::endl;
    std::filesystem::path tempGltfPath = std::filesystem::temp_directory_path() / "test_quantized_mesh.gltf";

    {
        // 3 vertices forming a triangle:
        // Vert 0: pos = (-1.0, -1.0, 0.0), norm = (0, 0, 1), tan = (1, 0, 0, 1), uv = (0.0, 0.0)
        // Vert 1: pos = ( 1.0, -1.0, 0.0), norm = (0, 0, 1), tan = (1, 0, 0, 1), uv = (1.0, 0.0)
        // Vert 2: pos = ( 0.0,  1.0, 0.0), norm = (0, 0, 1), tan = (1, 0, 0, 1), uv = (0.5, 1.0)
        
        // Stored with KHR_mesh_quantization:
        // POSITION: SHORT (normalized: true) -> -32768 maps to -1.0, 32767 maps to 1.0
        // NORMAL: BYTE (normalized: true, 4-byte aligned stride: 4) -> (0, 0, 127, 0)
        // TANGENT: SHORT (normalized: true, 8-byte aligned) -> (32767, 0, 0, 32767)
        // TEXCOORD_0: UNSIGNED_SHORT (normalized: true, 4-byte aligned) -> (0, 0), (65535, 0), (32768, 65535)
        // INDICES: UNSIGNED_SHORT -> 0, 1, 2

        struct RawVertexQuantized {
            // Position: 3 x int16_t (6 bytes) + 2 bytes padding = 8 bytes
            int16_t pos[4]; // x, y, z, pad
            // Normal: 3 x int8_t + 1 byte pad = 4 bytes
            int8_t norm[4]; // x, y, z, pad
            // Tangent: 4 x int16_t = 8 bytes
            int16_t tan[4]; // x, y, z, w
            // UV: 2 x uint16_t = 4 bytes
            uint16_t uv[2];  // u, v
        };

        RawVertexQuantized verts[3] = {};
        // Vert 0
        verts[0].pos[0] = -32768; verts[0].pos[1] = -32768; verts[0].pos[2] = 0; verts[0].pos[3] = 0;
        verts[0].norm[0] = 0; verts[0].norm[1] = 0; verts[0].norm[2] = 127; verts[0].norm[3] = 0;
        verts[0].tan[0] = 32767; verts[0].tan[1] = 0; verts[0].tan[2] = 0; verts[0].tan[3] = 32767;
        verts[0].uv[0] = 0; verts[0].uv[1] = 0;

        // Vert 1
        verts[1].pos[0] = 32767; verts[1].pos[1] = -32768; verts[1].pos[2] = 0; verts[1].pos[3] = 0;
        verts[1].norm[0] = 0; verts[1].norm[1] = 0; verts[1].norm[2] = 127; verts[1].norm[3] = 0;
        verts[1].tan[0] = 32767; verts[1].tan[1] = 0; verts[1].tan[2] = 0; verts[1].tan[3] = 32767;
        verts[1].uv[0] = 65535; verts[1].uv[1] = 0;

        // Vert 2
        verts[2].pos[0] = 0; verts[2].pos[1] = 32767; verts[2].pos[2] = 0; verts[2].pos[3] = 0;
        verts[2].norm[0] = 0; verts[2].norm[1] = 0; verts[2].norm[2] = 127; verts[2].norm[3] = 0;
        verts[2].tan[0] = 32767; verts[2].tan[1] = 0; verts[2].tan[2] = 0; verts[2].tan[3] = 32767;
        verts[2].uv[0] = 32768; verts[2].uv[1] = 65535;

        uint16_t indices[3] = {0, 1, 2};

        // Assemble binary buffer: [3 * sizeof(RawVertexQuantized)] followed by [3 * sizeof(uint16_t) + 2 bytes pad]
        std::vector<uint8_t> binBuffer;
        size_t vertSize = sizeof(verts);
        binBuffer.resize(vertSize + sizeof(indices) + 2, 0);
        std::memcpy(binBuffer.data(), verts, vertSize);
        std::memcpy(binBuffer.data() + vertSize, indices, sizeof(indices));

        std::string base64Uri = "data:application/octet-stream;base64," + base64_encode(binBuffer.data(), binBuffer.size());

        // Construct glTF JSON with KHR_mesh_quantization in extensionsRequired
        std::string gltfJson = R"({
  "asset": {
    "version": "2.0",
    "generator": "Pathways_Quantization_Test"
  },
  "extensionsUsed": [
    "KHR_mesh_quantization"
  ],
  "extensionsRequired": [
    "KHR_mesh_quantization"
  ],
  "buffers": [
    {
      "byteLength": )" + std::to_string(binBuffer.size()) + R"(,
      "uri": ")" + base64Uri + R"("
    }
  ],
  "bufferViews": [
    {
      "buffer": 0,
      "byteOffset": 0,
      "byteLength": )" + std::to_string(vertSize) + R"(,
      "byteStride": )" + std::to_string(sizeof(RawVertexQuantized)) + R"(,
      "target": 34962
    },
    {
      "buffer": 0,
      "byteOffset": )" + std::to_string(vertSize) + R"(,
      "byteLength": 6,
      "target": 34963
    }
  ],
  "accessors": [
    {
      "bufferView": 0,
      "byteOffset": 0,
      "componentType": 5122,
      "normalized": true,
      "count": 3,
      "type": "VEC3"
    },
    {
      "bufferView": 0,
      "byteOffset": 8,
      "componentType": 5120,
      "normalized": true,
      "count": 3,
      "type": "VEC3"
    },
    {
      "bufferView": 0,
      "byteOffset": 12,
      "componentType": 5122,
      "normalized": true,
      "count": 3,
      "type": "VEC4"
    },
    {
      "bufferView": 0,
      "byteOffset": 20,
      "componentType": 5123,
      "normalized": true,
      "count": 3,
      "type": "VEC2"
    },
    {
      "bufferView": 1,
      "byteOffset": 0,
      "componentType": 5123,
      "count": 3,
      "type": "SCALAR"
    }
  ],
  "meshes": [
    {
      "name": "QuantizedTriangleMesh",
      "primitives": [
        {
          "attributes": {
            "POSITION": 0,
            "NORMAL": 1,
            "TANGENT": 2,
            "TEXCOORD_0": 3
          },
          "indices": 4,
          "mode": 4
        }
      ]
    }
  ],
  "nodes": [
    {
      "mesh": 0,
      "name": "RootNode"
    }
  ],
  "scenes": [
    {
      "nodes": [0]
    }
  ],
  "scene": 0
})";

        std::ofstream out(tempGltfPath);
        out << gltfJson;
        out.close();
    }

    std::cout << "[Step 3] Loading synthetic glTF asset via GltfLoader..." << std::endl;
    GltfScene scene;
    bool loadSuccess = GltfLoader::load(tempGltfPath.string(), scene);
    check_true(loadSuccess, "GltfLoader must successfully load glTF with KHR_mesh_quantization");

    // -------------------------------------------------------------------------
    // Test 3: Extension Detection
    // -------------------------------------------------------------------------
    std::cout << "[Step 4] Verifying KHR_mesh_quantization extension registration..." << std::endl;
    check_true(scene.hasMeshQuantization, "scene.hasMeshQuantization must be true");
    check_true(std::find(scene.extensionsRequired.begin(), scene.extensionsRequired.end(),
                         "KHR_mesh_quantization") != scene.extensionsRequired.end(),
               "KHR_mesh_quantization must be in extensionsRequired");
    check_true(scene.meshes.size() == 1, "Must contain 1 mesh");
    check_true(scene.meshes[0].primitives.size() == 1, "Must contain 1 primitive");

    // -------------------------------------------------------------------------
    // Test 4: Decoding & Clamping Verification
    // -------------------------------------------------------------------------
    std::cout << "[Step 5] Verifying decoded vertex attributes..." << std::endl;
    const auto& prim = scene.meshes[0].primitives[0];
    check_true(prim.vertices.size() == 3, "Primitive must have 3 vertices");
    check_true(prim.indices.size() == 3, "Primitive must have 3 indices");
    check_true(prim.indices[0] == 0 && prim.indices[1] == 1 && prim.indices[2] == 2, "Indices must be 0, 1, 2");

    // Vert 0 checks:
    // Pos: clamped from -32768 -> exactly -1.0f
    assert_near(prim.vertices[0].position.x, -1.0f, 0.001f, "Vert 0 position.x clamped to -1.0f");
    assert_near(prim.vertices[0].position.y, -1.0f, 0.001f, "Vert 0 position.y clamped to -1.0f");
    assert_near(prim.vertices[0].position.z,  0.0f, 0.001f, "Vert 0 position.z is 0.0f");
    assert_near(prim.vertices[0].position.w,  0.0f, 0.001f, "Vert 0 UV.u is 0.0f");

    // Normal: (0, 0, 1) normalized
    assert_near(prim.vertices[0].normal.x, 0.0f, 0.001f, "Vert 0 normal.x is 0.0f");
    assert_near(prim.vertices[0].normal.y, 0.0f, 0.001f, "Vert 0 normal.y is 0.0f");
    assert_near(prim.vertices[0].normal.z, 1.0f, 0.001f, "Vert 0 normal.z is 1.0f");
    assert_near(prim.vertices[0].normal.w, 0.0f, 0.001f, "Vert 0 UV.v is 0.0f");

    // Tangent: (1, 0, 0, 1)
    assert_near(prim.vertices[0].tangent.x, 1.0f, 0.001f, "Vert 0 tangent.x is 1.0f");
    assert_near(prim.vertices[0].tangent.y, 0.0f, 0.001f, "Vert 0 tangent.y is 0.0f");
    assert_near(prim.vertices[0].tangent.z, 0.0f, 0.001f, "Vert 0 tangent.z is 0.0f");
    assert_near(prim.vertices[0].tangent.w, 1.0f, 0.001f, "Vert 0 tangent.w (sign) is 1.0f");

    // Vert 1 checks:
    assert_near(prim.vertices[1].position.x,  1.0f, 0.001f, "Vert 1 position.x is 1.0f");
    assert_near(prim.vertices[1].position.y, -1.0f, 0.001f, "Vert 1 position.y is -1.0f");
    assert_near(prim.vertices[1].position.w,  1.0f, 0.001f, "Vert 1 UV.u is 1.0f");
    assert_near(prim.vertices[1].normal.w,    0.0f, 0.001f, "Vert 1 UV.v is 0.0f");

    // Vert 2 checks:
    assert_near(prim.vertices[2].position.x, 0.0f, 0.001f, "Vert 2 position.x is 0.0f");
    assert_near(prim.vertices[2].position.y, 1.0f, 0.001f, "Vert 2 position.y is 1.0f");
    assert_near(prim.vertices[2].position.w, 0.5f, 0.002f, "Vert 2 UV.u is ~0.5f");
    assert_near(prim.vertices[2].normal.w,   1.0f, 0.001f, "Vert 2 UV.v is 1.0f");

    // Test SceneData pipeline end-to-end
    std::cout << "[Step 6] Testing SceneData ingestion from quantized asset..." << std::endl;
    SceneData sceneData = GltfLoader::loadSceneData(tempGltfPath.string());
    check_true(sceneData.triangles.size() == 1, "SceneData must have 1 triangle");
    const auto& tri = sceneData.triangles[0];
    assert_near(tri.v0.position.x, -1.0f, 0.001f);
    assert_near(tri.v1.position.x,  1.0f, 0.001f);
    assert_near(tri.v2.position.y,  1.0f, 0.001f);

    // Clean up temporary glTF file
    std::error_code ec;
    std::filesystem::remove(tempGltfPath, ec);

    // -------------------------------------------------------------------------
    // Test 5: Verify Standard glTF Asset Regression (DamagedHelmet.glb)
    // -------------------------------------------------------------------------
    std::cout << "[Step 7] Validating standard asset regression (DamagedHelmet.glb)..." << std::endl;
    std::filesystem::path helmetPath = "scenes/DamagedHelmet.glb";
    if (std::filesystem::exists(helmetPath)) {
        GltfScene helmetScene;
        bool helmetLoaded = GltfLoader::load(helmetPath.string(), helmetScene);
        check_true(helmetLoaded, "DamagedHelmet.glb must load successfully");
        check_true(!helmetScene.meshes.empty(), "DamagedHelmet must have meshes");
        check_true(!helmetScene.meshes[0].primitives.empty(), "DamagedHelmet must have primitives");
        check_true(!helmetScene.meshes[0].primitives[0].vertices.empty(), "DamagedHelmet vertices unpacked");
        std::cout << "  Unpacked " << helmetScene.meshes[0].primitives[0].vertices.size()
                  << " vertices, " << helmetScene.meshes[0].primitives[0].indices.size()
                  << " indices." << std::endl;
        check_true(!helmetScene.hasMeshQuantization, "DamagedHelmet does not use KHR_mesh_quantization");
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL KHR_mesh_quantization TESTS PASSED (100% SUCCESS)   " << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
