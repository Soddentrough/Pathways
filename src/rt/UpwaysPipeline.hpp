#pragma once

#include "core/Config.hpp"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pathways {

class Buffer;
class Image;
class UpwaysV5Engine;

inline constexpr uint32_t UPWAYS_FLAG_RESET_HISTORY      = 1u;
inline constexpr uint32_t UPWAYS_FLAG_ENABLE_SUPERRES    = 2u;
inline constexpr uint32_t UPWAYS_FLAG_ZERO_RESIDUAL_MODE = 4u;
inline constexpr uint32_t UPWAYS_FLAG_FUSED_BAND_PACK    = 8u;
inline constexpr uint32_t UPWAYS_FLAG_RUN_STABILIZE      = 16u;

inline constexpr uint32_t UPWAYS_DEFAULT_BAND_HALO_LR     = 24u;
inline constexpr float    UPWAYS_DEFAULT_BAND_TOP_FRACTION = 0.50f;

/**
 * Optional camera-motion payload for rotation-compensated prev-view normals
 * and affine prev-depth prediction in Upways 4.0 / 5.0 prep passes.
 * Default-constructed state has identity rotation and camMotionValid = 0.0f
 * so synthetic/offline callers and unit tests stay 100% bit-exact.
 */
struct UpwaysCameraMotion {
    glm::vec4 prevToCurrViewQuat{ 0.0f, 0.0f, 0.0f, 1.0f };  // xyzw unit quat rotating prev view -> curr view
    glm::vec4 currToPrevTransView{ 0.0f, 0.0f, 0.0f, 0.0f }; // xyz = (prevView * inverse(currView))[3].xyz, w = tanHalfFovY
    glm::vec4 camMotionParams{ 1.0f, 0.0f, 0.0f, 0.0f };     // x = aspectRatio, y = camMotionValid (1.0 live), z = prevJitterX, w = prevJitterY (LR pixels)
};

inline UpwaysCameraMotion makeUpwaysCameraMotion(
    const glm::mat4& currView,
    const glm::mat4& prevView,
    float tanHalfFovY,
    float aspectRatio,
    glm::vec2 prevJitterPx = glm::vec2(0.0f)
) noexcept {
    UpwaysCameraMotion m{};
    if (!(tanHalfFovY > 1e-4f) || !(aspectRatio > 1e-4f) ||
        !std::isfinite(tanHalfFovY) || !std::isfinite(aspectRatio)) {
        return m;
    }
    const glm::mat3 prevToCurrRot = glm::mat3(currView) * glm::transpose(glm::mat3(prevView));
    const glm::quat q = glm::normalize(glm::quat_cast(prevToCurrRot));
    const glm::mat4 currToPrev = prevView * glm::inverse(currView);
    const glm::vec3 t = glm::vec3(currToPrev[3]);
    if (!std::isfinite(q.x) || !std::isfinite(q.y) || !std::isfinite(q.z) || !std::isfinite(q.w) ||
        !std::isfinite(t.x) || !std::isfinite(t.y) || !std::isfinite(t.z)) {
        return m;
    }
    const float pjX = std::isfinite(prevJitterPx.x) ? prevJitterPx.x : 0.0f;
    const float pjY = std::isfinite(prevJitterPx.y) ? prevJitterPx.y : 0.0f;
    m.prevToCurrViewQuat = glm::vec4(q.x, q.y, q.z, q.w);
    m.currToPrevTransView = glm::vec4(t, tanHalfFovY);
    m.camMotionParams = glm::vec4(aspectRatio, 1.0f, pjX, pjY);
    return m;
}

/**
 * Exact 128-byte push constant layout shared across all four Upways 4.0 compute shaders:
 *   upways_denoise_prep.comp, upways_denoiser_wmma.comp,
 *   upways_upscaler_wmma.comp, upways_upscale_resolve.comp.
 */
struct UpwaysPushConstants {
    uint32_t  lrWidth;
    uint32_t  lrHeight;
    uint32_t  hrWidth;
    uint32_t  hrHeight;
    float     jitterX;
    float     jitterY;
    uint32_t  frameIndex;
    uint32_t  flags;
    uint32_t  offX0;
    uint32_t  offHw0;
    uint32_t  offS0;
    uint32_t  offB1;
    uint32_t  offB2;
    float     exposureScale;
    uint32_t  bandLrY0;
    uint32_t  bandHrY0;
    uint32_t  fullLrHeight;
    uint32_t  fullHrHeight;
    uint32_t  keepHrYStart;
    uint32_t  keepHrYEnd;
    glm::vec4 prevToCurrViewQuat{ 0.0f, 0.0f, 0.0f, 1.0f };
    glm::vec4 currToPrevTransView{ 0.0f, 0.0f, 0.0f, 0.0f };
    glm::vec4 camMotionParams{ 1.0f, 0.0f, 0.0f, 0.0f };
};

static_assert(sizeof(UpwaysPushConstants) == 128, "UpwaysPushConstants must be exactly 128 bytes");

struct UpwaysBandSplit {
    uint32_t fullLrH = 0u;
    uint32_t fullHrH = 0u;
    uint32_t haloLr = UPWAYS_DEFAULT_BAND_HALO_LR;
    uint32_t splitLrY = 0u;
    uint32_t topLrH = 0u;
    uint32_t botLrY0 = 0u;
    uint32_t botLrH = 0u;
    uint32_t splitHrY = 0u;
    uint32_t topHrH = 0u;
    uint32_t botHrY0 = 0u;
    uint32_t botHrH = 0u;
};

struct UpwaysBandParams {
    uint32_t bandLrY0 = 0u;
    uint32_t bandHrY0 = 0u;
    uint32_t fullLrHeight = 0u;
    uint32_t fullHrHeight = 0u;
    uint32_t keepHrYStart = 0u;
    uint32_t keepHrYEnd = 0u;
    bool fusedBandPack = false;
    uint32_t bandPackSlot = 0u;
};

inline UpwaysBandSplit computeUpwaysBandSplit(
    uint32_t fullLrH,
    uint32_t fullHrH,
    uint32_t haloLr = UPWAYS_DEFAULT_BAND_HALO_LR,
    float topFraction = UPWAYS_DEFAULT_BAND_TOP_FRACTION
) noexcept {
    UpwaysBandSplit s{};
    s.fullLrH = std::max(1u, fullLrH);
    s.fullHrH = std::max(1u, fullHrH);
    const uint32_t effHalo = (haloLr > 0u) ? std::clamp((haloLr + 2u) & ~3u, 8u, 64u) : 0u;
    if (s.fullLrH <= std::max(effHalo * 4u, 32u)) {
        s.haloLr = 0u;
        s.splitLrY = s.fullLrH / 2u;
        s.topLrH = s.fullLrH;
        s.botLrY0 = 0u;
        s.botLrH = s.fullLrH;
        s.splitHrY = (s.splitLrY * s.fullHrH) / s.fullLrH;
        s.topHrH = s.fullHrH;
        s.botHrY0 = 0u;
        s.botHrH = s.fullHrH;
        return s;
    }
    s.haloLr = effHalo;
    const double frac = std::clamp(static_cast<double>(topFraction), 0.25, 0.75);
    const uint32_t targetTop = static_cast<uint32_t>(std::llround(static_cast<double>(s.fullLrH) * frac));
    const uint32_t alignedHalf = (targetTop + 8u) & ~15u;
    s.splitLrY = std::clamp(alignedHalf, effHalo + 16u, s.fullLrH - (effHalo + 16u));
    s.topLrH = std::min(s.fullLrH, s.splitLrY + effHalo);
    s.botLrY0 = (s.splitLrY > effHalo) ? (s.splitLrY - effHalo) : 0u;
    s.botLrH = s.fullLrH - s.botLrY0;

    s.splitHrY = std::clamp((s.splitLrY * s.fullHrH + s.fullLrH / 2u) / s.fullLrH, 1u, s.fullHrH - 1u);
    s.topHrH = std::clamp((s.topLrH * s.fullHrH) / s.fullLrH, s.splitHrY, s.fullHrH);
    s.botHrY0 = std::min((s.botLrY0 * s.fullHrH + s.fullLrH - 1u) / s.fullLrH, s.splitHrY);
    s.botHrH = s.fullHrH - s.botHrY0;
    return s;
}

struct UpwaysPyramidDims {
    uint32_t     w0 = 1u, h0 = 1u;
    uint32_t     w1 = 1u, h1 = 1u;
    uint32_t     w2 = 1u, h2 = 1u;
    uint32_t     offX0 = 0u;
    uint32_t     offHw0 = 0u;
    uint32_t     offS0 = 0u;
    uint32_t     offB1 = 0u;
    uint32_t     offB2 = 0u;
    uint32_t     totalUints = 64u;
    VkDeviceSize totalBytes = 256ULL;
};

inline constexpr uint32_t upwaysAlign64Uints(uint32_t v) noexcept {
    return (v + 63u) & ~63u;
}

inline constexpr UpwaysPyramidDims computeUpwaysPyramidDims(uint32_t w0In, uint32_t h0In) noexcept {
    UpwaysPyramidDims d{};
    d.w0 = std::max(1u, w0In);
    d.h0 = std::max(1u, h0In);
    d.w1 = (d.w0 + 1u) >> 1u;
    d.h1 = (d.h0 + 1u) >> 1u;
    d.w2 = (d.w1 + 1u) >> 1u;
    d.h2 = (d.h1 + 1u) >> 1u;

    const uint32_t n0 = d.w0 * d.h0;
    const uint32_t n1 = d.w1 * d.h1;
    const uint32_t n2 = d.w2 * d.h2;

    d.offX0 = 0u;
    d.offHw0 = upwaysAlign64Uints(n1 * 24u);
    d.offS0 = d.offHw0 + upwaysAlign64Uints(std::max(n1 * 8u, n0 * 4u));
    d.offB1 = 0u;
    d.offB2 = d.offS0 + upwaysAlign64Uints(n1 * 8u);
    d.totalUints = d.offB2 + upwaysAlign64Uints(n2 * 8u);
    d.totalBytes = std::max<VkDeviceSize>(static_cast<VkDeviceSize>(d.totalUints) * 4ULL, 256ULL);
    return d;
}

// CPU-side parity math matching upways_common.glsl (safeLog1p / safeExpm1 / demod / remod)
inline float upwaysSafeLog1p(float x) noexcept {
    return std::log1p(std::clamp(x, 0.0f, 65504.0f));
}

inline glm::vec3 upwaysSafeLog1p(const glm::vec3& v) noexcept {
    return { upwaysSafeLog1p(v.x), upwaysSafeLog1p(v.y), upwaysSafeLog1p(v.z) };
}

inline float upwaysSafeExpm1(float x) noexcept {
    return std::max(std::expm1(std::clamp(x, 0.0f, 11.0898f)), 0.0f);
}

inline glm::vec3 upwaysSafeExpm1(const glm::vec3& v) noexcept {
    return { upwaysSafeExpm1(v.x), upwaysSafeExpm1(v.y), upwaysSafeExpm1(v.z) };
}

inline glm::vec3 upwaysDemodulate(const glm::vec3& radiance, const glm::vec3& albedo, float floorEps = 0.01f) noexcept {
    return radiance / glm::max(albedo, glm::vec3(floorEps));
}

inline glm::vec3 upwaysRemodulate(
    const glm::vec3& diffAlbedo,
    const glm::vec3& specAlbedo,
    const glm::vec3& diffIrr,
    const glm::vec3& specIrr,
    const glm::vec3& skipRgb = glm::vec3(0.0f)
) noexcept {
    return diffAlbedo * diffIrr + specAlbedo * specIrr + skipRgb;
}

struct UpwaysPassTimings {
    double prepMs = 0.0;      // Pass 1: upways_denoise_prep.comp
    double denoiserMs = 0.0;  // Pass 2: upways_denoiser_wmma.comp (Stages 0..3)
    double upscalerMs = 0.0;  // Pass 3: upways_upscaler_wmma.comp
    double resolveMs = 0.0;   // Pass 4: upways_upscale_resolve.comp
    double totalMs = 0.0;
    bool   valid = false;
};

enum class UpwaysComputePrecision : uint8_t {
    Fp8  = 0,
    Int8 = 1,
    Fp16 = 2
};

enum class UpwaysRuntimeVersion : uint8_t {
    V4 = 4,
    V5 = 5
};

inline constexpr const char* upwaysVersionName(UpwaysRuntimeVersion version) noexcept {
    return (version == UpwaysRuntimeVersion::V5) ? "5.0" : "4.0";
}

inline constexpr const char* upwaysVersionDisplayName(UpwaysRuntimeVersion version) noexcept {
    return (version == UpwaysRuntimeVersion::V5) ? "Upways 5.0 Pure-ML" : "Upways 4.0";
}

inline constexpr const char* upwaysPrecisionName(UpwaysComputePrecision precision) noexcept {
    switch (precision) {
        case UpwaysComputePrecision::Fp8:  return "fp8";
        case UpwaysComputePrecision::Int8: return "int8";
        case UpwaysComputePrecision::Fp16: return "fp16";
    }
    return "fp8";
}

inline constexpr const char* upwaysPrecisionDisplayName(UpwaysComputePrecision precision) noexcept {
    switch (precision) {
        case UpwaysComputePrecision::Fp8:  return "FP8 E4M3fn";
        case UpwaysComputePrecision::Int8: return "INT8 QAT";
        case UpwaysComputePrecision::Fp16: return "FP16";
    }
    return "FP8 E4M3fn";
}

/**
 * Upways 4.0 Wave32 Cooperative Matrix WMMA (FP8 E4M3fn / INT8 QAT / FP16) Two-Module Spatiotemporal Denoiser & SR Pipeline.
 *
 * Module A (UpwaysDenoiser):
 *   Pass 1: upways_denoise_prep.comp    -> temporal reprojection, disocclusion, X0 packing
 *   Pass 2: upways_denoiser_wmma.comp   -> 4-stage L1/L2 WMMA hourglass + clean diffuse/specular/radiance
 * Module B (UpwaysUpscaler):
 *   Pass 3: upways_upscaler_wmma.comp   -> 16 -> 32 -> 32 -> 16 subpixel delta & blend logit prediction
 *   Pass 4: upways_upscale_resolve.comp -> bicubic LR base + neural residual + HR temporal TAAU
 */
class UpwaysPipeline {
public:
    static constexpr uint32_t MAX_QUERY_SLOTS    = 2;
    static constexpr uint32_t QUERIES_PER_SLOT   = 5;
    static constexpr uint32_t NUM_DENOISER_STAGES = 4;
    static constexpr uint32_t BASE_ANCHOR_LAYERS = 4;

    static bool supportsFp8CoopMatrix(VkPhysicalDevice physicalDevice, VkInstance instance = VK_NULL_HANDLE);
    static bool supportsInt8CoopMatrix(VkPhysicalDevice physicalDevice, VkInstance instance = VK_NULL_HANDLE);
    static bool supportsFp16CoopMatrix(VkPhysicalDevice physicalDevice, VkInstance instance = VK_NULL_HANDLE);
    static UpwaysRuntimeVersion resolveVersion(
        UpwaysVersion requested,
        const std::string& weightsPath = ""
    );
    static UpwaysComputePrecision selectPrecision(
        VkPhysicalDevice physicalDevice,
        UpwaysPrecision requested = UpwaysPrecision::Auto,
        VkInstance instance = VK_NULL_HANDLE,
        UpwaysRuntimeVersion version = UpwaysRuntimeVersion::V4
    );

    // Upways 4.0 (24,064 B compact U-Net + KPN, 4 compute shaders)
    UpwaysPipeline(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VmaAllocator allocator,
        uint32_t inputWidth,
        uint32_t inputHeight,
        uint32_t outputWidth,
        uint32_t outputHeight,
        const std::vector<char>& denoisePrepSpv,
        const std::vector<char>& denoiserWmmaSpv,
        const std::vector<char>& upscalerWmmaSpv,
        const std::vector<char>& upscaleResolveSpv,
        const std::string& weightsPath = "",
        bool enableSuperRes = false,
        VkFormat imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT,
        UpwaysComputePrecision precision = UpwaysComputePrecision::Fp8,
        uint32_t outputFullHeight = 0u
    );

    // Upways 5.0 Pure-ML (158,400 params / 190,208 B FP8 & INT8, 5+1 compute shaders)
    UpwaysPipeline(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VmaAllocator allocator,
        uint32_t inputWidth,
        uint32_t inputHeight,
        uint32_t outputWidth,
        uint32_t outputHeight,
        const std::vector<char>& v5PrepSpv,
        const std::vector<char>& v5UnetWmmaSpv,
        const std::vector<char>& v5AffinitySpv,
        const std::vector<char>& v5UpscalerWmmaSpv,
        const std::vector<char>& v5ResolveSpv,
        const std::string& weightsPath = "",
        bool enableSuperRes = false,
        VkFormat imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT,
        UpwaysComputePrecision precision = UpwaysComputePrecision::Fp8,
        uint32_t outputFullHeight = 0u,
        const std::vector<char>& v5StabilizeSpv = {}
    );
    ~UpwaysPipeline();

    UpwaysPipeline(const UpwaysPipeline&) = delete;
    UpwaysPipeline& operator=(const UpwaysPipeline&) = delete;

    void transitionInitialLayouts(VkCommandBuffer cmd);

    void resize(
        uint32_t inputWidth,
        uint32_t inputHeight,
        uint32_t outputWidth,
        uint32_t outputHeight,
        bool enableSuperRes,
        uint32_t outputFullHeight = 0u
    );

    void updateDescriptors(
        VkImageView mlIrradianceView,
        VkImageView mlPrimaryGuideView,
        VkImageView mlSurfaceMvView,
        VkImageView mlDiffHitDistView = VK_NULL_HANDLE,
        VkImageView mlSpecVmvView = VK_NULL_HANDLE,
        VkImageView mlSkipCompactView = VK_NULL_HANDLE
    );

    void setBandPackTargets(
        VkBuffer slot0Buffer,
        VkDeviceSize slot0Size,
        VkBuffer slot1Buffer = VK_NULL_HANDLE,
        VkDeviceSize slot1Size = 0u
    );

    void recordFrame(
        VkCommandBuffer cmd,
        uint32_t frameIndex,
        bool resetHistory,
        glm::vec2 jitter,
        bool coupledWithExternalUpscaler = false,
        bool zeroResidualMode = false,
        float exposureScale = 1.0f,
        const UpwaysBandParams& band = {},
        const UpwaysCameraMotion& cameraMotion = {}
    );

    [[nodiscard]] Image* getOutputImage() const;
    [[nodiscard]] Image* getCleanRadLrImage() const;
    [[nodiscard]] uint32_t getInputWidth() const;
    [[nodiscard]] uint32_t getInputHeight() const;
    [[nodiscard]] uint32_t getOutputWidth() const;
    [[nodiscard]] uint32_t getOutputHeight() const;
    [[nodiscard]] uint32_t getOutputFullHeight() const;
    [[nodiscard]] bool isSuperResEnabled() const;
    [[nodiscard]] bool isHistoryValid() const;
    [[nodiscard]] UpwaysComputePrecision getPrecision() const;
    [[nodiscard]] UpwaysRuntimeVersion getVersion() const { return m_version; }
    [[nodiscard]] bool usesFp8CoopMatrix() const { return getPrecision() == UpwaysComputePrecision::Fp8; }
    [[nodiscard]] bool isInitialized() const;

    [[nodiscard]] const UpwaysPassTimings& getLastPassTimings() const;
    [[nodiscard]] std::array<float, 4> getPassTimingsMs() const {
        const UpwaysPassTimings& t = getLastPassTimings();
        return {
            static_cast<float>(t.prepMs),
            static_cast<float>(t.denoiserMs),
            static_cast<float>(t.upscalerMs),
            static_cast<float>(t.resolveMs)
        };
    }

private:
    struct ArrayImage2D {
        VkImage       image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImageView   view = VK_NULL_HANDLE;
        uint32_t      width = 0;
        uint32_t      height = 0;
        uint32_t      layers = 0;
    };

    void createArrayImage(ArrayImage2D& outImg, uint32_t width, uint32_t height, uint32_t layers);
    void destroyArrayImage(ArrayImage2D& img);

    void initWeights(const std::string& weightsPath);
    void initRenderResources(uint32_t slotMask);
    void initDisplayResources();
    void ensureActiveSlotSized();
    void refreshDescriptors();

    void createDescriptorSetLayouts();
    void allocateDescriptorSets();
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void createPipelines(
        const std::vector<char>& denoisePrepSpv,
        const std::vector<char>& denoiserWmmaSpv,
        const std::vector<char>& upscalerWmmaSpv,
        const std::vector<char>& upscaleResolveSpv
    );

    void collectQueryResults(uint32_t targetSlot, bool waitForResult);

    UpwaysRuntimeVersion   m_version = UpwaysRuntimeVersion::V4;
    std::unique_ptr<UpwaysV5Engine> m_v5Engine;

    VkDevice         m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VmaAllocator     m_allocator = VK_NULL_HANDLE;

    uint32_t         m_inputWidth = 0;
    uint32_t         m_inputHeight = 0;
    uint32_t         m_outputWidth = 0;
    uint32_t               m_outputHeight = 0;
    uint32_t               m_outputFullHeight = 0;
    bool                   m_enableSuperRes = false;
    UpwaysComputePrecision m_precision = UpwaysComputePrecision::Fp8;
    VkDeviceSize           m_weightQuantBytes = 15616u;
    VkFormat               m_imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    UpwaysPyramidDims m_pyramidDims{};

    bool             m_hasSubgroupSizeControl = false;
    bool             m_hasComputeFullSubgroups = false;

    uint32_t         m_pingPongIndex = 0;
    bool             m_historyValid = false;
    bool             m_initialLayoutsTransitioned = false;
    bool             m_hrTransitioned = false;
    bool             m_transientsTransitioned = false;
    std::array<bool, 2> m_lrSlotTransitioned = { false, false };
    std::array<uint32_t, 2> m_slotWidth = { 0u, 0u };
    std::array<uint32_t, 2> m_slotHeight = { 0u, 0u };

    // Static model weights + dynamic activation/subpixel SSBOs
    std::unique_ptr<Buffer> m_weightBuffer;
    std::unique_ptr<Buffer> m_actBuffer;
    std::unique_ptr<Buffer> m_srSubpixelBuffer;
    std::unique_ptr<Buffer> m_dummyBandPackBuffer;
    std::array<VkBuffer, 2>     m_bandPackBuf  = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDeviceSize, 2> m_bandPackSize = { 0u, 0u };
    std::unique_ptr<Image>  m_dummyBlackImage;
    std::unique_ptr<Image>  m_dummyR32uiImage;
    std::unique_ptr<Image>  m_dummyRg32uiImage;
    std::unique_ptr<Image>  m_dummyRgba32uiImage;
    std::unique_ptr<Image>  m_dummyR8Image;

    // Ping-pong LR history (indexed by [prev] / [curr])
    std::unique_ptr<Image> m_diffHist[2];    // RGBA16F
    std::unique_ptr<Image> m_specHist[2];    // RGBA16F
    std::unique_ptr<Image> m_geomHist[2];    // RGBA16F
    std::unique_ptr<Image> m_hiddenHist[2];  // R32_UINT

    // Single-frame LR intermediate targets
    ArrayImage2D           m_baseAnchorLr;   // 4-layer RGBA16F
    std::unique_ptr<Image> m_cleanRadLr;     // RGBA16F

    // Display-resolution HR ping-pong & final output
    std::unique_ptr<Image> m_colorHr[2];     // RGBA16F
    std::unique_ptr<Image> m_outputImage;    // RGBA16F

    // Cached compact live G-buffer views (6 attachments: 41 B/px)
    VkImageView m_cachedMlIrradianceView = VK_NULL_HANDLE;
    VkImageView m_cachedMlPrimaryGuideView = VK_NULL_HANDLE;
    VkImageView m_cachedMlSurfaceMvView = VK_NULL_HANDLE;
    VkImageView m_cachedMlDiffHitDistView = VK_NULL_HANDLE;
    VkImageView m_cachedMlSpecVmvView = VK_NULL_HANDLE;
    VkImageView m_cachedMlSkipCompactView = VK_NULL_HANDLE;

    // Descriptors & compute pipelines
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_prepDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_denoiserDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_upscalerDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_resolveDescLayout = VK_NULL_HANDLE;

    std::array<VkDescriptorSet, 2> m_prepDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<std::array<VkDescriptorSet, 2>, 2> m_denoiserDescSets = {{
        { VK_NULL_HANDLE, VK_NULL_HANDLE },
        { VK_NULL_HANDLE, VK_NULL_HANDLE }
    }};
    VkDescriptorSet m_upscalerDescSet = VK_NULL_HANDLE;
    std::array<std::array<VkDescriptorSet, 2>, 2> m_resolveDescSets = {{
        { VK_NULL_HANDLE, VK_NULL_HANDLE },
        { VK_NULL_HANDLE, VK_NULL_HANDLE }
    }};

    VkPipelineCache  m_pipelineCache = VK_NULL_HANDLE;
    VkPipelineLayout m_prepPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_denoiserPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_upscalerPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_resolvePipeLayout = VK_NULL_HANDLE;

    VkPipeline m_prepPipeline = VK_NULL_HANDLE;
    std::array<VkPipeline, NUM_DENOISER_STAGES> m_denoiserPipelines = {
        VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE
    };
    VkPipeline m_upscalerPipeline = VK_NULL_HANDLE;
    VkPipeline m_resolvePipeline = VK_NULL_HANDLE;

    // Hardware timestamp query pool
    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    float       m_timestampPeriodNs = 1.0f;
    uint32_t    m_querySlot = 0;
    std::array<bool, MAX_QUERY_SLOTS> m_querySlotSubmitted = { false, false };
    UpwaysPassTimings m_lastTimings{};
};

} // namespace pathways
