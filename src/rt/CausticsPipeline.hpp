#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <array>
#include <functional>

#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "vulkan/Texture.hpp"
#include "scene/Camera.hpp"

namespace pathways {

class CausticsPipeline {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr uint32_t MAX_SCENE_TEXTURES = 512;

    using ImmediateSubmitFn = std::function<void(const std::function<void(VkCommandBuffer)>&)>;

    CausticsPipeline(VkDevice device,
                     VmaAllocator allocator,
                     bool hasSubgroupSizeControl,
                     uint32_t width,
                     uint32_t height,
                     uint32_t photonCount,
                     const std::vector<char>& traceSpv,
                     const std::vector<char>& splatSpv,
                     const std::vector<char>& filterSpv,
                     ImmediateSubmitFn immediateSubmit = nullptr);
    ~CausticsPipeline();

    CausticsPipeline(const CausticsPipeline&) = delete;
    CausticsPipeline& operator=(const CausticsPipeline&) = delete;

    void resize(uint32_t width, uint32_t height, ImmediateSubmitFn immediateSubmit = nullptr);

    void updateDescriptors(
        Buffer* triangleBuffer,
        Buffer* sphereBuffer,
        Buffer* materialBuffer,
        Buffer* lightBuffer,
        VkAccelerationStructureKHR tlasHandle,
        const std::vector<std::unique_ptr<Texture>>& sceneTextures,
        Texture* dummyWhite,
        Buffer* instanceBuffer,
        const std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT>& cameraUBOs,
        Image* normalDepthImage,
        Image* motionVectorImage
    );

    void recordTrace(
        VkCommandBuffer cmd,
        uint32_t frameSlot,
        uint32_t numTriangles,
        uint32_t numSpheres,
        uint32_t numMaterials,
        uint32_t numLights,
        uint32_t numOpaqueTriangles,
        uint32_t frameIndex,
        bool hasDielectrics,
        const glm::vec3& dielectricBoundsMin,
        const glm::vec3& dielectricBoundsMax
    );

    void recordSplatAndFilter(
        VkCommandBuffer cmd,
        uint32_t frameSlot,
        Image* normalDepthImage,
        float cameraFov,
        uint32_t frameIndex,
        bool progressiveAccumulation,
        bool cameraMovedLastFrame,
        bool hasDielectrics,
        const glm::vec3& dielectricBoundsMin,
        const glm::vec3& dielectricBoundsMax
    );

    void recordFullPass(
        VkCommandBuffer cmd,
        uint32_t frameSlot,
        uint32_t numTriangles,
        uint32_t numSpheres,
        uint32_t numMaterials,
        uint32_t numLights,
        uint32_t numOpaqueTriangles,
        uint32_t frameIndex,
        Image* normalDepthImage,
        float cameraFov,
        bool progressiveAccumulation,
        bool cameraMovedLastFrame,
        bool hasDielectrics,
        const glm::vec3& dielectricBoundsMin,
        const glm::vec3& dielectricBoundsMax
    );

    Image* getFilteredCausticImage() const { return m_filteredCausticImage.get(); }
    Image* getPrevCausticImage() const { return m_prevCausticImage.get(); }
    Buffer* getPhotonBuffer() const { return m_causticPhotonBuffer.get(); }
    Buffer* getAtomicBuffer() const { return m_causticAtomicBuffer.get(); }

private:
    void createDescriptorLayouts();
    void allocateDescriptorSets();
    void createPipelines(const std::vector<char>& traceSpv,
                         const std::vector<char>& splatSpv,
                         const std::vector<char>& filterSpv);
    void createResources(ImmediateSubmitFn immediateSubmit);
    void destroyResources();
    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    bool m_hasSubgroupSizeControl = false;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_photonCount = 65536;

    // Buffers and Images
    std::unique_ptr<Buffer> m_causticPhotonBuffer;
    std::unique_ptr<Buffer> m_causticAtomicBuffer;
    std::unique_ptr<Image> m_filteredCausticImage;
    std::unique_ptr<Image> m_prevCausticImage;

    // Layouts and Pools
    VkDescriptorSetLayout m_traceDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_splatDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_filterDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_descPool = VK_NULL_HANDLE;

    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_traceDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_splatDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_filterDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    VkPipelineLayout m_tracePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_splatPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_filterPipelineLayout = VK_NULL_HANDLE;

    VkPipeline m_tracePipeline = VK_NULL_HANDLE;
    VkPipeline m_splatPipeline = VK_NULL_HANDLE;
    VkPipeline m_filterPipeline = VK_NULL_HANDLE;
};

} // namespace pathways
