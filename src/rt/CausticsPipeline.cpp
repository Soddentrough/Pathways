#include "CausticsPipeline.hpp"
#include "core/Logger.hpp"

#include <stdexcept>
#include <algorithm>

namespace pathways {

CausticsPipeline::CausticsPipeline(VkDevice device,
                                   VmaAllocator allocator,
                                   bool hasSubgroupSizeControl,
                                   uint32_t width,
                                   uint32_t height,
                                   uint32_t photonCount,
                                   const std::vector<char>& traceSpv,
                                   const std::vector<char>& splatSpv,
                                   const std::vector<char>& filterSpv,
                                   ImmediateSubmitFn immediateSubmit)
    : m_device(device),
      m_allocator(allocator),
      m_hasSubgroupSizeControl(hasSubgroupSizeControl),
      m_width(width),
      m_height(height),
      m_photonCount(photonCount)
{
    if (traceSpv.empty() || splatSpv.empty() || filterSpv.empty()) {
        throw std::runtime_error("CausticsPipeline: Empty SPIR-V bytecode passed to constructor");
    }

    createDescriptorLayouts();
    allocateDescriptorSets();
    createPipelines(traceSpv, splatSpv, filterSpv);
    createResources(immediateSubmit);

    Logger::info("CausticsPipeline initialized successfully ({}x{}, {} photons).",
                 m_width, m_height, m_photonCount);
}

CausticsPipeline::~CausticsPipeline() {
    destroyResources();

    if (m_tracePipeline)  { vkDestroyPipeline(m_device, m_tracePipeline, nullptr); m_tracePipeline = VK_NULL_HANDLE; }
    if (m_splatPipeline)  { vkDestroyPipeline(m_device, m_splatPipeline, nullptr); m_splatPipeline = VK_NULL_HANDLE; }
    if (m_filterPipeline) { vkDestroyPipeline(m_device, m_filterPipeline, nullptr); m_filterPipeline = VK_NULL_HANDLE; }

    if (m_tracePipelineLayout)  { vkDestroyPipelineLayout(m_device, m_tracePipelineLayout, nullptr); m_tracePipelineLayout = VK_NULL_HANDLE; }
    if (m_splatPipelineLayout)  { vkDestroyPipelineLayout(m_device, m_splatPipelineLayout, nullptr); m_splatPipelineLayout = VK_NULL_HANDLE; }
    if (m_filterPipelineLayout) { vkDestroyPipelineLayout(m_device, m_filterPipelineLayout, nullptr); m_filterPipelineLayout = VK_NULL_HANDLE; }

    if (m_descPool) { vkDestroyDescriptorPool(m_device, m_descPool, nullptr); m_descPool = VK_NULL_HANDLE; }

    if (m_traceDescLayout)  { vkDestroyDescriptorSetLayout(m_device, m_traceDescLayout, nullptr); m_traceDescLayout = VK_NULL_HANDLE; }
    if (m_splatDescLayout)  { vkDestroyDescriptorSetLayout(m_device, m_splatDescLayout, nullptr); m_splatDescLayout = VK_NULL_HANDLE; }
    if (m_filterDescLayout) { vkDestroyDescriptorSetLayout(m_device, m_filterDescLayout, nullptr); m_filterDescLayout = VK_NULL_HANDLE; }
}

VkShaderModule CausticsPipeline::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("CausticsPipeline: Failed to create shader module!");
    }
    return shaderModule;
}

void CausticsPipeline::createDescriptorLayouts() {
    // 1a. Trace Layout: Triangles(2), Spheres(3), Materials(4), Lights(5), TLAS(6), Textures(8), Photons(20), Instances(30)
    std::vector<VkDescriptorSetLayoutBinding> traceBindings = {
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_SCENE_TEXTURES, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 30, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo traceLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    traceLayoutInfo.bindingCount = static_cast<uint32_t>(traceBindings.size());
    traceLayoutInfo.pBindings = traceBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &traceLayoutInfo, nullptr, &m_traceDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic trace descriptor set layout");
    }

    // 1b. Splat Layout: CameraUBO(1), NormalDepth(12), Photons(20), AtomicBuffer(21)
    std::vector<VkDescriptorSetLayoutBinding> splatBindings = {
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 21, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo splatLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    splatLayoutInfo.bindingCount = static_cast<uint32_t>(splatBindings.size());
    splatLayoutInfo.pBindings = splatBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &splatLayoutInfo, nullptr, &m_splatDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic splat descriptor set layout");
    }

    // 1c. Filter Layout: CameraUBO(1), NormalDepth(12), MotionVector(14), AtomicBuffer(21), FilteredCaustic(22), PrevCaustic(23)
    std::vector<VkDescriptorSetLayoutBinding> filterBindings = {
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 21, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 22, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 23, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo filterLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    filterLayoutInfo.bindingCount = static_cast<uint32_t>(filterBindings.size());
    filterLayoutInfo.pBindings = filterBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &filterLayoutInfo, nullptr, &m_filterDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic filter descriptor set layout");
    }
}

void CausticsPipeline::allocateDescriptorSets() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 8 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2048 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32 }
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 16;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustics descriptor pool");
    }

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_descPool;
        allocInfo.descriptorSetCount = 1;

        allocInfo.pSetLayouts = &m_traceDescLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_traceDescSets[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate caustic trace descriptor set");
        }

        allocInfo.pSetLayouts = &m_splatDescLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_splatDescSets[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate caustic splat descriptor set");
        }

        allocInfo.pSetLayouts = &m_filterDescLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_filterDescSets[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate caustic filter descriptor set");
        }
    }
}

void CausticsPipeline::createPipelines(const std::vector<char>& traceSpv,
                                       const std::vector<char>& splatSpv,
                                       const std::vector<char>& filterSpv) {
    // 1. Trace Pipeline Layout
    VkPushConstantRange tracePCRange{};
    tracePCRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tracePCRange.offset = 0;
    tracePCRange.size = sizeof(uint32_t) * 8 + sizeof(float) * 8; // 64 bytes
    VkPipelineLayoutCreateInfo tracePipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    tracePipeLayoutInfo.setLayoutCount = 1;
    tracePipeLayoutInfo.pSetLayouts = &m_traceDescLayout;
    tracePipeLayoutInfo.pushConstantRangeCount = 1;
    tracePipeLayoutInfo.pPushConstantRanges = &tracePCRange;
    if (vkCreatePipelineLayout(m_device, &tracePipeLayoutInfo, nullptr, &m_tracePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic trace pipeline layout");
    }

    // 2. Splat Pipeline Layout
    VkPushConstantRange splatPCRange{};
    splatPCRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    splatPCRange.offset = 0;
    splatPCRange.size = sizeof(uint32_t) * 3 + sizeof(float) + sizeof(glm::vec4) * 2; // 48 bytes
    VkPipelineLayoutCreateInfo splatPipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    splatPipeLayoutInfo.setLayoutCount = 1;
    splatPipeLayoutInfo.pSetLayouts = &m_splatDescLayout;
    splatPipeLayoutInfo.pushConstantRangeCount = 1;
    splatPipeLayoutInfo.pPushConstantRanges = &splatPCRange;
    if (vkCreatePipelineLayout(m_device, &splatPipeLayoutInfo, nullptr, &m_splatPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic splat pipeline layout");
    }

    // 3. Filter Pipeline Layout
    VkPushConstantRange filterPCRange{};
    filterPCRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    filterPCRange.offset = 0;
    filterPCRange.size = sizeof(uint32_t) * 4; // 16 bytes
    VkPipelineLayoutCreateInfo filterPipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    filterPipeLayoutInfo.setLayoutCount = 1;
    filterPipeLayoutInfo.pSetLayouts = &m_filterDescLayout;
    filterPipeLayoutInfo.pushConstantRangeCount = 1;
    filterPipeLayoutInfo.pPushConstantRanges = &filterPCRange;
    if (vkCreatePipelineLayout(m_device, &filterPipeLayoutInfo, nullptr, &m_filterPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic filter pipeline layout");
    }

    // 4. Compute Pipelines (Wave32 optimized)
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
    };
    subgroupSize32.requiredSubgroupSize = 32;

    VkShaderModule traceMod = createShaderModule(traceSpv);
    VkComputePipelineCreateInfo tracePipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    tracePipeInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, traceMod, "main", nullptr };
    if (m_hasSubgroupSizeControl) tracePipeInfo.stage.pNext = &subgroupSize32;
    tracePipeInfo.layout = m_tracePipelineLayout;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &tracePipeInfo, nullptr, &m_tracePipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, traceMod, nullptr);
        throw std::runtime_error("Failed to create caustic trace compute pipeline");
    }
    vkDestroyShaderModule(m_device, traceMod, nullptr);

    VkShaderModule splatMod = createShaderModule(splatSpv);
    VkComputePipelineCreateInfo splatPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    splatPipeInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, splatMod, "main", nullptr };
    if (m_hasSubgroupSizeControl) splatPipeInfo.stage.pNext = &subgroupSize32;
    splatPipeInfo.layout = m_splatPipelineLayout;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &splatPipeInfo, nullptr, &m_splatPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, splatMod, nullptr);
        throw std::runtime_error("Failed to create caustic splat compute pipeline");
    }
    vkDestroyShaderModule(m_device, splatMod, nullptr);

    VkShaderModule filterMod = createShaderModule(filterSpv);
    VkComputePipelineCreateInfo filterPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    filterPipeInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, filterMod, "main", nullptr };
    if (m_hasSubgroupSizeControl) filterPipeInfo.stage.pNext = &subgroupSize32;
    filterPipeInfo.layout = m_filterPipelineLayout;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &filterPipeInfo, nullptr, &m_filterPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, filterMod, nullptr);
        throw std::runtime_error("Failed to create caustic filter compute pipeline");
    }
    vkDestroyShaderModule(m_device, filterMod, nullptr);
}

void CausticsPipeline::createResources(ImmediateSubmitFn immediateSubmit) {
    uint32_t photonCount = std::max(m_photonCount, 65536u);
    VkDeviceSize photonBufferSize = static_cast<VkDeviceSize>(photonCount) * 64;
    m_causticPhotonBuffer = std::make_unique<Buffer>(
        m_allocator, photonBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    VkDeviceSize atomicBufferSize = static_cast<VkDeviceSize>(m_width * m_height) * 3 * sizeof(uint32_t);
    m_causticAtomicBuffer = std::make_unique<Buffer>(
        m_allocator, atomicBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    m_filteredCausticImage = std::make_unique<Image>(
        m_device, m_allocator, m_width, m_height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_prevCausticImage = std::make_unique<Image>(
        m_device, m_allocator, m_width, m_height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    if (immediateSubmit) {
        immediateSubmit([this](VkCommandBuffer cmd) {
            m_filteredCausticImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                                     VK_PIPELINE_STAGE_2_NONE, 0,
                                                     VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_prevCausticImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                                 VK_PIPELINE_STAGE_2_NONE, 0,
                                                 VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            vkCmdFillBuffer(cmd, m_causticAtomicBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0);

            VkClearColorValue clearZero{};
            VkImageSubresourceRange sRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(cmd, m_filteredCausticImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &sRange);
            vkCmdClearColorImage(cmd, m_prevCausticImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &sRange);
        });
    }
}

void CausticsPipeline::destroyResources() {
    m_causticPhotonBuffer.reset();
    m_causticAtomicBuffer.reset();
    m_filteredCausticImage.reset();
    m_prevCausticImage.reset();
}

void CausticsPipeline::resize(uint32_t width, uint32_t height, ImmediateSubmitFn immediateSubmit) {
    if (m_width == width && m_height == height) return;
    m_width = width;
    m_height = height;
    destroyResources();
    createResources(immediateSubmit);
    Logger::info("CausticsPipeline resized to {}x{}.", m_width, m_height);
}

void CausticsPipeline::updateDescriptors(
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
    Image* motionVectorImage)
{
    if (!m_descPool || !m_causticPhotonBuffer || !m_causticAtomicBuffer ||
        !m_filteredCausticImage || !m_prevCausticImage || !normalDepthImage ||
        !motionVectorImage || !triangleBuffer) {
        return;
    }

    VkDescriptorBufferInfo triInfo{ triangleBuffer->getBuffer(), 0, triangleBuffer->getSize() };
    VkDescriptorBufferInfo sphereInfo{ sphereBuffer ? sphereBuffer->getBuffer() : triangleBuffer->getBuffer(), 0, sphereBuffer ? sphereBuffer->getSize() : triangleBuffer->getSize() };
    VkDescriptorBufferInfo matInfo{ materialBuffer ? materialBuffer->getBuffer() : triangleBuffer->getBuffer(), 0, materialBuffer ? materialBuffer->getSize() : triangleBuffer->getSize() };
    VkDescriptorBufferInfo lightInfo{ lightBuffer ? lightBuffer->getBuffer() : triangleBuffer->getBuffer(), 0, lightBuffer ? lightBuffer->getSize() : triangleBuffer->getSize() };

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &tlasHandle;

    std::vector<VkDescriptorImageInfo> texInfos(MAX_SCENE_TEXTURES);
    for (size_t i = 0; i < MAX_SCENE_TEXTURES; ++i) {
        if (i < sceneTextures.size() && sceneTextures[i]) {
            texInfos[i] = sceneTextures[i]->getDescriptorInfo();
        } else if (dummyWhite) {
            texInfos[i] = dummyWhite->getDescriptorInfo();
        }
    }

    VkDescriptorBufferInfo photonBufInfo{ m_causticPhotonBuffer->getBuffer(), 0, m_causticPhotonBuffer->getSize() };
    VkDescriptorBufferInfo atomicBufInfo{ m_causticAtomicBuffer->getBuffer(), 0, m_causticAtomicBuffer->getSize() };

    VkDescriptorImageInfo ndImageInfo{ VK_NULL_HANDLE, normalDepthImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo mvImageInfo{ VK_NULL_HANDLE, motionVectorImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo filteredCausticInfo{ VK_NULL_HANDLE, m_filteredCausticImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo prevCausticInfo{ VK_NULL_HANDLE, m_prevCausticImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkDescriptorBufferInfo, MAX_FRAMES_IN_FLIGHT> camInfos;
    std::array<VkDescriptorBufferInfo, MAX_FRAMES_IN_FLIGHT> instanceInfos;
    VkBuffer actualInstanceBuffer = (instanceBuffer && instanceBuffer->getBuffer() != VK_NULL_HANDLE) ? instanceBuffer->getBuffer() : triangleBuffer->getBuffer();
    VkDeviceSize actualInstanceSize = (instanceBuffer && instanceBuffer->getSize() > 0) ? instanceBuffer->getSize() : triangleBuffer->getSize();

    std::vector<VkWriteDescriptorSet> writes;

    for (uint32_t slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
        if (!cameraUBOs[slot]) continue;
        camInfos[slot] = { cameraUBOs[slot]->getBuffer(), 0, sizeof(CameraUniform) };
        instanceInfos[slot] = { actualInstanceBuffer, 0, actualInstanceSize };

        // 1. Caustic Trace Set
        VkDescriptorSet traceSet = m_traceDescSets[slot];
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sphereInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightInfo, nullptr });
        if (tlasHandle != VK_NULL_HANDLE) {
            writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, traceSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr });
        }
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 8, 0, MAX_SCENE_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texInfos.data(), nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 20, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &photonBufInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 30, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &instanceInfos[slot], nullptr });

        // 2. Caustic Splat Set
        VkDescriptorSet splatSet = m_splatDescSets[slot];
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, splatSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &camInfos[slot], nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, splatSet, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ndImageInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, splatSet, 20, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &photonBufInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, splatSet, 21, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &atomicBufInfo, nullptr });

        // 3. Caustic Filter Set
        VkDescriptorSet filterSet = m_filterDescSets[slot];
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &camInfos[slot], nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ndImageInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mvImageInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 21, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &atomicBufInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 22, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &filteredCausticInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 23, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &prevCausticInfo, nullptr, nullptr });
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void CausticsPipeline::recordTrace(
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
    const glm::vec3& dielectricBoundsMax)
{
    if (!m_tracePipeline || !m_causticPhotonBuffer || numLights == 0) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tracePipelineLayout, 0, 1, &m_traceDescSets[frameSlot], 0, nullptr);

    struct CausticTracePC {
        uint32_t numTriangles;
        uint32_t numSpheres;
        uint32_t numMaterials;
        uint32_t numLights;
        uint32_t photonCount;
        uint32_t frameIndex;
        uint32_t numOpaqueTriangles;
        uint32_t maxBounces;
        glm::vec4 targetBBoxMin;
        glm::vec4 targetBBoxMax;
    } pc;

    pc.numTriangles = numTriangles;
    pc.numSpheres = numSpheres;
    pc.numMaterials = numMaterials;
    pc.numLights = numLights;
    pc.photonCount = m_photonCount;
    pc.frameIndex = frameIndex;
    pc.numOpaqueTriangles = numOpaqueTriangles;
    pc.maxBounces = 4u;
    pc.targetBBoxMin = glm::vec4(dielectricBoundsMin, hasDielectrics ? 1.0f : 0.0f);
    pc.targetBBoxMax = glm::vec4(dielectricBoundsMax, 0.0f);

    vkCmdPushConstants(cmd, m_tracePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t groups = (m_photonCount + 31) / 32;
    vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier: Photon buffer write -> Photon buffer read in Splat pass
    VkBufferMemoryBarrier2 photonBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    photonBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    photonBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    photonBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    photonBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    photonBarrier.buffer = m_causticPhotonBuffer->getBuffer();
    photonBarrier.offset = 0;
    photonBarrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers = &photonBarrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void CausticsPipeline::recordSplatAndFilter(
    VkCommandBuffer cmd,
    uint32_t frameSlot,
    Image* normalDepthImage,
    float cameraFov,
    uint32_t frameIndex,
    bool progressiveAccumulation,
    bool cameraMovedLastFrame,
    bool hasDielectrics,
    const glm::vec3& dielectricBoundsMin,
    const glm::vec3& dielectricBoundsMax)
{
    if (!m_splatPipeline || !m_filterPipeline ||
        !m_causticAtomicBuffer || !m_causticPhotonBuffer || !m_filteredCausticImage) return;

    // 1. Fast GPU atomic accumulator buffer clear
    vkCmdFillBuffer(cmd, m_causticAtomicBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0);

    // 2. Barrier: FillBuffer -> Compute Shader Read/Write & NormalDepth Image Read Barrier
    VkBufferMemoryBarrier2 fillBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    fillBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    fillBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    fillBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    fillBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    fillBarrier.buffer = m_causticAtomicBuffer->getBuffer();
    fillBarrier.offset = 0;
    fillBarrier.size = VK_WHOLE_SIZE;

    VkImageMemoryBarrier2 ndBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    ndBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    ndBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    ndBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    ndBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    ndBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    ndBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ndBarrier.image = normalDepthImage ? normalDepthImage->getImage() : VK_NULL_HANDLE;
    ndBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo preSplatDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    preSplatDep.bufferMemoryBarrierCount = 1;
    preSplatDep.pBufferMemoryBarriers = &fillBarrier;
    if (normalDepthImage) {
        preSplatDep.imageMemoryBarrierCount = 1;
        preSplatDep.pImageMemoryBarriers = &ndBarrier;
    }
    vkCmdPipelineBarrier2(cmd, &preSplatDep);

    // 3. Dispatch Splat Pass
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_splatPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_splatPipelineLayout, 0, 1, &m_splatDescSets[frameSlot], 0, nullptr);

    struct CausticSplatPC {
        uint32_t width;
        uint32_t height;
        uint32_t photonCount;
        float fov;
        glm::vec4 targetBBoxMin;
        glm::vec4 targetBBoxMax;
    } splatPC;
    splatPC.width = m_width;
    splatPC.height = m_height;
    splatPC.photonCount = m_photonCount;
    splatPC.fov = cameraFov;
    splatPC.targetBBoxMin = glm::vec4(dielectricBoundsMin, hasDielectrics ? 1.0f : 0.0f);
    splatPC.targetBBoxMax = glm::vec4(dielectricBoundsMax, 0.0f);

    vkCmdPushConstants(cmd, m_splatPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(splatPC), &splatPC);
    uint32_t splatGroups = (m_photonCount + 255) / 256;
    vkCmdDispatch(cmd, splatGroups, 1, 1);

    // 4. Barrier: Splat atomic writes -> Filter read
    VkBufferMemoryBarrier2 splatToFilterBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    splatToFilterBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    splatToFilterBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    splatToFilterBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    splatToFilterBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    splatToFilterBarrier.buffer = m_causticAtomicBuffer->getBuffer();
    splatToFilterBarrier.offset = 0;
    splatToFilterBarrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo filterDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    filterDep.bufferMemoryBarrierCount = 1;
    filterDep.pBufferMemoryBarriers = &splatToFilterBarrier;
    vkCmdPipelineBarrier2(cmd, &filterDep);

    // 5. Dispatch Bilateral Filter & Temporal Accumulation
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_filterPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_filterPipelineLayout, 0, 1, &m_filterDescSets[frameSlot], 0, nullptr);

    uint32_t accumHist = 0u;
    if (progressiveAccumulation && !cameraMovedLastFrame) {
        accumHist = 2u; // stationary progressive
    }

    struct CausticFilterPC {
        uint32_t width;
        uint32_t height;
        uint32_t frameIndex;
        uint32_t accumulateHistory;
    } filterPC;
    filterPC.width = m_width;
    filterPC.height = m_height;
    filterPC.frameIndex = frameIndex;
    filterPC.accumulateHistory = accumHist;

    vkCmdPushConstants(cmd, m_filterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(filterPC), &filterPC);
    uint32_t filterGroupsX = (m_width + 15) / 16;
    uint32_t filterGroupsY = (m_height + 15) / 16;
    vkCmdDispatch(cmd, filterGroupsX, filterGroupsY, 1);

    // 6. Barrier: Filter write -> Copy to Prev Image & Shading Read
    VkImageMemoryBarrier2 filterPostBarrier[2] = {};
    filterPostBarrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    filterPostBarrier[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    filterPostBarrier[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    filterPostBarrier[0].dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    filterPostBarrier[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    filterPostBarrier[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    filterPostBarrier[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    filterPostBarrier[0].image = m_filteredCausticImage->getImage();
    filterPostBarrier[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    filterPostBarrier[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    filterPostBarrier[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    filterPostBarrier[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    filterPostBarrier[1].dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    filterPostBarrier[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    filterPostBarrier[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    filterPostBarrier[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    filterPostBarrier[1].image = m_prevCausticImage->getImage();
    filterPostBarrier[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo copyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    copyDep.imageMemoryBarrierCount = 2;
    copyDep.pImageMemoryBarriers = filterPostBarrier;
    vkCmdPipelineBarrier2(cmd, &copyDep);

    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.extent = { m_width, m_height, 1 };
    vkCmdCopyImage(cmd, m_filteredCausticImage->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                   m_prevCausticImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);

    // Final barrier: m_filteredCausticImage ready for shader read in Bounce 0 shade
    VkImageMemoryBarrier2 finalBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    finalBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    finalBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    finalBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    finalBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    finalBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    finalBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    finalBarrier.image = m_filteredCausticImage->getImage();
    finalBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo finalDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    finalDep.imageMemoryBarrierCount = 1;
    finalDep.pImageMemoryBarriers = &finalBarrier;
    vkCmdPipelineBarrier2(cmd, &finalDep);
}

void CausticsPipeline::recordFullPass(
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
    const glm::vec3& dielectricBoundsMax)
{
    if (!hasDielectrics || numLights == 0) return;

    recordTrace(cmd, frameSlot, numTriangles, numSpheres, numMaterials, numLights,
                numOpaqueTriangles, frameIndex, hasDielectrics,
                dielectricBoundsMin, dielectricBoundsMax);

    recordSplatAndFilter(cmd, frameSlot, normalDepthImage, cameraFov, frameIndex,
                         progressiveAccumulation, cameraMovedLastFrame,
                         hasDielectrics, dielectricBoundsMin, dielectricBoundsMax);
}

} // namespace pathways
