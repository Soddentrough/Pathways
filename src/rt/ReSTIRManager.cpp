#include "ReSTIRManager.hpp"
#include "core/Logger.hpp"
#include <stdexcept>

namespace pathways {

static inline VkBufferMemoryBarrier2 makeBufferBarrier2(
    VkBuffer buffer,
    VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
    VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE)
{
    VkBufferMemoryBarrier2 b{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    b.srcStageMask = srcStage;
    b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStage;
    b.dstAccessMask = dstAccess;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = buffer;
    b.offset = offset;
    b.size = size;
    return b;
}

ReSTIRManager::ReSTIRManager(VkDevice device, VmaAllocator allocator,
                             uint32_t width, uint32_t height,
                             const std::vector<char>& temporalSpv,
                             const std::vector<char>& spatialSpv)
    : m_device(device), m_allocator(allocator), m_width(width), m_height(height)
{
    if (temporalSpv.empty() || spatialSpv.empty()) {
        Logger::error("ReSTIRManager: Provided SPIR-V bytecode is empty!");
        return;
    }

    initBuffers();
    createDescriptorLayout();
    allocateDescriptorSets();
    createPipelines(temporalSpv, spatialSpv);

    m_initialized = true;
    Logger::info("ReSTIRManager initialized successfully at {}x{}.", m_width, m_height);
}

ReSTIRManager::~ReSTIRManager() {
    if (m_temporalPipeline)  vkDestroyPipeline(m_device, m_temporalPipeline, nullptr);
    if (m_spatialPipeline)   vkDestroyPipeline(m_device, m_spatialPipeline, nullptr);
    if (m_pipelineLayout)    vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);

    if (m_descSetLayout)     vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);
    if (m_descriptorPool)    vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);

    for (auto& b : m_temporalBuffers)  b.reset();
    for (auto& b : m_spatialBuffers)   b.reset();
}

void ReSTIRManager::initBuffers() {
    VkDeviceSize resSize = static_cast<VkDeviceSize>(m_width) * m_height * sizeof(ReservoirDI);
    if (resSize == 0) return;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_spatialBuffers[i] = std::make_unique<Buffer>(
            m_allocator, resSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    }

    for (uint32_t i = 0; i < 2; ++i) {
        m_temporalBuffers[i] = std::make_unique<Buffer>(
            m_allocator, resSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
    }

    if (!m_dummyConfidenceImage) {
        m_dummyConfidenceImage = std::make_unique<Image>(
            m_device, m_allocator,
            1, 1,
            VK_FORMAT_R16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT
        );
    }

    m_temporalIndex = 0;
}

void ReSTIRManager::resize(uint32_t width, uint32_t height) {
    if (m_width == width && m_height == height) return;
    m_width = width;
    m_height = height;
    initBuffers();
    Logger::info("ReSTIRManager resized to {}x{}.", m_width, m_height);
}

void ReSTIRManager::createDescriptorLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // Reserved / Unused
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // PrevTemporalBuffer
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // InRayGeomQueue
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // InRayHitQueue
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // LightsBuffer
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // MaterialsBuffer
        { 6, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // CameraUBO
        { 7, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // MotionVectorImage
        { 8, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // NormalDepthImage
        { 9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },  // PrevNormalDepthImage
        { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // LightTreeBuffer
        { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // OutTemporalBuffer / InTemporalBuffer
        { 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // OutSpatialBuffer
        { 13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }, // PixelToRayBuffer
        { 14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }  // ReconstructConfidenceImage
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("ReSTIRManager: Failed to create descriptor set layout!");
    }
}

void ReSTIRManager::allocateDescriptorSets() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  32 }
    };

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("ReSTIRManager: Failed to create descriptor pool!");
    }

    std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts;
    layouts.fill(m_descSetLayout);

    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();

    if (vkAllocateDescriptorSets(m_device, &allocInfo, m_descSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("ReSTIRManager: Failed to allocate descriptor sets!");
    }
}

VkShaderModule ReSTIRManager::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("ReSTIRManager: Failed to create shader module!");
    }
    return shaderModule;
}

void ReSTIRManager::createPipelines(const std::vector<char>& temporalSpv,
                                    const std::vector<char>& spatialSpv)
{
    VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ReSTIRPushConstants) };

    VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_descSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pcRange;

    if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("ReSTIRManager: Failed to create pipeline layout!");
    }

    VkShaderModule temporalMod  = createShaderModule(temporalSpv);
    VkShaderModule spatialMod   = createShaderModule(spatialSpv);

    auto makeComputePipeline = [&](VkShaderModule module, VkPipeline* outPipeline) {
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipelineInfo.stage.module = module;
        pipelineInfo.stage.pName = "main";
        pipelineInfo.layout = m_pipelineLayout;

        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, outPipeline) != VK_SUCCESS) {
            throw std::runtime_error("ReSTIRManager: Failed to create compute pipeline!");
        }
    };

    makeComputePipeline(temporalMod, &m_temporalPipeline);
    makeComputePipeline(spatialMod, &m_spatialPipeline);

    vkDestroyShaderModule(m_device, temporalMod, nullptr);
    vkDestroyShaderModule(m_device, spatialMod, nullptr);
}

void ReSTIRManager::recordFrame(VkCommandBuffer cmd, uint32_t frameSlot, uint32_t width, uint32_t height,
                                uint32_t numLights, uint32_t numTriangles, bool hasLightTree,
                                uint32_t frameIndex, uint32_t mCap,
                                Buffer* rayGeomQueue, Buffer* rayHitQueue, Buffer* pixelToRayQueue,
                                Buffer* lightsBuffer, Buffer* materialsBuffer,
                                Buffer* cameraUBO, Buffer* lightTreeBuffer,
                                VkImageView motionVectorView, VkImageView normalDepthView, VkImageView prevNormalDepthView,
                                VkImageView reconstructConfidenceView)
{
    if (!m_initialized || width == 0 || height == 0 || numLights == 0) return;

    if (width != m_width || height != m_height) {
        resize(width, height);
    }

    uint32_t prevTemporalIdx = 1 - m_temporalIndex;
    uint32_t currTemporalIdx = m_temporalIndex;

    Buffer* prevTemporalBuf = m_temporalBuffers[prevTemporalIdx].get();
    Buffer* currTemporalBuf = m_temporalBuffers[currTemporalIdx].get();
    Buffer* spatialBuf      = m_spatialBuffers[frameSlot].get();

    if (!prevTemporalBuf || !currTemporalBuf || !spatialBuf) return;

    // 1. Update Descriptor Set for this frameSlot
    VkDescriptorSet dset = m_descSets[frameSlot];

    VkDescriptorBufferInfo dummyCandidateInfo{ currTemporalBuf->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo prevTemporalInfo{ prevTemporalBuf->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo geomInfo{ rayGeomQueue ? rayGeomQueue->getBuffer() : currTemporalBuf->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo hitInfo{ rayHitQueue ? rayHitQueue->getBuffer() : currTemporalBuf->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo lightsInfo{ lightsBuffer ? lightsBuffer->getBuffer() : currTemporalBuf->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo matInfo{ materialsBuffer ? materialsBuffer->getBuffer() : currTemporalBuf->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo camInfo{ cameraUBO ? cameraUBO->getBuffer() : currTemporalBuf->getBuffer(), 0, VK_WHOLE_SIZE };

    VkDescriptorImageInfo mvImgInfo{ VK_NULL_HANDLE, motionVectorView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo ndImgInfo{ VK_NULL_HANDLE, normalDepthView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo prevNdImgInfo{ VK_NULL_HANDLE, (prevNormalDepthView != VK_NULL_HANDLE) ? prevNormalDepthView : normalDepthView, VK_IMAGE_LAYOUT_GENERAL };

    VkImageView activeConfView = reconstructConfidenceView;
    if (activeConfView == VK_NULL_HANDLE && m_dummyConfidenceImage) {
        if (m_dummyConfidenceImage->getLayout() != VK_IMAGE_LAYOUT_GENERAL) {
            m_dummyConfidenceImage->transitionLayout(
                cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT
            );
        }
        activeConfView = m_dummyConfidenceImage->getImageView();
    }
    VkDescriptorImageInfo confImgInfo{ VK_NULL_HANDLE, activeConfView, VK_IMAGE_LAYOUT_GENERAL };

    VkDescriptorBufferInfo lightTreeInfo{ (lightTreeBuffer && hasLightTree) ? lightTreeBuffer->getBuffer() : lightsInfo.buffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo currTemporalInfo{ currTemporalBuf->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo spatialInfo{ spatialBuf->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo pixelToRayInfo{ pixelToRayQueue ? pixelToRayQueue->getBuffer() : currTemporalBuf->getBuffer(), 0, VK_WHOLE_SIZE };

    std::vector<VkWriteDescriptorSet> writes = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dummyCandidateInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &prevTemporalInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &geomInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hitInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightsInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 6, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &camInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  &mvImgInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  &ndImgInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  &prevNdImgInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightTreeInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &currTemporalInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &spatialInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &pixelToRayInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  &confImgInfo, nullptr, nullptr }
    };

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    ReSTIRPushConstants pc{};
    pc.width = width;
    pc.height = height;
    pc.numLights = numLights;
    pc.frameIndex = frameIndex;
    pc.mCap = mCap;
    pc.numTriangles = numTriangles;
    pc.hasLightTree = (hasLightTree && lightTreeBuffer) ? 1u : 0u;
    pc.passIndex = 0;

    uint32_t dispatchX = (width + 15) / 16;
    uint32_t dispatchY = (height + 15) / 16;

    // --- PASS 1: Fused Candidate Generation & Temporal Resampling ---
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &dset, 0, nullptr);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, dispatchX, dispatchY, 1);

    // Barrier: CurrTemporalBuffer write -> Spatial pass read
    VkBufferMemoryBarrier2 t2sBarrier = makeBufferBarrier2(
        currTemporalBuf->getBuffer(),
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    VkDependencyInfo t2sDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    t2sDep.bufferMemoryBarrierCount = 1;
    t2sDep.pBufferMemoryBarriers = &t2sBarrier;
    vkCmdPipelineBarrier2(cmd, &t2sDep);

    // --- PASS 2: Spatial Resampling with On-Chip LDS Tile ---
    pc.passIndex = 1;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_spatialPipeline);
    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    vkCmdDispatch(cmd, dispatchX, dispatchY, 1);

    // Barrier: SpatialBuffer write -> Shading pass read
    VkBufferMemoryBarrier2 s2shadeBarrier = makeBufferBarrier2(
        spatialBuf->getBuffer(),
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    VkDependencyInfo s2shadeDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    s2shadeDep.bufferMemoryBarrierCount = 1;
    s2shadeDep.pBufferMemoryBarriers = &s2shadeBarrier;
    vkCmdPipelineBarrier2(cmd, &s2shadeDep);

    // Advance temporal ping-pong for next frame
    m_temporalIndex = 1 - m_temporalIndex;
}

} // namespace pathways
