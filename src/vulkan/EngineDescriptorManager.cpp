#include "vulkan/EngineDescriptorManager.hpp"
#include "rt/WavefrontPipeline.hpp"
#include "rt/NRCManager.hpp"
#include "rt/ReSTIRManager.hpp"
#include "rt/PostProcessPipeline.hpp"
#include "mgpu/MultiGpuManager.hpp"

#include <stdexcept>

namespace pathways {

EngineDescriptorManager::EngineDescriptorManager(VkDevice device, VmaAllocator allocator)
    : m_device(device), m_allocator(allocator) {
}

EngineDescriptorManager::~EngineDescriptorManager() {
    destroy();
}

void EngineDescriptorManager::destroy() {
    if (m_rtDescLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_rtDescLayout, nullptr);
        m_rtDescLayout = VK_NULL_HANDLE;
    }
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
    m_rtDescSets.fill(VK_NULL_HANDLE);
    m_secTransferBuffer.reset();
}

void EngineDescriptorManager::init(
    const std::array<Image*, MAX_FRAMES_IN_FLIGHT>& frameImages,
    const std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT>& cameraUBOs
) {
    // 1. Descriptor Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 32 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 256;
    VkResult res = vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create primary descriptor pool!");
    }

    // 2. Ray Tracing Descriptor Set Layout (VK_KHR_ray_tracing_pipeline)
    VkShaderStageFlags rtStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

    std::vector<VkDescriptorSetLayoutBinding> rtBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, rtStages, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, rtStages, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, rtStages, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_SCENE_TEXTURES, rtStages, nullptr },
        { 11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, rtStages, nullptr },
        { 14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 15, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo rtLayoutInfo{};
    rtLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    rtLayoutInfo.bindingCount = static_cast<uint32_t>(rtBindings.size());
    rtLayoutInfo.pBindings = rtBindings.data();
    res = vkCreateDescriptorSetLayout(m_device, &rtLayoutInfo, nullptr, &m_rtDescLayout);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ray tracing descriptor set layout!");
    }

    // 3. Allocate Descriptor Sets
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        VkDescriptorSetAllocateInfo rtAllocInfo{};
        rtAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        rtAllocInfo.descriptorPool = m_descriptorPool;
        rtAllocInfo.descriptorSetCount = 1;
        rtAllocInfo.pSetLayouts = &m_rtDescLayout;
        vkAllocateDescriptorSets(m_device, &rtAllocInfo, &m_rtDescSets[i]);
    }

    // 4. Initial bindings for frame images & camera UBOs
    std::vector<VkDescriptorImageInfo> frameImageInfos(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (frameImages[i]) {
            frameImageInfos[i].sampler = VK_NULL_HANDLE;
            frameImageInfos[i].imageView = frameImages[i]->getImageView();
            frameImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
    }

    std::vector<VkDescriptorBufferInfo> uboBufferInfos(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (cameraUBOs[i]) {
            uboBufferInfos[i] = { cameraUBOs[i]->getBuffer(), 0, sizeof(CameraUniform) };
        }
    }

    std::vector<VkWriteDescriptorSet> writes;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (frameImages[i]) {
            writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &frameImageInfos[i], nullptr, nullptr });
        }
        if (cameraUBOs[i]) {
            writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboBufferInfos[i], nullptr });
        }
    }
    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void EngineDescriptorManager::updatePrimaryImageDescriptors(const ImageDescriptorParams& params) {
    if (!params.accumImage || !params.outputImage) return;

    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageView = params.accumImage->getImageView();
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo directLightInfo{};
    if (params.directLightImage) {
        directLightInfo.imageView = params.directLightImage->getImageView();
        directLightInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkDescriptorImageInfo normDepthInfo{};
    if (params.normalDepthImage) {
        normDepthInfo.imageView = params.normalDepthImage->getImageView();
        normDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkDescriptorImageInfo mvImageInfo{};
    if (params.motionVectorImage) {
        mvImageInfo.imageView = params.motionVectorImage->getImageView();
        mvImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    VkDescriptorImageInfo causticInfo{};
    causticInfo.imageView = params.filteredCausticImage ? params.filteredCausticImage->getImageView() : accumImageInfo.imageView;
    causticInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkDescriptorImageInfo, MAX_FRAMES_IN_FLIGHT> frameImageInfos;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (params.frameImages[i]) {
            frameImageInfos[i].sampler = VK_NULL_HANDLE;
            frameImageInfos[i].imageView = params.frameImages[i]->getImageView();
            frameImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
    }

    std::vector<VkWriteDescriptorSet> writes;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_rtDescSets[i] != VK_NULL_HANDLE) {
            if (params.frameImages[i]) {
                VkWriteDescriptorSet w0{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                w0.dstSet = m_rtDescSets[i];
                w0.dstBinding = 0;
                w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w0.descriptorCount = 1;
                w0.pImageInfo = &frameImageInfos[i];
                writes.push_back(w0);
            }

            if (params.directLightImage && params.normalDepthImage) {
                VkWriteDescriptorSet w11{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                w11.dstSet = m_rtDescSets[i];
                w11.dstBinding = 11;
                w11.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w11.descriptorCount = 1;
                w11.pImageInfo = &directLightInfo;
                writes.push_back(w11);

                VkWriteDescriptorSet w12{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                w12.dstSet = m_rtDescSets[i];
                w12.dstBinding = 12;
                w12.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w12.descriptorCount = 1;
                w12.pImageInfo = &normDepthInfo;
                writes.push_back(w12);
            }

            if (params.motionVectorImage) {
                VkWriteDescriptorSet w14{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                w14.dstSet = m_rtDescSets[i];
                w14.dstBinding = 14;
                w14.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w14.descriptorCount = 1;
                w14.pImageInfo = &mvImageInfo;
                writes.push_back(w14);
            }

            if (causticInfo.imageView != VK_NULL_HANDLE) {
                VkWriteDescriptorSet w15{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                w15.dstSet = m_rtDescSets[i];
                w15.dstBinding = 15;
                w15.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w15.descriptorCount = 1;
                w15.pImageInfo = &causticInfo;
                writes.push_back(w15);
            }
        }
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void EngineDescriptorManager::updatePrimarySceneDescriptors(const SceneDescriptorParams& params) {
    if (!params.triangleBuffer || !params.sphereBuffer || !params.materialBuffer || !params.lightBuffer) return;

    VkDescriptorBufferInfo triBufferInfo{ params.triangleBuffer->getBuffer(), 0, params.triangleBuffer->getSize() };
    VkDescriptorBufferInfo sphereBufferInfo{ params.sphereBuffer->getBuffer(), 0, params.sphereBuffer->getSize() };
    VkDescriptorBufferInfo matBufferInfo{ params.materialBuffer->getBuffer(), 0, params.materialBuffer->getSize() };
    VkDescriptorBufferInfo lightBufferInfo{ params.lightBuffer->getBuffer(), 0, params.lightBuffer->getSize() };

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    VkAccelerationStructureKHR tlasHandle = params.tlasHandle;
    asInfo.pAccelerationStructures = &tlasHandle;

    VkDescriptorImageInfo envInfo = params.environmentMap ? params.environmentMap->getDescriptorInfo() : 
                                    (params.dummyWhite ? params.dummyWhite->getDescriptorInfo() : VkDescriptorImageInfo{});

    std::vector<VkDescriptorImageInfo> texInfos(MAX_SCENE_TEXTURES);
    for (size_t i = 0; i < MAX_SCENE_TEXTURES; ++i) {
        if (params.sceneTextures && i < params.sceneTextures->size() && (*params.sceneTextures)[i]) {
            texInfos[i] = (*params.sceneTextures)[i]->getDescriptorInfo();
        } else if (params.dummyWhite) {
            texInfos[i] = params.dummyWhite->getDescriptorInfo();
        }
    }

    std::vector<VkWriteDescriptorSet> writes;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_rtDescSets[i] == VK_NULL_HANDLE) continue;
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sphereBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, m_rtDescSets[i], 6, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 8, 0, MAX_SCENE_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texInfos.data(), nullptr, nullptr });
        VkDescriptorImageInfo bnInfo = params.blueNoiseTexture ? params.blueNoiseTexture->getDescriptorInfo() : 
                                      (params.dummyWhite ? params.dummyWhite->getDescriptorInfo() : VkDescriptorImageInfo{});
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 13, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &bnInfo, nullptr, nullptr });
    }
    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void EngineDescriptorManager::updateWavefrontDescriptors(const WavefrontDescriptorParams& params) {
    if (!params.wavefrontPipeline || !params.accumImage || !params.triangleBuffer) return;

    VkDescriptorImageInfo envInfo = params.environmentMap ? params.environmentMap->getDescriptorInfo() : 
                                    (params.dummyWhite ? params.dummyWhite->getDescriptorInfo() : VkDescriptorImageInfo{});

    std::vector<VkDescriptorImageInfo> texInfos(MAX_SCENE_TEXTURES);
    for (size_t i = 0; i < MAX_SCENE_TEXTURES; ++i) {
        if (params.sceneTextures && i < params.sceneTextures->size() && (*params.sceneTextures)[i]) {
            texInfos[i] = (*params.sceneTextures)[i]->getDescriptorInfo();
        } else if (params.dummyWhite) {
            texInfos[i] = params.dummyWhite->getDescriptorInfo();
        }
    }

    VkBuffer nrcQueryBuf = params.nrcManager ? params.nrcManager->getQueryQueue()->getBuffer() : VK_NULL_HANDLE;
    VkBuffer nrcTrainBuf = params.nrcManager ? params.nrcManager->getTrainQueue()->getBuffer() : VK_NULL_HANDLE;
    VkBuffer nrcCountBuf = params.nrcManager ? params.nrcManager->getCounters()->getBuffer() : VK_NULL_HANDLE;

    for (uint32_t slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
        if (!params.cameraUBOs || !(*params.cameraUBOs)[slot] || !params.frameImages[slot]) continue;
        VkBuffer restirReservoirBuf = params.restirManager ? params.restirManager->getSpatialReservoirBuffer(slot)->getBuffer() : VK_NULL_HANDLE;
        params.wavefrontPipeline->updateSceneDescriptors(
            slot,
            params.frameImages[slot]->getImageView(),
            (*params.cameraUBOs)[slot]->getBuffer(),
            params.triangleBuffer->getBuffer(), params.triangleBuffer->getSize(),
            params.sphereBuffer->getBuffer(), params.sphereBuffer->getSize(),
            params.materialBuffer->getBuffer(), params.materialBuffer->getSize(),
            params.lightBuffer->getBuffer(), params.lightBuffer->getSize(),
            params.tlasHandle,
            envInfo,
            texInfos,
            nrcQueryBuf,
            nrcTrainBuf,
            nrcCountBuf,
            params.motionVectorImage ? params.motionVectorImage->getImageView() : VK_NULL_HANDLE,
            params.normalDepthImage ? params.normalDepthImage->getImageView() : VK_NULL_HANDLE,
            params.lightTreeBuffer ? params.lightTreeBuffer->getBuffer() : VK_NULL_HANDLE,
            params.lightTreeBuffer ? params.lightTreeBuffer->getSize() : 0,
            params.mlAlbedoRoughnessImage ? params.mlAlbedoRoughnessImage->getImageView() : VK_NULL_HANDLE,
            params.mlSpecularMotionImage ? params.mlSpecularMotionImage->getImageView() : VK_NULL_HANDLE,
            params.mlDiffuseImage ? params.mlDiffuseImage->getImageView() : VK_NULL_HANDLE,
            params.mlSpecularImage ? params.mlSpecularImage->getImageView() : VK_NULL_HANDLE,
            params.instanceBuffer ? params.instanceBuffer->getBuffer() : VK_NULL_HANDLE,
            params.instanceBuffer ? params.instanceBuffer->getSize() : 0,
            params.filteredCausticImage ? params.filteredCausticImage->getImageView() : VK_NULL_HANDLE,
            restirReservoirBuf,
            params.materialArchetypeBuffer ? params.materialArchetypeBuffer->getBuffer() : VK_NULL_HANDLE,
            params.materialArchetypeBuffer ? params.materialArchetypeBuffer->getSize() : 0,
            params.shadeMaterialBuffer ? params.shadeMaterialBuffer->getBuffer() : VK_NULL_HANDLE,
            params.shadeMaterialBuffer ? params.shadeMaterialBuffer->getSize() : 0
        );
    }

    if (params.nrcManager && params.frameImages[0]) {
        params.nrcManager->updateDescriptors(params.frameImages[0]->getImageView());
    }
}

void EngineDescriptorManager::updateMergeDescriptors(const MergeDescriptorParams& params) {
    if (!params.postProcess || !params.motionVectorImage || !params.normalDepthImage) {
        return;
    }

    uint32_t bytesPerPixel = (params.accumFormat == AccumFormat::RGBA16_SFLOAT) ? 24 :
                             (params.accumFormat == AccumFormat::R11G11B10_UFLOAT) ? 20 : 32;
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(params.width) * params.height * bytesPerPixel;

    for (uint32_t slot = 0; slot < 2; ++slot) {
        VkBuffer secBuffer = VK_NULL_HANDLE;
        VkDeviceSize curSize = bufferSize;

        if (params.mgpu && params.mgpu->isSecondaryInitialized() && 
            (params.mgpu->isZeroCopyActive() || params.mgpu->isP2PDirectBarActive())) {
            secBuffer = params.mgpu->getPrimarySharedBuffer(slot);
            curSize = params.mgpu->getSharedBufferSize();
        }
        if (secBuffer == VK_NULL_HANDLE) {
            if (!m_secTransferBuffer || m_secTransferBuffer->getSize() < bufferSize) {
                m_secTransferBuffer = std::make_unique<Buffer>(
                    m_allocator, bufferSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT,
                    0, 0,
                    params.concurrentQueues
                );
            }
            secBuffer = m_secTransferBuffer->getBuffer();
            curSize = m_secTransferBuffer->getSize();
        }

        params.postProcess->updateMergeDescriptors(
            slot, secBuffer, curSize,
            params.frameImages[slot],
            params.motionVectorImage,
            params.normalDepthImage,
            params.width, params.height,
            params.accumFormat
        );
    }
}

} // namespace pathways
