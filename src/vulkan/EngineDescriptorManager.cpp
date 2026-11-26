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
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
        m_descriptorPool = VK_NULL_HANDLE;
    }
    m_secTransferBuffer.reset();
}

void EngineDescriptorManager::init() {
    // 1. Descriptor Pool (shared by PostProcessPipeline & transfer passes)
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
        Buffer* restirCurrBuf = params.restirManager ? params.restirManager->getReservoirBuffer(slot) : nullptr;
        Buffer* restirPrevBuf = params.restirManager ? params.restirManager->getReservoirBuffer(1 - slot) : nullptr;
        VkBuffer restirReservoirBuf = restirCurrBuf ? restirCurrBuf->getBuffer() : VK_NULL_HANDLE;
        VkBuffer restirHistoryReservoirBuf = restirPrevBuf ? restirPrevBuf->getBuffer() : VK_NULL_HANDLE;
        Buffer* giCurrBuf = params.restirManager ? params.restirManager->getGiReservoirBuffer(slot) : nullptr;
        Buffer* giPrevBuf = params.restirManager ? params.restirManager->getGiReservoirBuffer(1 - slot) : nullptr;
        Buffer* x1CtxBuf = params.restirManager ? params.restirManager->getX1ContextBuffer() : nullptr;
        params.wavefrontPipeline->updateSceneDescriptors(
            slot,
            params.frameImages[slot]->getImageView(),
            (*params.cameraUBOs)[slot]->getBuffer(),
            params.triangleBuffer->getBuffer(), params.triangleBuffer->getSize(),
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
            params.shadeMaterialBuffer ? params.shadeMaterialBuffer->getSize() : 0,
            restirHistoryReservoirBuf,
            x1CtxBuf ? x1CtxBuf->getBuffer() : VK_NULL_HANDLE,
            giCurrBuf ? giCurrBuf->getBuffer() : VK_NULL_HANDLE,
            giPrevBuf ? giPrevBuf->getBuffer() : VK_NULL_HANDLE,
            params.prevNormalDepthImage ? params.prevNormalDepthImage->getImageView() : VK_NULL_HANDLE,
            params.mlSkipImage ? params.mlSkipImage->getImageView() : VK_NULL_HANDLE,
            params.mlSpecAlbedoImage ? params.mlSpecAlbedoImage->getImageView() : VK_NULL_HANDLE,
            params.mlDirectDiffuseImage ? params.mlDirectDiffuseImage->getImageView() : VK_NULL_HANDLE,
            params.mlIndirectDiffuseImage ? params.mlIndirectDiffuseImage->getImageView() : VK_NULL_HANDLE,
            params.mlDirectSpecularImage ? params.mlDirectSpecularImage->getImageView() : VK_NULL_HANDLE,
            params.mlIndirectSpecularImage ? params.mlIndirectSpecularImage->getImageView() : VK_NULL_HANDLE,
            params.mlIrradianceImage ? params.mlIrradianceImage->getImageView() : VK_NULL_HANDLE,
            params.mlPrimaryGuideImage ? params.mlPrimaryGuideImage->getImageView() : VK_NULL_HANDLE,
            params.mlSurfaceMvImage ? params.mlSurfaceMvImage->getImageView() : VK_NULL_HANDLE,
            params.mlDiffHitDistImage ? params.mlDiffHitDistImage->getImageView() : VK_NULL_HANDLE,
            params.mlSpecVmvImage ? params.mlSpecVmvImage->getImageView() : VK_NULL_HANDLE,
            params.mlSkipCompactImage ? params.mlSkipCompactImage->getImageView() : VK_NULL_HANDLE,
            params.envImportanceBuffer ? params.envImportanceBuffer->getBuffer() : VK_NULL_HANDLE,
            params.envImportanceBuffer ? params.envImportanceBuffer->getSize() : 0
        );
    }

    if (params.nrcManager) {
        VkImageView mlDiffView = params.mlDiffuseImage ? params.mlDiffuseImage->getImageView() : VK_NULL_HANDLE;
        VkImageView mlSpecView = params.mlSpecularImage ? params.mlSpecularImage->getImageView() : VK_NULL_HANDLE;
        VkImageView mlArView = params.mlAlbedoRoughnessImage ? params.mlAlbedoRoughnessImage->getImageView() : VK_NULL_HANDLE;
        VkImageView mlSaView = params.mlSpecAlbedoImage ? params.mlSpecAlbedoImage->getImageView() : VK_NULL_HANDLE;
        VkImageView mlIndDiffView = params.mlIndirectDiffuseImage ? params.mlIndirectDiffuseImage->getImageView() : VK_NULL_HANDLE;
        VkImageView mlIndSpecView = params.mlIndirectSpecularImage ? params.mlIndirectSpecularImage->getImageView() : VK_NULL_HANDLE;
        VkImageView mlIrrView = params.mlIrradianceImage ? params.mlIrradianceImage->getImageView() : VK_NULL_HANDLE;
        for (uint32_t slot = 0; slot < 2; ++slot) {
            if (params.frameImages[slot]) {
                params.nrcManager->updateDescriptors(
                    slot,
                    params.frameImages[slot]->getImageView(),
                    mlDiffView, mlSpecView,
                    mlArView, mlSaView,
                    mlIndDiffView, mlIndSpecView,
                    mlIrrView
                );
            }
        }
    }
}

void EngineDescriptorManager::updateMergeDescriptors(const MergeDescriptorParams& params) {
    if (!params.postProcess || !params.motionVectorImage || !params.normalDepthImage) {
        return;
    }

    constexpr VkDeviceSize kDummyTransferSize = 65536;
    const bool mgpuSharedActive = params.mgpu && params.mgpu->isSecondaryInitialized() &&
                                  (params.mgpu->isZeroCopyActive() || params.mgpu->isP2PDirectBarActive());
    uint32_t bytesPerPixel = (params.accumFormat == AccumFormat::RGBA16_SFLOAT) ? 80 :
                             (params.accumFormat == AccumFormat::R11G11B10_UFLOAT) ? 76 : 88;
    VkDeviceSize bufferSize = mgpuSharedActive
        ? params.mgpu->getSharedBufferSize()
        : ((params.mgpu && params.mgpu->isSecondaryInitialized())
            ? (static_cast<VkDeviceSize>(params.width) * params.height * bytesPerPixel)
            : kDummyTransferSize);

    for (uint32_t slot = 0; slot < 2; ++slot) {
        VkBuffer secBuffer = VK_NULL_HANDLE;
        VkDeviceSize curSize = bufferSize;

        if (mgpuSharedActive) {
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
            params.accumFormat,
            params.mlDiffuseImage,
            params.mlSpecularImage,
            params.mlAlbedoRoughnessImage,
            params.mlSpecularMotionImage,
            params.mlSkipImage,
            params.mlSpecAlbedoImage,
            params.mlIrradianceImage
        );
    }
}

} // namespace pathways
