#include "vulkan/RenderTargetManager.hpp"
#include "core/Logger.hpp"

namespace pathways {

RenderTargetManager::RenderTargetManager(VkDevice device, VmaAllocator allocator, VkQueue queue)
    : m_device(device), m_allocator(allocator), m_queue(queue) {
}

RenderTargetManager::~RenderTargetManager() {
    destroy();
}

void RenderTargetManager::destroy() {
    destroyGBufferResources();
    for (auto& img : m_frameImages) img.reset();
    m_accumImage.reset();
    m_outputImage.reset();
    m_motionVectorImage.reset();
    m_mlAlbedoRoughnessImage.reset();
    m_mlSpecularMotionImage.reset();
    m_mlDiffuseImage.reset();
    m_mlSpecularImage.reset();
}

void RenderTargetManager::destroyGBufferResources() {
    m_directLightImage.reset();
    m_normalDepthImage.reset();
    m_prevNormalDepthImage.reset();
}

void RenderTargetManager::transitionAllToGeneral(VkCommandBuffer cmd) {
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_frameImages[i]) {
            m_frameImages[i]->transitionLayout(
                cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
            );
        }
    }

    if (m_accumImage) {
        m_accumImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }

    if (m_outputImage) {
        m_outputImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }

    if (m_motionVectorImage) {
        m_motionVectorImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }

    if (m_mlAlbedoRoughnessImage) {
        m_mlAlbedoRoughnessImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }
    if (m_mlSpecularMotionImage) {
        m_mlSpecularMotionImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }
    if (m_mlDiffuseImage) {
        m_mlDiffuseImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }
    if (m_mlSpecularImage) {
        m_mlSpecularImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }

    vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = cmd;

    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
    vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_queue);
}

void RenderTargetManager::createRenderTargets(uint32_t width, uint32_t height, AccumFormat accumFormat,
                                             bool isHdr, VkFormat swapchainFormat,
                                             const std::vector<uint32_t>& concurrentQueues,
                                             VkCommandBuffer cmd) {
    VkFormat frameFmt = (accumFormat == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_frameImages[i] = std::make_unique<Image>(
            m_device, m_allocator, width, height,
            frameFmt,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            concurrentQueues
        );
    }
    m_accumImage = std::make_unique<Image>(
        m_device, m_allocator, width, height,
        frameFmt,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        concurrentQueues
    );

    // Internal post-process and tonemapping target is strictly 10-bit (or 16-bit float for scRGB HDR).
    // The internal pipeline is never degraded to 8-bit; SDR 8-bit presentation occurs via blit to the swapchain.
    VkFormat outputFmt = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    if (isHdr && swapchainFormat == VK_FORMAT_R16G16B16A16_SFLOAT) {
        outputFmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    m_outputImage = std::make_unique<Image>(
        m_device, m_allocator, width, height,
        outputFmt,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        concurrentQueues
    );

    m_motionVectorImage = std::make_unique<Image>(
        m_device, m_allocator, width, height,
        VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        concurrentQueues
    );

    m_mlAlbedoRoughnessImage = std::make_unique<Image>(
        m_device, m_allocator, width, height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_mlSpecularMotionImage = std::make_unique<Image>(
        m_device, m_allocator, width, height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    m_mlDiffuseImage = std::make_unique<Image>(
        m_device, m_allocator, width, height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        concurrentQueues
    );
    m_mlSpecularImage = std::make_unique<Image>(
        m_device, m_allocator, width, height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    transitionAllToGeneral(cmd);
}

void RenderTargetManager::recreateAccumImages(uint32_t width, uint32_t height, AccumFormat accumFormat,
                                             const std::vector<uint32_t>& concurrentQueues,
                                             VkCommandBuffer cmd) {
    VkFormat frameFmt = (accumFormat == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_frameImages[i] = std::make_unique<Image>(
            m_device, m_allocator, width, height,
            frameFmt,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT,
            concurrentQueues
        );
    }
    m_accumImage = std::make_unique<Image>(
        m_device, m_allocator, width, height,
        frameFmt,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        concurrentQueues
    );

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo transBegin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    transBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &transBegin);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_frameImages[i]->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }
    m_accumImage->transitionLayout(
        cmd, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );
    vkEndCommandBuffer(cmd);
    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = cmd;
    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
    vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_queue);
}

void RenderTargetManager::createGBufferResources(uint32_t width, uint32_t height,
                                                 const std::vector<uint32_t>& concurrentQueues,
                                                 VkCommandBuffer cmd) {
    m_directLightImage = std::make_unique<Image>(
        m_device, m_allocator, width, height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_normalDepthImage = std::make_unique<Image>(
        m_device, m_allocator, width, height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
        VK_IMAGE_ASPECT_COLOR_BIT,
        concurrentQueues
    );

    m_prevNormalDepthImage = std::make_unique<Image>(
        m_device, m_allocator, width, height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &beginInfo);
    m_directLightImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    m_normalDepthImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    m_prevNormalDepthImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = cmd;
    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
    vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_queue);

    Logger::info("G-Buffer resources allocated successfully.");
}

void RenderTargetManager::resize(uint32_t width, uint32_t height, AccumFormat accumFormat,
                                bool isHdr, VkFormat swapchainFormat,
                                const std::vector<uint32_t>& concurrentQueues,
                                VkCommandBuffer cmd) {
    createRenderTargets(width, height, accumFormat, isHdr, swapchainFormat, concurrentQueues, cmd);
}

} // namespace pathways
