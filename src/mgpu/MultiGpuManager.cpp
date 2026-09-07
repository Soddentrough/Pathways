#include "mgpu/MultiGpuManager.hpp"
#include "core/Logger.hpp"

#include <fstream>
#include <filesystem>
#include <cstring>
#include <bit>

namespace pathways {

GpuDeviceNode::~GpuDeviceNode() {
    if (!context) return;
    VkDevice device = context->getDevice();
    vkDeviceWaitIdle(device);

    if (queryPool) vkDestroyQueryPool(device, queryPool, nullptr);
    if (rtPipeline) vkDestroyPipeline(device, rtPipeline, nullptr);
    if (rtPipelineLayout) vkDestroyPipelineLayout(device, rtPipelineLayout, nullptr);
    if (rtDescLayout) vkDestroyDescriptorSetLayout(device, rtDescLayout, nullptr);
    if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (renderFence) vkDestroyFence(device, renderFence, nullptr);
    if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);

    accumTarget.reset();
    p2pStagingBuffer.reset();
    triangleBuffer.reset();
    sphereBuffer.reset();
    materialBuffer.reset();
    lightBuffer.reset();
    cameraUBO.reset();
}

MultiGpuManager::MultiGpuManager(const Config& config, VulkanContext* primaryContext, const SceneData& scene)
    : m_primaryContext(primaryContext), m_mode(config.mgpu_mode), m_config(config) {

    if (m_mode == MultiGpuMode::Off) {
        Logger::info("Multi-GPU execution disabled (running in Single-GPU mode).");
        return;
    }

    auto devices = VulkanContext::enumeratePhysicalDevices(primaryContext->getInstance());
    if (devices.size() < 2) {
        Logger::warn("Multi-GPU requested but only {} physical Vulkan device(s) found. Falling back to single GPU.", devices.size());
        m_mode = MultiGpuMode::Off;
        return;
    }

    Logger::info("Initializing Multi-GPU Manager across {} discrete AMD GPUs...", devices.size());
    initSecondaryDevice(config, scene);
}

MultiGpuManager::~MultiGpuManager() {
    if (m_asyncTask.valid()) {
        m_asyncTask.wait();
    }
    m_devices.clear();
}

double MultiGpuManager::getSecondaryGpuTimeMs() const {
    if (m_devices.empty()) return 0.0;
    return m_devices[0]->lastFrameTimeMs;
}

const std::string& MultiGpuManager::getSecondaryDeviceName() const {
    static const std::string empty;
    if (m_devices.empty()) return empty;
    return m_devices[0]->deviceName;
}

std::vector<char> MultiGpuManager::loadShaderSPIRV(const std::string& filename) {
    std::vector<std::string> searchPaths = {
        filename,
        std::string("shaders/") + filename,
        std::string(SHADER_DIR) + "/" + filename,
        std::string("build/shaders/") + filename,
        std::string("../build/shaders/") + filename
    };

    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (file.is_open()) {
                size_t fileSize = static_cast<size_t>(file.tellg());
                std::vector<char> buffer(fileSize);
                file.seekg(0);
                file.read(buffer.data(), fileSize);
                return buffer;
            }
        }
    }
    throw std::runtime_error("Could not find compiled SPIR-V file: " + filename);
}

VkShaderModule MultiGpuManager::createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    VkResult res = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create secondary GPU shader module!");
    }
    return shaderModule;
}

void MultiGpuManager::initSecondaryDevice(const Config& config, const SceneData& scene) {
    Config secConfig = config;
    secConfig.gpu_index = 1; // Explicit secondary GPU
    secConfig.headless = true; // Secondary GPU always runs headless compute

    auto secNode = std::make_unique<GpuDeviceNode>();
    secNode->deviceIndex = 1;
    secNode->context = std::make_unique<VulkanContext>(secConfig);
    secNode->deviceName = secNode->context->getDeviceName();
    secNode->timestampPeriod = secNode->context->getDeviceProperties().limits.timestampPeriod;

    VkDevice secDevice = secNode->context->getDevice();
    VmaAllocator secAlloc = secNode->context->getAllocator();

    // 1. Command Pool & Command Buffer
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = secNode->context->getGraphicsQueueFamily();
    vkCreateCommandPool(secDevice, &poolInfo, nullptr, &secNode->commandPool);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = secNode->commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(secDevice, &cmdAllocInfo, &secNode->commandBuffer);

    // 2. Fence & Query Pool
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(secDevice, &fenceInfo, nullptr, &secNode->renderFence);

    VkQueryPoolCreateInfo queryInfo{};
    queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = 2; // Start, End
    vkCreateQueryPool(secDevice, &queryInfo, nullptr, &secNode->queryPool);

    // 3. Render Targets on secondary device
    secNode->accumTarget = std::make_unique<Image>(
        secDevice, secAlloc, config.width, config.height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // Staging buffer for PCIe 5.0 inter-GPU peer transfer
    VkDeviceSize bufferSize = config.width * config.height * 4 * sizeof(float);
    secNode->p2pStagingBuffer = std::make_unique<Buffer>(
        secAlloc, bufferSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
    );

    // 4. Scene Buffers on secondary device
    VkDeviceSize triSize = std::max(sizeof(TriangleGPU) * scene.triangles.size(), sizeof(TriangleGPU));
    secNode->triangleBuffer = std::make_unique<Buffer>(
        secAlloc, triSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.triangles.empty()) {
        secNode->triangleBuffer->copyFrom(scene.triangles.data(), sizeof(TriangleGPU) * scene.triangles.size());
    }

    VkDeviceSize sphereSize = std::max(sizeof(SphereGPU) * scene.spheres.size(), sizeof(SphereGPU));
    secNode->sphereBuffer = std::make_unique<Buffer>(
        secAlloc, sphereSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.spheres.empty()) {
        secNode->sphereBuffer->copyFrom(scene.spheres.data(), sizeof(SphereGPU) * scene.spheres.size());
    }

    VkDeviceSize matSize = std::max(sizeof(MaterialGPU) * scene.materials.size(), sizeof(MaterialGPU));
    secNode->materialBuffer = std::make_unique<Buffer>(
        secAlloc, matSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.materials.empty()) {
        secNode->materialBuffer->copyFrom(scene.materials.data(), sizeof(MaterialGPU) * scene.materials.size());
    }

    VkDeviceSize lightSize = std::max(sizeof(LightGPU) * scene.lights.size(), sizeof(LightGPU));
    secNode->lightBuffer = std::make_unique<Buffer>(
        secAlloc, lightSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.lights.empty()) {
        secNode->lightBuffer->copyFrom(scene.lights.data(), sizeof(LightGPU) * scene.lights.size());
    }

    VkDeviceSize uboSize = sizeof(CameraUniform);
    secNode->cameraUBO = std::make_unique<Buffer>(
        secAlloc, uboSize,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    // Hardware Acceleration Structures on secondary device (VK_KHR_ray_query)
    if (secNode->context->hasRayTracing()) {
        std::vector<Vertex> asVertices;
        if (!scene.triangles.empty()) {
            asVertices.reserve(scene.triangles.size() * 3);
            for (const auto& tri : scene.triangles) {
                asVertices.push_back(tri.v0);
                asVertices.push_back(tri.v1);
                asVertices.push_back(tri.v2);
            }
        } else {
            Vertex v{};
            asVertices.assign(3, v);
        }

        VkDeviceSize vertexBufferSize = sizeof(Vertex) * asVertices.size();
        secNode->asVertexBuffer = std::make_unique<Buffer>(
            secAlloc, vertexBufferSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        );
        secNode->asVertexBuffer->copyFrom(asVertices.data(), vertexBufferSize);

        secNode->asManager = std::make_unique<AccelerationStructureManager>(
            secDevice, secAlloc,
            secNode->context->getGraphicsQueue(), secNode->context->getGraphicsQueueFamily()
        );

        ASGeometryInput geom{};
        geom.vertexBufferAddress = secNode->asVertexBuffer->getDeviceAddress(secDevice);
        geom.indexBufferAddress = 0;
        geom.vertexCount = static_cast<uint32_t>(asVertices.size());
        geom.triangleCount = static_cast<uint32_t>(asVertices.size() / 3);
        geom.vertexStride = sizeof(Vertex);
        geom.indexType = VK_INDEX_TYPE_NONE_KHR;
        geom.isOpaque = true;

        secNode->blas = secNode->asManager->buildBLAS({ geom });

        ASInstanceInput inst{};
        inst.blasAddress = secNode->blas->getDeviceAddress();
        inst.transform = glm::mat4(1.0f);
        inst.customIndex = 0;
        inst.mask = 0xFF;
        inst.hitGroupId = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

        secNode->tlas = secNode->asManager->buildTLAS({ inst });
        Logger::info("Secondary GPU Acceleration Structures initialized successfully (BLAS & TLAS).");
        if (m_config.enable_hardware_rt) {
            Logger::info("Secondary GPU RT Pipeline Active. Extensions in use: VK_KHR_ray_query, VK_KHR_acceleration_structure, VK_KHR_buffer_device_address, VK_KHR_deferred_host_operations");
        } else {
            Logger::info("Secondary GPU Hardware RT is DISABLED via config. Running Software Primitive Traversal (LDS/SSBO). HW RT extensions bypassed.");
        }
    }

    // Textures & HDRI Environment Map on secondary device
    VkQueue secQueue = secNode->context->getGraphicsQueue();
    VkCommandPool secPool = secNode->commandPool;

    secNode->dummyWhite = Texture::createDummyWhite(secDevice, secAlloc, secQueue, secPool);
    secNode->dummyNormal = Texture::createDummyNormal(secDevice, secAlloc, secQueue, secPool);

    if (!config.hdri_path.empty() && std::filesystem::exists(config.hdri_path)) {
        secNode->environmentMap = Texture::loadFromFile(secDevice, secAlloc, secQueue, secPool, config.hdri_path);
    }
    if (!secNode->environmentMap) {
        secNode->environmentMap = Texture::createProceduralHdrSky(secDevice, secAlloc, secQueue, secPool);
    }

    secNode->sceneTextures.clear();
    for (const auto& texData : scene.textures) {
        if (!texData.pixels.empty() && texData.width > 0 && texData.height > 0) {
            auto tex = Texture::createFromPixels(
                secDevice, secAlloc, secQueue, secPool,
                texData.width, texData.height,
                VK_FORMAT_R8G8B8A8_UNORM, texData.pixels.data(),
                texData.pixels.size(), false
            );
            secNode->sceneTextures.push_back(std::move(tex));
        } else {
            secNode->sceneTextures.push_back(Texture::createDummyWhite(secDevice, secAlloc, secQueue, secPool));
        }
    }

    // 5. Descriptor Pool & Sets on secondary device
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 32 }
    };
    VkDescriptorPoolCreateInfo descPoolInfo{};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    descPoolInfo.pPoolSizes = poolSizes.data();
    descPoolInfo.maxSets = 8;
    vkCreateDescriptorPool(secDevice, &descPoolInfo, nullptr, &secNode->descriptorPool);

    std::vector<VkDescriptorSetLayoutBinding> rtBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo rtLayoutInfo{};
    rtLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    rtLayoutInfo.bindingCount = static_cast<uint32_t>(rtBindings.size());
    rtLayoutInfo.pBindings = rtBindings.data();
    vkCreateDescriptorSetLayout(secDevice, &rtLayoutInfo, nullptr, &secNode->rtDescLayout);

    VkDescriptorSetAllocateInfo descAllocInfo{};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = secNode->descriptorPool;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &secNode->rtDescLayout;
    vkAllocateDescriptorSets(secDevice, &descAllocInfo, &secNode->rtDescSet);

    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageView = secNode->accumTarget->getImageView();
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo uboInfo{ secNode->cameraUBO->getBuffer(), 0, sizeof(CameraUniform) };
    VkDescriptorBufferInfo triInfo{ secNode->triangleBuffer->getBuffer(), 0, secNode->triangleBuffer->getSize() };
    VkDescriptorBufferInfo sphereInfo{ secNode->sphereBuffer->getBuffer(), 0, secNode->sphereBuffer->getSize() };
    VkDescriptorBufferInfo matInfo{ secNode->materialBuffer->getBuffer(), 0, secNode->materialBuffer->getSize() };
    VkDescriptorBufferInfo lightInfo{ secNode->lightBuffer->getBuffer(), 0, secNode->lightBuffer->getSize() };

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    VkAccelerationStructureKHR tlasHandle = secNode->tlas ? secNode->tlas->getHandle() : VK_NULL_HANDLE;
    asInfo.pAccelerationStructures = &tlasHandle;

    VkDescriptorImageInfo envInfo = secNode->environmentMap ? secNode->environmentMap->getDescriptorInfo() : secNode->dummyWhite->getDescriptorInfo();

    std::vector<VkDescriptorImageInfo> texInfos(16);
    for (size_t i = 0; i < 16; ++i) {
        if (i < secNode->sceneTextures.size() && secNode->sceneTextures[i]) {
            texInfos[i] = secNode->sceneTextures[i]->getDescriptorInfo();
        } else {
            texInfos[i] = secNode->dummyWhite->getDescriptorInfo();
        }
    }

    std::vector<VkWriteDescriptorSet> writes = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sphereInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, secNode->rtDescSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 8, 0, 16, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texInfos.data(), nullptr, nullptr }
    };
    vkUpdateDescriptorSets(secDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // 6. Pipeline Layout & Compute Pipeline on secondary device
    VkPushConstantRange pushConstant{};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(uint32_t) * 12;

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts = &secNode->rtDescLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges = &pushConstant;
    vkCreatePipelineLayout(secDevice, &pipeLayoutInfo, nullptr, &secNode->rtPipelineLayout);

    auto rtCode = loadShaderSPIRV("raytrace_comp.comp.spv");
    VkShaderModule rtModule = createShaderModule(secDevice, rtCode);

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    VkComputePipelineCreateInfo compPipeInfo{};
    compPipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compPipeInfo.stage.module = rtModule;
    compPipeInfo.stage.pName = "main";
    if (secNode->context->hasSubgroupSizeControl()) {
        compPipeInfo.stage.pNext = &subgroupSize32;
    }
    compPipeInfo.layout = secNode->rtPipelineLayout;
    vkCreateComputePipelines(secDevice, VK_NULL_HANDLE, 1, &compPipeInfo, nullptr, &secNode->rtPipeline);
    vkDestroyShaderModule(secDevice, rtModule, nullptr);

    // 7. Transition secondary accumTarget to GENERAL layout
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(secNode->commandBuffer, &beginInfo);

    secNode->accumTarget->transitionLayout(
        secNode->commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    vkEndCommandBuffer(secNode->commandBuffer);

    VkSubmitInfo initSubmit{};
    initSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    initSubmit.commandBufferCount = 1;
    initSubmit.pCommandBuffers = &secNode->commandBuffer;
    vkQueueSubmit(secNode->context->getGraphicsQueue(), 1, &initSubmit, VK_NULL_HANDLE);
    vkQueueWaitIdle(secNode->context->getGraphicsQueue());

    Logger::info("Secondary GPU Node fully initialized: {} (PCIe 5.0 x16)", secNode->deviceName);
    m_devices.push_back(std::move(secNode));
    m_active = true;
}

void MultiGpuManager::launchSecondaryWork(const CameraUniform& cameraUniform,
                                         uint32_t frameIndex,
                                         uint32_t tileOffsetX, uint32_t tileOffsetY,
                                         uint32_t tileWidth, uint32_t tileHeight,
                                         uint32_t numTriangles, uint32_t numSpheres,
                                         uint32_t numMaterials, uint32_t numLights,
                                         uint32_t useHardwareRT,
                                         uint32_t hasEnvMap,
                                         float envMapIntensity,
                                         uint32_t accumulateHistory) {
    if (!m_active || m_devices.empty()) return;

    m_asyncTask = std::async(std::launch::async, [this, cameraUniform,
                                                 tileOffsetX, tileOffsetY, tileWidth, tileHeight,
                                                 numTriangles, numSpheres, numMaterials, numLights,
                                                 useHardwareRT, hasEnvMap, envMapIntensity, accumulateHistory]() {
        GpuDeviceNode* node = m_devices[0].get();
        VkDevice device = node->context->getDevice();
        VkQueue queue = node->context->getGraphicsQueue();

        // 1. Update Camera UBO on secondary GPU
        node->cameraUBO->copyFrom(&cameraUniform, sizeof(CameraUniform));

        // 2. Wait for previous execution to complete
        vkWaitForFences(device, 1, &node->renderFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &node->renderFence);

        // 3. Record secondary GPU commands
        vkResetCommandBuffer(node->commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(node->commandBuffer, &beginInfo);

        vkCmdResetQueryPool(node->commandBuffer, node->queryPool, 0, 2);
        vkCmdWriteTimestamp2(node->commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, node->queryPool, 0);

        // Bind RT Pipeline
        vkCmdBindPipeline(node->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, node->rtPipeline);
        vkCmdBindDescriptorSets(node->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, node->rtPipelineLayout, 0, 1, &node->rtDescSet, 0, nullptr);

        uint32_t envBits = std::bit_cast<uint32_t>(envMapIntensity);
        uint32_t rtPushConstants[12] = {
            numTriangles, numSpheres, numMaterials, numLights,
            tileOffsetX, tileOffsetY, tileWidth, tileHeight,
            useHardwareRT,
            hasEnvMap,
            envBits,
            accumulateHistory
        };
        vkCmdPushConstants(node->commandBuffer, node->rtPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rtPushConstants), rtPushConstants);

        uint32_t groupsX = (tileWidth + 15) / 16;
        uint32_t groupsY = (tileHeight + 15) / 16;
        vkCmdDispatch(node->commandBuffer, groupsX, groupsY, 1);

        vkCmdWriteTimestamp2(node->commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, node->queryPool, 1);

        // Transition accumTarget to TRANSFER_SRC_OPTIMAL to copy to host-visible staging buffer
        node->accumTarget->transitionLayout(
            node->commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
        );

        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageOffset = { static_cast<int32_t>(tileOffsetX), static_cast<int32_t>(tileOffsetY), 0 };
        copyRegion.imageExtent = { tileWidth, tileHeight, 1 };

        vkCmdCopyImageToBuffer(node->commandBuffer, node->accumTarget->getImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               node->p2pStagingBuffer->getBuffer(), 1, &copyRegion);

        node->accumTarget->transitionLayout(
            node->commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        vkEndCommandBuffer(node->commandBuffer);

        // 4. Submit secondary workload
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &node->commandBuffer;
        vkQueueSubmit(queue, 1, &submitInfo, node->renderFence);

        // 5. Wait for secondary fence and retrieve timestamps
        vkWaitForFences(device, 1, &node->renderFence, VK_TRUE, UINT64_MAX);

        uint64_t timestamps[2] = {0, 0};
        vkGetQueryPoolResults(device, node->queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        node->lastFrameTimeMs = (timestamps[1] - timestamps[0]) * node->timestampPeriod * 1e-6;
    });
}

void MultiGpuManager::syncAndTransfer(void* dstHostPtr, size_t byteSize) {
    if (!m_active || m_devices.empty()) return;

    if (m_asyncTask.valid()) {
        m_asyncTask.get();
    }

    GpuDeviceNode* node = m_devices[0].get();
    void* srcPtr = node->p2pStagingBuffer->map();
    std::memcpy(dstHostPtr, srcPtr, byteSize);
    node->p2pStagingBuffer->unmap();
}

void MultiGpuManager::resize(uint32_t width, uint32_t height) {
    if (!m_active || m_devices.empty()) return;

    if (m_asyncTask.valid()) {
        m_asyncTask.wait();
    }

    m_config.width = width;
    m_config.height = height;

    for (auto& node : m_devices) {
        VkDevice secDevice = node->context->getDevice();
        VmaAllocator secAlloc = node->context->getAllocator();
        vkDeviceWaitIdle(secDevice);

        node->accumTarget = std::make_unique<Image>(
            secDevice, secAlloc, width, height,
            VK_FORMAT_R32G32B32A32_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );

        // Transition accumTarget to GENERAL layout
        vkResetCommandBuffer(node->commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(node->commandBuffer, &beginInfo);

        node->accumTarget->transitionLayout(
            node->commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        vkEndCommandBuffer(node->commandBuffer);

        VkSubmitInfo initSubmit{};
        initSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        initSubmit.commandBufferCount = 1;
        initSubmit.pCommandBuffers = &node->commandBuffer;
        vkQueueSubmit(node->context->getGraphicsQueue(), 1, &initSubmit, VK_NULL_HANDLE);
        vkQueueWaitIdle(node->context->getGraphicsQueue());

        VkDeviceSize bufferSize = static_cast<VkDeviceSize>(width) * height * 4 * sizeof(float);
        node->p2pStagingBuffer = std::make_unique<Buffer>(
            secAlloc, bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
        );

        VkDescriptorImageInfo accumImageInfo{};
        accumImageInfo.imageView = node->accumTarget->getImageView();
        accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = node->rtDescSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo = &accumImageInfo;

        vkUpdateDescriptorSets(secDevice, 1, &write, 0, nullptr);
    }

    Logger::info("MultiGpuManager resized secondary GPU targets to {}x{}", width, height);
}

} // namespace pathways
