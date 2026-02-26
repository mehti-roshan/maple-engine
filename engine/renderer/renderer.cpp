#include <engine/file/file.h>
#include <engine/logging/log_macros.h>
#include <engine/renderer/renderer.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "engine/renderer/vk_buffer.h"
#include "engine/renderer/vk_device_features.h"
#include "engine/renderer/vk_instance_ctx.h"
#include "engine/renderer/vk_logical_device.h"
#include "engine/renderer/vk_sampler.h"
#include "engine/renderer/vk_swapchain.h"
#include "engine/renderer/vk_texture.h"

#define GLM_FORCE_RADIANS
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <engine/third_party/stb_image.h>

#define TINYOBJLOADER_IMPLEMENTATION
#include <engine/third_party/tiny_obj_loader.h>

#include "vk_physical_device.h"

#ifdef NDEBUG
constexpr bool debug = false;
#else
constexpr bool debug = true;
#endif

[[nodiscard]] vk::raii::ShaderModule createShaderModule(vk::raii::Device& device, const std::vector<char>& code) {
  vk::ShaderModuleCreateInfo createInfo{
    .codeSize = code.size(),
    .pCode = reinterpret_cast<const uint32_t*>(code.data()),
  };

  return vk::raii::ShaderModule(device, createInfo);
}

void transition_image_layout(vk::Image image,
                             vk::CommandBuffer commandBuffer,
                             vk::ImageLayout oldLayout,
                             vk::ImageLayout newLayout,
                             vk::AccessFlags2 srcAccessMask,
                             vk::AccessFlags2 dstAccessMask,
                             vk::PipelineStageFlags2 srcStageMask,
                             vk::PipelineStageFlags2 dstStageMask,
                             vk::ImageAspectFlags aspectFlags) {
  vk::ImageMemoryBarrier2 barrier = {
    .srcStageMask = srcStageMask,
    .srcAccessMask = srcAccessMask,
    .dstStageMask = dstStageMask,
    .dstAccessMask = dstAccessMask,
    .oldLayout = oldLayout,
    .newLayout = newLayout,
    .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
    .image = image,
    .subresourceRange = {.aspectMask = aspectFlags, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
  vk::DependencyInfo dependencyInfo = {.dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
  commandBuffer.pipelineBarrier2(dependencyInfo);
}

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

namespace maple {

static std::vector<const char*> requiredDeviceExtensions = {vk::KHRSwapchainExtensionName};
auto requiredFeatures = DeviceFeature::SamplerAnisotropy | DeviceFeature::ShaderDrawParameters | DeviceFeature::Synchronization2 |
  DeviceFeature::DynamicRendering | DeviceFeature::ExtendedDynamicState;

void Renderer::Init(const std::vector<const char*>& glfwExtensions, SurfaceCreateCallback surfaceCallback, FrameBufferSizeCallback fbCallback) {
  mFrameBufferSizeCallback = fbCallback;
  mInstanceCtx = std::move(VulkanInstanceContext(glfwExtensions, debug));
  mSurface = vk::raii::SurfaceKHR(mInstanceCtx.mInstance, surfaceCallback(*mInstanceCtx.mInstance));

  mPhysicalDevice = VulkanPhysicalDevice(VulkanPhysicalDevice::CreateInfo{
    .surface = mSurface,
    .availableDevices = mInstanceCtx.mInstance.enumeratePhysicalDevices(),
    .requiredDeviceExtensions = requiredDeviceExtensions,
    .requiredFeatureMask = requiredFeatures,
  });

  mDevice = VulkanLogicalDevice(VulkanLogicalDevice::CreateInfo{
    .physicalDevice = mPhysicalDevice,
    .requiredDeviceExtensions = requiredDeviceExtensions,
    .requiredFeatures = requiredFeatures,
  });

  mMemoryManager = VulkanMemoryManager(mInstanceCtx.mInstance, mPhysicalDevice.device, &mDevice.device);
  mSwapChain = VulkanSwapChain({.physicalDevice = mPhysicalDevice,
                                .device = mDevice,
                                .surface = mSurface,
                                .memoryManager = mMemoryManager,
                                .framebufferSizeCb = mFrameBufferSizeCallback});
  createDescriptorSetLayout();
  createGraphicsPipeline();
  createCommandPools();
  createFrameData();
  createTexture();
  createTextureSampler();
  createMeshBuffer();
  createUniformBuffers();
  createDescriptorPool();
  createDescriptorSets();
}

void Renderer::DrawFrame() {
  auto fenceResult = mDevice.device.waitForFences(*mFrameData[mFrameIdx].drawFence, vk::True, UINT64_MAX);
  auto [result, imageIdx] = mSwapChain.swapchain.acquireNextImage(UINT64_MAX, mFrameData[mFrameIdx].presentCompleteSem, nullptr);
  mDevice.device.resetFences(*mFrameData[mFrameIdx].drawFence);

  mFrameData[mFrameIdx].cmd.reset();

  updateUniformBuffer(mFrameIdx);
  recordCommandBuffer(imageIdx);

  vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
  const vk::SubmitInfo submitInfo{
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &*mFrameData[mFrameIdx].presentCompleteSem,
    .pWaitDstStageMask = &waitDestinationStageMask,
    .commandBufferCount = 1,
    .pCommandBuffers = &*mFrameData[mFrameIdx].cmd,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &*mRenderCompleteSems[imageIdx],
  };

  mDevice.queues.graphics.submit(submitInfo, *mFrameData[mFrameIdx].drawFence);

  const vk::PresentInfoKHR presentInfoKHR{
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &*mRenderCompleteSems[imageIdx],
    .swapchainCount = 1,
    .pSwapchains = &*mSwapChain.swapchain,
    .pImageIndices = &imageIdx,
  };

  auto presentResult = mDevice.queues.present.presentKHR(presentInfoKHR);

  if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || mFrameBufferResized) {
    mFrameBufferResized = false;
    mSwapChain.ReCreate({.physicalDevice = mPhysicalDevice,
                         .device = mDevice,
                         .surface = mSurface,
                         .memoryManager = mMemoryManager,
                         .framebufferSizeCb = mFrameBufferSizeCallback});
  } else if (presentResult != vk::Result::eSuccess) {
    MAPLE_FATAL("Failed to present swap chain image");
  }

  mFrameIdx = (mFrameIdx + 1) % MAX_FRAMES_IN_FLIGHT;
}

void Renderer::updateUniformBuffer(uint32_t currentImage) {
  static auto startTime = std::chrono::high_resolution_clock::now();

  auto currentTime = std::chrono::high_resolution_clock::now();
  float time = std::chrono::duration<float, std::chrono::seconds::period>(currentTime - startTime).count();

  UniformBufferObject ubo{
    .model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    .view = glm::lookAt(glm::vec3(2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    .proj = glm::perspective(
      glm::radians(45.0f), static_cast<float>(mSwapChain.extent.width) / static_cast<float>(mSwapChain.extent.height), 0.1f, 2000.0f)};
  ubo.proj[1][1] *= -1;  // Invert Y for Vulkan

  mUniformBuffers[currentImage].Upload(&ubo, sizeof(ubo));
}

void Renderer::createDescriptorSetLayout() {
  std::array bindings = {
    vk::DescriptorSetLayoutBinding{
      .binding = 0,
      .descriptorType = vk::DescriptorType::eUniformBuffer,
      .descriptorCount = 1,
      .stageFlags = vk::ShaderStageFlagBits::eVertex,
    },
    vk::DescriptorSetLayoutBinding{
      .binding = 1,
      .descriptorType = vk::DescriptorType::eCombinedImageSampler,
      .descriptorCount = 1,
      .stageFlags = vk::ShaderStageFlagBits::eFragment,
    },
  };

  vk::DescriptorSetLayoutCreateInfo layoutInfo{.bindingCount = bindings.size(), .pBindings = bindings.data()};
  mDescriptorSetLayout = vk::raii::DescriptorSetLayout(mDevice.device, layoutInfo);
}

void Renderer::createGraphicsPipeline() {
  auto shaderModule = createShaderModule(mDevice.device, file::ReadFile("assets/shaders/slang.spv"));

  vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
  vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
  vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

  std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

  vk::PipelineDynamicStateCreateInfo dynamicState{
    .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
    .pDynamicStates = dynamicStates.data(),
  };

  auto bindingDescription = mMesh.GetBindingDescription();
  auto attributeDescriptions = mMesh.GetAttributeDescriptions();
  vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
    .vertexBindingDescriptionCount = 1,
    .pVertexBindingDescriptions = &bindingDescription,
    .vertexAttributeDescriptionCount = static_cast<uint32_t>(attributeDescriptions.size()),
    .pVertexAttributeDescriptions = attributeDescriptions.data(),
  };

  vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};

  vk::Viewport viewport{0.0f, 0.0f, static_cast<float>(mSwapChain.extent.width), static_cast<float>(mSwapChain.extent.height), 0.0f, 1.0f};
  vk::Rect2D scissor{{0, 0}, mSwapChain.extent};
  vk::PipelineViewportStateCreateInfo viewportState{
    .viewportCount = 1,
    .pViewports = &viewport,
    .scissorCount = 1,
    .pScissors = &scissor,
  };

  vk::PipelineRasterizationStateCreateInfo rasterizer{
    .depthClampEnable = vk::False,
    .rasterizerDiscardEnable = vk::False,
    .polygonMode = vk::PolygonMode::eFill,
    .cullMode = vk::CullModeFlagBits::eBack,
    .frontFace = vk::FrontFace::eCounterClockwise,
    .depthBiasEnable = vk::False,
    .depthBiasSlopeFactor = 1.0f,
    .lineWidth = 1.0f,
  };

  vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

  vk::PipelineDepthStencilStateCreateInfo depthStencil{
    .depthTestEnable = vk::True,
    .depthWriteEnable = vk::True,
    .depthCompareOp = vk::CompareOp::eLess,
    .depthBoundsTestEnable = vk::False,
    .stencilTestEnable = vk::False,
  };

  vk::PipelineColorBlendAttachmentState colorBlendAttachment{
    .blendEnable = vk::False,
    .colorWriteMask =
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };

  vk::PipelineColorBlendStateCreateInfo colorBlending{
    .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

  vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount = 1, .pSetLayouts = &*mDescriptorSetLayout, .pushConstantRangeCount = 0};
  mPipelineLayout = vk::raii::PipelineLayout(mDevice.device, pipelineLayoutInfo);

  vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{
    .colorAttachmentCount = 1, .pColorAttachmentFormats = &mSwapChain.format.format, .depthAttachmentFormat = mSwapChain.depthFormat};
  vk::GraphicsPipelineCreateInfo pipelineInfo{
    .pNext = &pipelineRenderingCreateInfo,
    .stageCount = 2,
    .pStages = shaderStages,
    .pVertexInputState = &vertexInputInfo,
    .pInputAssemblyState = &inputAssembly,
    .pViewportState = &viewportState,
    .pRasterizationState = &rasterizer,
    .pMultisampleState = &multisampling,
    .pDepthStencilState = &depthStencil,
    .pColorBlendState = &colorBlending,
    .pDynamicState = &dynamicState,
    .layout = mPipelineLayout,
    .renderPass = nullptr,
  };

  mGraphicsPipeline = vk::raii::Pipeline(mDevice.device, nullptr, pipelineInfo);
}

void Renderer::createCommandPools() {
  auto qIndices = mPhysicalDevice.queueFamilyIndices;

  vk::CommandPoolCreateInfo poolInfo{};
  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = qIndices.graphics;

  mCommandPools.graphics = vk::raii::CommandPool(mDevice.device, poolInfo);

  poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
  poolInfo.queueFamilyIndex = qIndices.transfer;
  mCommandPools.transfer = vk::raii::CommandPool(mDevice.device, poolInfo);
}

void Renderer::createTexture() {
  int32_t texWidth, texHeight, texChannels;
  stbi_uc* pixels = stbi_load("assets/textures/viking_room.png", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
  vk::DeviceSize imageSize = texWidth * texHeight * 4;

  if (!pixels) MAPLE_FATAL("failed to load texture image");

  auto stage = mMemoryManager.createBuffer(imageSize,
                                           vk::BufferUsageFlagBits::eTransferSrc,
                                           VMA_MEMORY_USAGE_AUTO,
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  stage.Upload(pixels, stage.size);
  stbi_image_free(pixels);

  mTexture = mMemoryManager.createTexture({static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1},
                                          vk::Format::eR8G8B8A8Srgb,
                                          vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
                                          vk::ImageAspectFlagBits::eColor);

  auto cmd = beginSingleTimeCommands();
  mTexture.image.TransitionLayout(cmd, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
  mTexture.image.UploadBuffer(cmd, stage, texWidth, texHeight);
  mTexture.image.TransitionLayout(cmd, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
  endSingleTimeCommands(cmd);
}

void Renderer::createTextureSampler() {
  vk::SamplerCreateInfo samplerInfo{
    .magFilter = vk::Filter::eLinear,
    .minFilter = vk::Filter::eLinear,
    .mipmapMode = vk::SamplerMipmapMode::eLinear,
    .addressModeU = vk::SamplerAddressMode::eRepeat,
    .addressModeV = vk::SamplerAddressMode::eRepeat,
    .addressModeW = vk::SamplerAddressMode::eRepeat,
    .mipLodBias = 0.0f,
    .anisotropyEnable = vk::True,
    .maxAnisotropy = mPhysicalDevice.device.getProperties().limits.maxSamplerAnisotropy,
    .compareEnable = vk::False,
    .compareOp = vk::CompareOp::eAlways,
    .minLod = 0.0f,
    .maxLod = 1.0f,
    .borderColor = vk::BorderColor::eIntOpaqueBlack,
    .unnormalizedCoordinates = vk::False,
  };

  mSampler = VulkanSampler(*mDevice.device, samplerInfo);
}

void Renderer::createMeshBuffer() {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn, err;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "assets/models/viking_room.obj"))
    MAPLE_FATAL("Failed to load obj file: {} {}", warn, err);

  std::unordered_map<Vertex, uint32_t> uniqueVerts{};

  for (const auto& shape : shapes) {
    for (const auto& idx : shape.mesh.indices) {
      Vertex v{
        .pos =
          {
            attrib.vertices[3 * idx.vertex_index + 0],
            attrib.vertices[3 * idx.vertex_index + 1],
            attrib.vertices[3 * idx.vertex_index + 2],
          },
        .color = {1.0f, 1.0f, 1.0f},
        .texCoord =
          {
            attrib.texcoords[2 * idx.texcoord_index + 0],
            1.0f - attrib.texcoords[2 * idx.texcoord_index + 1],
          },
      };

      if (uniqueVerts.count(v) == 0) {
        uniqueVerts[v] = static_cast<uint32_t>(mMesh.vertices.size());
        mMesh.vertices.push_back(v);
      }
      mMesh.indices.push_back(uniqueVerts[v]);
    }
  }

  auto stage = mMemoryManager.createBuffer(mMesh.GetTotalSizeBytes(),
                                           vk::BufferUsageFlagBits::eTransferSrc,
                                           VMA_MEMORY_USAGE_AUTO,
                                           VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
  stage.Upload(mMesh.vertices.data(), mMesh.GetVerticesSizeBytes());
  stage.Upload(mMesh.indices.data(), mMesh.GetIndicesSizeBytes(), mMesh.GetVerticesSizeBytes());

  mMeshBuffer = mMemoryManager.createBuffer(
    stage.size,
    vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eIndexBuffer,
    VMA_MEMORY_USAGE_AUTO,
    0);

  copyBuffer(stage.buffer, mMeshBuffer.buffer, stage.size);
}

void Renderer::createUniformBuffers() {
  mUniformBuffers.clear();

  vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    // With the correct flags vma will automatically map the memory and persist it, no need to map once and store the pointers
    mUniformBuffers.push_back(mMemoryManager.createBuffer(bufferSize,
                                                          vk::BufferUsageFlagBits::eUniformBuffer,
                                                          VMA_MEMORY_USAGE_AUTO,
                                                          VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT));
  }
}

void Renderer::createDescriptorPool() {
  std::array poolSize = {
    vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT),
    vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT),
  };
  vk::DescriptorPoolCreateInfo poolInfo{.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet,
                                        .maxSets = MAX_FRAMES_IN_FLIGHT,
                                        .poolSizeCount = poolSize.size(),
                                        .pPoolSizes = poolSize.data()};
  mDescriptorPool = vk::raii::DescriptorPool(mDevice.device, poolInfo);
}

void Renderer::createDescriptorSets() {
  std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *mDescriptorSetLayout);
  vk::DescriptorSetAllocateInfo allocInfo{
    .descriptorPool = mDescriptorPool, .descriptorSetCount = static_cast<uint32_t>(layouts.size()), .pSetLayouts = layouts.data()};
  mDescriptorSets = vk::raii::DescriptorSets(mDevice.device, allocInfo);
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::DescriptorBufferInfo bufferInfo{.buffer = mUniformBuffers[i].buffer, .offset = 0, .range = sizeof(UniformBufferObject)};
    vk::DescriptorImageInfo imageInfo{
      .sampler = mSampler.sampler,
      .imageView = mTexture.view,
      .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    std::array descriptorWrites = {
      vk::WriteDescriptorSet{
        .dstSet = mDescriptorSets[i],
        .dstBinding = 0,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eUniformBuffer,
        .pBufferInfo = &bufferInfo,
      },
      vk::WriteDescriptorSet{
        .dstSet = mDescriptorSets[i],
        .dstBinding = 1,
        .dstArrayElement = 0,
        .descriptorCount = 1,
        .descriptorType = vk::DescriptorType::eCombinedImageSampler,
        .pImageInfo = &imageInfo,
      },
    };

    mDevice.device.updateDescriptorSets(descriptorWrites, {});
  }
}

void Renderer::recordCommandBuffer(uint32_t imageIdx) {
  mFrameData[mFrameIdx].cmd.begin({});

  // Color attachment transition
  transition_image_layout(mSwapChain.images[imageIdx].img,
                          mFrameData[mFrameIdx].cmd,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eColorAttachmentOptimal,
                          {},
                          vk::AccessFlagBits2::eColorAttachmentWrite,
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                          vk::ImageAspectFlagBits::eColor);

  // Depth attachment transition
  transition_image_layout(mSwapChain.depthTexture.image.image,
                          mFrameData[mFrameIdx].cmd,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eDepthStencilAttachmentOptimal,
                          vk::AccessFlagBits2::eDepthStencilAttachmentRead,
                          vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                          vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                          vk::PipelineStageFlagBits2::eEarlyFragmentTests | vk::PipelineStageFlagBits2::eLateFragmentTests,
                          vk::ImageAspectFlagBits::eDepth);

  vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
  vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
  vk::RenderingAttachmentInfo depthAttachmentInfo = {
    .imageView = mSwapChain.depthTexture.view,
    .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
    .loadOp = vk::AttachmentLoadOp::eClear,
    .storeOp = vk::AttachmentStoreOp::eDontCare,
    .clearValue = clearDepth,
  };

  vk::RenderingAttachmentInfo colorAttachmentInfo = {
    .imageView = mSwapChain.images[imageIdx].view,
    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
    .loadOp = vk::AttachmentLoadOp::eClear,
    .storeOp = vk::AttachmentStoreOp::eStore,
    .clearValue = clearColor,
  };

  vk::RenderingInfo renderingInfo = {
    .renderArea = {.offset = {0, 0}, .extent = mSwapChain.extent},
    .layerCount = 1,
    .colorAttachmentCount = 1,
    .pColorAttachments = &colorAttachmentInfo,
    .pDepthAttachment = &depthAttachmentInfo,
  };

  mFrameData[mFrameIdx].cmd.beginRendering(renderingInfo);

  mFrameData[mFrameIdx].cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, mGraphicsPipeline);

  mFrameData[mFrameIdx].cmd.setViewport(
    0, vk::Viewport{0.0f, 0.0f, static_cast<float>(mSwapChain.extent.width), static_cast<float>(mSwapChain.extent.height), 0.0f, 1.0f});
  mFrameData[mFrameIdx].cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), mSwapChain.extent));

  mFrameData[mFrameIdx].cmd.bindVertexBuffers(0, {mMeshBuffer.buffer}, {0});
  mFrameData[mFrameIdx].cmd.bindIndexBuffer(mMeshBuffer.buffer, mMesh.GetVerticesSizeBytes(), mMesh.GetVkIndexType());
  mFrameData[mFrameIdx].cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, mPipelineLayout, 0, *mDescriptorSets[mFrameIdx], nullptr);
  mFrameData[mFrameIdx].cmd.drawIndexed(mMesh.indices.size(), 1, 0, 0, 0);

  mFrameData[mFrameIdx].cmd.endRendering();

  transition_image_layout(mSwapChain.images[imageIdx].img,
                          mFrameData[mFrameIdx].cmd,
                          vk::ImageLayout::eColorAttachmentOptimal,
                          vk::ImageLayout::ePresentSrcKHR,
                          vk::AccessFlagBits2::eColorAttachmentWrite,
                          {},
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                          vk::PipelineStageFlagBits2::eBottomOfPipe,
                          vk::ImageAspectFlagBits::eColor);
  mFrameData[mFrameIdx].cmd.end();
}

void Renderer::createFrameData() {
  mFrameData.clear();
  mFrameData.resize(MAX_FRAMES_IN_FLIGHT);

  vk::CommandBufferAllocateInfo allocInfo{
    .commandPool = mCommandPools.graphics,
    .level = vk::CommandBufferLevel::ePrimary,
    .commandBufferCount = MAX_FRAMES_IN_FLIGHT,
  };

  vk::raii::CommandBuffers cmdBuffers(mDevice.device, allocInfo);

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    mFrameData[i].cmd = std::move(cmdBuffers[i]);
    mFrameData[i].presentCompleteSem = vk::raii::Semaphore(mDevice.device, vk::SemaphoreCreateInfo{});
    mFrameData[i].drawFence = vk::raii::Fence(mDevice.device, {.flags = vk::FenceCreateFlagBits::eSignaled});
  }

  mRenderCompleteSems.clear();
  mRenderCompleteSems.reserve(mSwapChain.images.size());
  for (size_t i = 0; i < mSwapChain.images.size(); i++) mRenderCompleteSems.emplace_back(mDevice.device, vk::SemaphoreCreateInfo{});
}

}  // namespace maple