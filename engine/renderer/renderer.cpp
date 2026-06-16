#include <engine/file/file.h>
#include <engine/logging/log_macros.h>
#include <engine/renderer/renderer.h>

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <glm/geometric.hpp>
#include <glm/trigonometric.hpp>
#include <unordered_map>
#include <vector>
#include <vulkan/vulkan_enums.hpp>

#include "engine/renderer/vk_device_features.h"
#include "engine/renderer/vk_instance_ctx.h"
#include "engine/renderer/vk_logical_device.h"
#include "engine/renderer/vkm/vkm_allocator.h"
#include "engine/renderer/vkm/vkm_descriptor_pool.h"
#include "engine/renderer/vkm/vkm_descriptor_sets.h"
#include "engine/renderer/vkm/vkm_mesh.h"
#include "engine/renderer/vkm/vkm_pipeline.h"
#include "engine/renderer/vkm/vkm_sampler.h"

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
#include "vkh/vkh_transition_image_layout.h"

#ifdef NDEBUG
constexpr bool debug = false;
#else
constexpr bool debug = true;
#endif

size_t numInstances = 10;

struct UniformBufferObject {
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

  mAllocator = vkm::Allocator(mDevice.device, mPhysicalDevice.device);
  mSwapChain = VulkanSwapChain({.physicalDevice = mPhysicalDevice,
                                .device = mDevice,
                                .surface = mSurface,
                                .allocator = mAllocator,
                                .framebufferSizeCb = mFrameBufferSizeCallback});

  createCommandPools();
  createFrameData();
  createMeshBuffer();
  createUniformBuffers();
  createTexture();
  createTextureSampler();
  createDescriptorPool();
  createDescriptorSets();
  mPipeline = vkm::Pipeline(vkm::Pipeline::CreateInfo{
    .shaderFile = "assets/shaders/slang.spv",
    .device = mDevice.device,
    .swapChainData =
      {
        .swapchain = mSwapChain.swapchain,
        .format = mSwapChain.format,
        .depthFormat = mSwapChain.depthFormat,
      },
    .descriptorSetLayout = mDescriptorSets.layout,
    .vertexLayoutDescription =
      {
        vkm::Vertex::GetVertInputBindDesc(),
        vkm::Vertex::GetVertAttribDescriptions(),
      },
  });
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
    mSwapChain.ReCreate({
      .physicalDevice = mPhysicalDevice,
      .device = mDevice,
      .surface = mSurface,
      .allocator = mAllocator,
      .framebufferSizeCb = mFrameBufferSizeCallback,
    });
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
    .view = glm::lookAt(glm::vec3(2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 0.0f, 1.0f)),
    .proj = glm::perspective(
      glm::radians(70.0f), static_cast<float>(mSwapChain.extent.width) / static_cast<float>(mSwapChain.extent.height), 0.1f, 2000.0f),
  };
  ubo.proj[1][1] *= -1;  // Invert Y for Vulkan

  mUniformBuffers[currentImage].Upload(&ubo, sizeof(ubo));

  std::vector<glm::mat4> modelMatrices(numInstances, glm::mat4(1.0f));
  for (auto& mat : modelMatrices) {
    float amount = 2.0f;
    mat = glm::translate(mat,
                         glm::vec3(float(rand()) / float(RAND_MAX) * amount - (amount / 2.0f),
                                   float(rand()) / float(RAND_MAX) * amount - (amount / 2.0f),
                                   float(rand()) / float(RAND_MAX) * amount - (amount / 2.0f)));

    float deg = float(rand()) / float(RAND_MAX) * 360.0f;
    glm::vec3 axis = glm::normalize(glm::vec3(0, 0, 1.0f));
    mat = glm::rotate(mat, glm::radians(time * 4.f), axis);
  }

  mInstanceDataSSBOs[currentImage].Upload(modelMatrices.data(), modelMatrices.size() * sizeof(decltype(modelMatrices)::value_type));
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

  auto stage = mAllocator.CreateBuffer(imageSize, vkm::Allocator::BufType::Stage);
  stage.Upload(pixels, imageSize);
  stbi_image_free(pixels);

  mImage = mAllocator.CreateImage({
    .extent = {static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1},
    .usage = vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
  });

  auto cmd = beginSingleTimeCommands();
  mImage.TransitionLayout(cmd, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
  mImage.UploadBuffer(cmd, stage.buffer, texWidth, texHeight);
  mImage.TransitionLayout(cmd, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
  endSingleTimeCommands(cmd);

  // auto stage = mMemoryManager.createBuffer(imageSize,
  //                                          vk::BufferUsageFlagBits::eTransferSrc,
  //                                          VMA_MEMORY_USAGE_AUTO,
  //                                          VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);

  // stage.Upload(pixels, stage.size);
  // stbi_image_free(pixels);

  // mTexture = mMemoryManager.createTexture({static_cast<uint32_t>(texWidth), static_cast<uint32_t>(texHeight), 1},
  //                                         vk::Format::eR8G8B8A8Srgb,
  //                                         vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
  //                                         vk::ImageAspectFlagBits::eColor);

  // auto cmd = beginSingleTimeCommands();
  // mTexture.image.TransitionLayout(cmd, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
  // mTexture.image.UploadBuffer(cmd, stage, texWidth, texHeight);
  // mTexture.image.TransitionLayout(cmd, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
  // endSingleTimeCommands(cmd);
}

void Renderer::createTextureSampler() {
  mSampler = vkm::Sampler(mDevice.device, {.maxAnisotropy = mPhysicalDevice.device.getProperties().limits.maxSamplerAnisotropy});
}

void Renderer::createMeshBuffer() {
  tinyobj::attrib_t attrib;
  std::vector<tinyobj::shape_t> shapes;
  std::vector<tinyobj::material_t> materials;
  std::string warn, err;

  if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err, "assets/models/viking_room.obj"))
    MAPLE_FATAL("Failed to load obj file: {} {}", warn, err);

  std::unordered_map<vkm::Vertex, uint32_t> uniqueVerts{};

  for (const auto& shape : shapes) {
    for (const auto& idx : shape.mesh.indices) {
      vkm::Vertex v{
        .pos =
          {
            attrib.vertices[3 * idx.vertex_index + 0],
            attrib.vertices[3 * idx.vertex_index + 1],
            attrib.vertices[3 * idx.vertex_index + 2],
          },
        .uv =
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

  auto stage = mAllocator.CreateBuffer(mMesh.GetTotalSizeBytes(), vkm::Allocator::BufType::Stage);
  stage.Upload(mMesh.vertices.data(), mMesh.GetVerticesSizeBytes());
  stage.Upload(mMesh.indices.data(), mMesh.GetIndicesSizeBytes(), mMesh.GetVerticesSizeBytes());

  mMeshBuffer = mAllocator.CreateBuffer(stage.size, vkm::Allocator::BufType::Mesh);

  auto cmd = beginSingleTimeCommands();
  stage.CopyToBuffer(cmd, mMeshBuffer.buffer, {.size = stage.size});
  endSingleTimeCommands(cmd);
}

void Renderer::createUniformBuffers() {
  mUniformBuffers.clear();

  vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++)
    // With the correct flags vma will automatically map the memory and persist it, no need to map once and store the pointers
    mUniformBuffers.push_back(mAllocator.CreateBuffer(bufferSize, vkm::Allocator::BufType::UBO));

  mInstanceDataSSBOs.clear();
  vk::DeviceSize bufSize = sizeof(glm::mat4) * numInstances;
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) mInstanceDataSSBOs.push_back(mAllocator.CreateBuffer(bufSize, vkm::Allocator::BufType::SSBO));
}

void Renderer::createDescriptorPool() {
  mDescriptorPool = vkm::DescriptorPool(vkm::DescriptorPool::CreateInfo{
    .device = mDevice.device,
    .maxSets = MAX_FRAMES_IN_FLIGHT,
    .resourceSizes =
      {
        {vk::DescriptorType::eUniformBuffer, MAX_FRAMES_IN_FLIGHT},
        {vk::DescriptorType::eCombinedImageSampler, MAX_FRAMES_IN_FLIGHT},
        {vk::DescriptorType::eStorageBuffer, MAX_FRAMES_IN_FLIGHT},
      },
    .freeDescriptorSet = true,
  });
}

void Renderer::createDescriptorSets() {
  mDescriptorSets = vkm::DescriptorSets(vkm::DescriptorSets::CreateInfo{
    .device = mDevice.device,
    .pool = mDescriptorPool.pool,
    .count = MAX_FRAMES_IN_FLIGHT,
    .description =
      {
        vkm::DescriptorSets::Layout{
          .bindingSlot = 0, .type = vkm::DescriptorSets::Type::Uniform, .arrayCount = 1, .usedStages = vk::ShaderStageFlagBits::eVertex},
        vkm::DescriptorSets::Layout{.bindingSlot = 1,
                                    .type = vkm::DescriptorSets::Type::CombinedImageSampler,
                                    .arrayCount = 1,
                                    .usedStages = vk::ShaderStageFlagBits::eFragment},
        vkm::DescriptorSets::Layout{
          .bindingSlot = 2, .type = vkm::DescriptorSets::Type::SSBO, .arrayCount = 1, .usedStages = vk::ShaderStageFlagBits::eVertex},
      },
  });

  for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::DescriptorBufferInfo uniformBufferInfo{.buffer = mUniformBuffers[i].buffer, .offset = 0, .range = sizeof(UniformBufferObject)};
    vk::DescriptorBufferInfo instanceDataBufInfo{.buffer = mInstanceDataSSBOs[i].buffer, .offset = 0, .range = sizeof(glm::mat4) * numInstances};

    vk::DescriptorImageInfo imageInfo{
      .sampler = mSampler.sampler,
      .imageView = mImage.view,
      .imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal,
    };

    mDescriptorSets.UpdateDescriptorSets(mDevice.device,
                                         {
                                           vkm::DescriptorSets::UpdateData{i, 0, 0, 1, &uniformBufferInfo, nullptr},
                                           vkm::DescriptorSets::UpdateData{i, 1, 0, 1, nullptr, &imageInfo},
                                           vkm::DescriptorSets::UpdateData{i, 2, 0, 1, &instanceDataBufInfo, nullptr},
                                         });
  }
}

void Renderer::recordCommandBuffer(uint32_t imageIdx) {
  auto& cmd = mFrameData[mFrameIdx].cmd;
  auto& swapchainImg = mSwapChain.images[imageIdx].img;

  cmd.begin({});

  // Color attachment transition
  vkh::transition_image_layout(swapchainImg,
                               cmd,
                               vk::ImageLayout::eUndefined,
                               vk::ImageLayout::eColorAttachmentOptimal,
                               {},
                               vk::AccessFlagBits2::eColorAttachmentWrite,
                               vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                               vk::ImageAspectFlagBits::eColor);

  // Depth attachment transition
  vkh::transition_image_layout(mSwapChain.depthImage.img,
                               cmd,
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
    .imageView = mSwapChain.depthImage.view,
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

  cmd.beginRendering(renderingInfo);

  cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, mPipeline.GetPipeline());

  cmd.setViewport(0, vk::Viewport{0.0f, 0.0f, static_cast<float>(mSwapChain.extent.width), static_cast<float>(mSwapChain.extent.height), 0.0f, 1.0f});
  cmd.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), mSwapChain.extent));

  cmd.bindVertexBuffers(0, {mMeshBuffer.buffer}, {0});
  cmd.bindIndexBuffer(mMeshBuffer.buffer, mMesh.GetVerticesSizeBytes(), mMesh.GetVkIndexType());
  cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, mPipeline.GetLayout(), 0, *mDescriptorSets.sets[mFrameIdx], nullptr);
  cmd.drawIndexed(mMesh.indices.size(), numInstances, 0, 0, 0);

  cmd.endRendering();

  vkh::transition_image_layout(swapchainImg,
                               cmd,
                               vk::ImageLayout::eColorAttachmentOptimal,
                               vk::ImageLayout::ePresentSrcKHR,
                               vk::AccessFlagBits2::eColorAttachmentWrite,
                               {},
                               vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits2::eBottomOfPipe,
                               vk::ImageAspectFlagBits::eColor);
  cmd.end();
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