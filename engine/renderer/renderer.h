#pragma once

#include <functional>
#include <glm/glm.hpp>

#include "engine/renderer/vk_graphics_pipeline.h"
#include "engine/renderer/vk_logical_device.h"
#include "engine/renderer/vk_physical_device.h"
#include "engine/renderer/vk_swapchain.h"
#define GLM_ENABLE_EXPERIMENTAL
#include <engine/third_party/vma/vk_mem_alloc.h>

#include <glm/gtx/hash.hpp>
#include <vector>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#include "engine/renderer/mesh.h"
#include "engine/renderer/vk_buffer.h"
#include "engine/renderer/vk_memory_manager.h"
#include "engine/renderer/vk_sampler.h"
#include "engine/renderer/vk_texture.h"
#include "vk_instance_ctx.h"

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static vk::VertexInputBindingDescription getBindingDescription() { return {0, sizeof(Vertex), vk::VertexInputRate::eVertex}; }
  static std::vector<vk::VertexInputAttributeDescription> getAttributeDescriptions() {
    return {vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))};
  }

  bool operator==(const Vertex& other) const { return pos == other.pos && color == other.color && texCoord == other.texCoord; }
};

namespace std {
template <>
struct hash<Vertex> {
  size_t operator()(Vertex const& vertex) const {
    return ((hash<glm::vec3>()(vertex.pos) ^ (hash<glm::vec3>()(vertex.color) << 1)) >> 1) ^ (hash<glm::vec2>()(vertex.texCoord) << 1);
  }
};
}  // namespace std

namespace maple {
class Renderer {
 public:
  ~Renderer() { mDevice.device.waitIdle(); }

  void Init(const std::vector<const char*>& glfwExtensions, SurfaceCreateCallback, FrameBufferSizeCallback);
  void DrawFrame();

  void SetFrameBufferResized() { mFrameBufferResized = true; }

 private:
  static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

  bool mFrameBufferResized = false;
  uint32_t mFrameIdx = 0;
  FrameBufferSizeCallback mFrameBufferSizeCallback;

  VulkanInstanceContext mInstanceCtx;
  vk::raii::SurfaceKHR mSurface = nullptr;
  VulkanPhysicalDevice mPhysicalDevice;
  VulkanLogicalDevice mDevice;
  VulkanMemoryManager mMemoryManager;
  VulkanSwapChain mSwapChain;

  VulkanGraphicsPipeline mGraphicsPipeline;
  vk::raii::DescriptorSetLayout mDescriptorSetLayout = nullptr;
  vk::raii::DescriptorPool mDescriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> mDescriptorSets;

  struct CommandPools {
    vk::raii::CommandPool graphics = nullptr;
    vk::raii::CommandPool transfer = nullptr;
  };
  CommandPools mCommandPools;
  // TODO: implement
  // std::vector<VulkanBuffer> mPendingUploads;
  // vk::raii::CommandBuffer mTransferCmd = nullptr;
  // vk::raii::Fence mPendingUploadFence = nullptr;
  struct FrameData {
    vk::raii::CommandBuffer cmd = nullptr;
    vk::raii::Semaphore presentCompleteSem = nullptr;
    vk::raii::Fence drawFence = nullptr;
  };
  std::vector<FrameData> mFrameData;
  std::vector<vk::raii::Semaphore> mRenderCompleteSems;

  Mesh<Vertex, uint32_t> mMesh;
  VulkanBuffer mMeshBuffer;

  std::vector<VulkanBuffer> mUniformBuffers;

  VulkanTexture mTexture;
  VulkanSampler mSampler;

  void createDescriptorSetLayout();
  void createCommandPools();
  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();
  // TODO: implement
  // void createTransferData();

  void createFrameData();

  void createTexture();
  void createTextureSampler();
  void createMeshBuffer();

  void recordCommandBuffer(uint32_t imageIdx);
  void updateUniformBuffer(uint32_t currentImage);

  void copyBuffer(VkBuffer srcBuffer, VkBuffer dstBuffer, vk::DeviceSize size) {
    auto commandCopyBuffer = beginSingleTimeCommands();
    vk::BufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = size};
    commandCopyBuffer.copyBuffer(srcBuffer, dstBuffer, copyRegion);
    endSingleTimeCommands(commandCopyBuffer);
  }

  [[nodiscard]]
  vk::raii::CommandBuffer beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo{
      .commandPool = mCommandPools.graphics, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
    vk::raii::CommandBuffer commandBuffer = std::move(mDevice.device.allocateCommandBuffers(allocInfo).front());
    vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    commandBuffer.begin(beginInfo);
    return commandBuffer;
  }

  void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) {
    commandBuffer.end();
    vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};
    mDevice.queues.graphics.submit(submitInfo, nullptr);
    mDevice.queues.graphics.waitIdle();
  }
};
}  // namespace maple