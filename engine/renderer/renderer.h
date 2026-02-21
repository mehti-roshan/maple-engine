#pragma once

#include <functional>
#include <glm/glm.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>
#include <vector>

#include "engine/renderer/mesh.h"
#include "engine/renderer/vk_buffer.h"
#include "engine/renderer/vk_memory_manager.h"
#include "engine/renderer/vk_sampler.h"
#include "engine/renderer/vk_texture.h"
#include "log_macros.h"

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

#include <engine/third_party/vma/vk_mem_alloc.h>

typedef struct VkInstance_T* VkInstance;
typedef struct VkSurfaceKHR_T* VkSurfaceKHR;

// creates a VkSurfaceKHR using the provided VkInstance
using SurfaceCreateCallback = std::function<VkSurfaceKHR(VkInstance)>;
// query the framebuffer size from the window library
using FrameBufferSizeCallback = std::function<void(uint32_t&, uint32_t&)>;

struct SwapChainDetails {
  vk::SurfaceFormatKHR format;
  vk::Extent2D extent;
};

struct Queues {
  vk::raii::Queue graphics = nullptr;
  vk::raii::Queue present = nullptr;
  vk::raii::Queue tranfer = nullptr;
  vk::raii::Queue compute = nullptr;
};

struct CommandPools {
  vk::raii::CommandPool graphics = nullptr;
  vk::raii::CommandPool transfer = nullptr;
};

struct FrameData {
  vk::raii::CommandBuffer cmd = nullptr;
  vk::raii::Semaphore presentCompleteSem = nullptr;
  vk::raii::Fence drawFence = nullptr;
};

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
  ~Renderer() { mDevice.waitIdle(); }

  void Init(const std::vector<const char*>& glfwExtensions, SurfaceCreateCallback, FrameBufferSizeCallback);
  void DrawFrame();

  void SetFrameBufferResized() { mFrameBufferResized = true; }

 private:
  static constexpr int MAX_FRAMES_IN_FLIGHT = 2;

  bool mFrameBufferResized = false;
  uint32_t mFrameIdx = 0;
  FrameBufferSizeCallback mFrameBufferSizeCallback;

  vk::raii::Context mContext;
  vk::raii::Instance mInstance = nullptr;
  vk::raii::DebugUtilsMessengerEXT mDebugMessenger = nullptr;
  vk::raii::SurfaceKHR mSurface = nullptr;
  vk::raii::PhysicalDevice mPhysicalDevice = nullptr;
  vk::raii::Device mDevice = nullptr;
  VulkanMemoryManager mMemoryManager;
  Queues mQueues;
  vk::raii::SwapchainKHR mSwapChain = nullptr;
  SwapChainDetails mSwapChainDetails;
  std::vector<vk::Image> mSwapChainImages;
  std::vector<vk::raii::ImageView> mSwapChainImageViews;

  vk::raii::DescriptorSetLayout mDescriptorSetLayout = nullptr;
  vk::raii::DescriptorPool mDescriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> mDescriptorSets;
  vk::raii::PipelineLayout mPipelineLayout = nullptr;
  vk::raii::Pipeline mGraphicsPipeline = nullptr;

  CommandPools mCommandPools;
  // TODO: implement
  // std::vector<VulkanBuffer> mPendingUploads;
  // vk::raii::CommandBuffer mTransferCmd = nullptr;
  // vk::raii::Fence mPendingUploadFence = nullptr;
  std::vector<FrameData> mFrameData;
  std::vector<vk::raii::Semaphore> mRenderCompleteSems;

  Mesh<Vertex, uint32_t> mMesh;
  VulkanBuffer mMeshBuffer;

  std::vector<VulkanBuffer> mUniformBuffers;

  VulkanTexture mDepthImage;

  VulkanTexture mTexture;
  VulkanSampler mSampler;

  void createInstance(const std::vector<const char*>& glfwExtensions);
  void setupDebugMessenger();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createMemoryManager();
  void createSwapChain();
  void createImageViews();
  void createDescriptorSetLayout();
  void createGraphicsPipeline();
  void createCommandPools();
  void createDepthResources();
  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();
  // TODO: implement
  // void createTransferData();

  void createFrameData();

  void createTexture();
  void createTextureSampler();
  void createMeshBuffer();

  void recreateSwapChain();
  void cleanupSwapChain();

  void recordCommandBuffer(uint32_t imageIdx);
  void updateUniformBuffer(uint32_t currentImage);

  vk::Format findSupportedFormat(const std::vector<vk::Format>& candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) {
    for (vk::Format format : candidates) {
      vk::FormatProperties props = mPhysicalDevice.getFormatProperties(format);

      if (tiling == vk::ImageTiling::eLinear && (props.linearTilingFeatures & features) == features) {
        return format;
      } else if (tiling == vk::ImageTiling::eOptimal && (props.optimalTilingFeatures & features) == features) {
        return format;
      }
    }

    MAPLE_FATAL("Failed to find supported format");
  }

  vk::Format findDepthFormat() {
    return findSupportedFormat({vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint},
                               vk::ImageTiling::eOptimal,
                               vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }

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
    vk::raii::CommandBuffer commandBuffer = std::move(mDevice.allocateCommandBuffers(allocInfo).front());
    vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    commandBuffer.begin(beginInfo);
    return commandBuffer;
  }

  void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) {
    commandBuffer.end();
    vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};
    mQueues.graphics.submit(submitInfo, nullptr);
    mQueues.graphics.waitIdle();
  }
};
}  // namespace maple