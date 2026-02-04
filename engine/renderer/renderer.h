#pragma once
#include <functional>
#include <glm/fwd.hpp>
#include <vector>

#include "log_macros.h"

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#if defined(__INTELLISENSE__) || !defined(USE_CPP20_MODULES)
#include <vulkan/vulkan_raii.hpp>
#else
import vulkan_hpp;
#endif

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

namespace maple {

class Renderer {
 public:
  ~Renderer() { mDevice.waitIdle(); }

  void Init(const std::vector<const char*>& glfwExtensions, SurfaceCreateCallback, FrameBufferSizeCallback);
  void DrawFrame();

  void SetFrameBufferResized() { mFrameBufferResized = true; }

 private:
  FrameBufferSizeCallback mFrameBufferSizeCallback;
  bool mFrameBufferResized = false;
  uint32_t mFrameIdx = 0;

  vk::raii::Context mContext;
  vk::raii::Instance mInstance = nullptr;
  vk::raii::DebugUtilsMessengerEXT mDebugMessenger = nullptr;
  vk::raii::SurfaceKHR mSurface = nullptr;
  vk::raii::PhysicalDevice mPhysicalDevice = nullptr;
  vk::raii::Device mDevice = nullptr;
  vk::raii::Queue mGraphicsQueue = nullptr;
  vk::raii::Queue mPresentQueue = nullptr;
  vk::raii::SwapchainKHR mSwapChain = nullptr;
  SwapChainDetails mSwapChainDetails;
  std::vector<vk::Image> mSwapChainImages;
  std::vector<vk::raii::ImageView> mSwapChainImageViews;

  vk::raii::DescriptorSetLayout mDescriptorSetLayout = nullptr;
  vk::raii::DescriptorPool mDescriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> mDescriptorSets;
  vk::raii::PipelineLayout mPipelineLayout = nullptr;
  vk::raii::Pipeline mGraphicsPipeline = nullptr;

  vk::raii::CommandPool mCommandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> mCommandBuffers;

  std::vector<vk::raii::Semaphore> mPresentCompleteSems;
  std::vector<vk::raii::Semaphore> mRenderCompleteSems;
  std::vector<vk::raii::Fence> mDrawFences;

  vk::raii::Buffer mVertexBuffer = nullptr;
  vk::raii::DeviceMemory mVertexBufferMemory = nullptr;
  vk::raii::Buffer mIndexBuffer = nullptr;
  vk::raii::DeviceMemory mIndexBufferMemory = nullptr;

  vk::raii::Buffer IndexBuffer = nullptr;
  vk::raii::DeviceMemory IndexBufferMemory = nullptr;

  std::vector<vk::raii::Buffer> mUniformBuffers;
  std::vector<vk::raii::DeviceMemory> mUniformBuffersMemory;
  std::vector<void*> mUniformBuffersMapped;

  vk::raii::Image mDepthImage = nullptr;
  vk::raii::DeviceMemory mDepthImageMemory = nullptr;
  vk::raii::ImageView mDepthImageView = nullptr;

  vk::raii::Image mTextureImage = nullptr;
  vk::raii::DeviceMemory mTextureImageMemory = nullptr;
  vk::raii::ImageView mTextureImageView = nullptr;
  vk::raii::Sampler mTextureSampler = nullptr;

  void createInstance(const std::vector<const char*>& glfwExtensions);
  void setupDebugMessenger();
  void pickPhysicalDevice();
  void createLogicalDevice();
  void createSwapChain();
  void createImageViews();
  void createGraphicsPipeline();
  void createDescriptorSetLayout();
  void createCommandPool();
  void createDepthResources();
  void createTextureImage();
  void createTextureImageView();
  void createTextureSampler();
  void createVertexBuffer();
  void createIndexBuffer();
  void createUniformBuffers();
  void createDescriptorPool();
  void createDescriptorSets();
  void updateUniformBuffer(uint32_t currentImage);
  void createCommandBuffers();
  void recordCommandBuffer(uint32_t imageIdx);
  void createSyncObjects();

  void recreateSwapChain();
  void cleanupSwapChain();

  uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties);
  void createBuffer(vk::DeviceSize size,
                    vk::BufferUsageFlags usage,
                    vk::MemoryPropertyFlags properties,
                    vk::raii::Buffer& buffer,
                    vk::raii::DeviceMemory& bufferMemory);
  void copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size);
  void createImage(
    glm::u32vec2 size, vk::Format, vk::ImageTiling, vk::ImageUsageFlags, vk::MemoryPropertyFlags, vk::raii::Image&, vk::raii::DeviceMemory&);

  vk::raii::CommandBuffer beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo{.commandPool = mCommandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1};
    vk::raii::CommandBuffer commandBuffer = std::move(mDevice.allocateCommandBuffers(allocInfo).front());

    vk::CommandBufferBeginInfo beginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit};
    commandBuffer.begin(beginInfo);

    return commandBuffer;
  }

  void endSingleTimeCommands(vk::raii::CommandBuffer& commandBuffer) {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{.commandBufferCount = 1, .pCommandBuffers = &*commandBuffer};
    mGraphicsQueue.submit(submitInfo, nullptr);
    mGraphicsQueue.waitIdle();
  }

  void transitionImageLayout(const vk::raii::Image& image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout) {
    auto commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier{
      .oldLayout = oldLayout, .newLayout = newLayout, .image = image, .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}};

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined && newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = {};
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

      sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
      destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal && newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      sourceStage = vk::PipelineStageFlagBits::eTransfer;
      destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      MAPLE_FATAL("Unsupported layout transition");
    }
    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, nullptr, barrier);
    endSingleTimeCommands(commandBuffer);
  }

  void copyBufferToImage(const vk::raii::Buffer& buffer, vk::raii::Image& image, uint32_t width, uint32_t height) {
    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();
    vk::BufferImageCopy region{.bufferOffset = 0,
                               .bufferRowLength = 0,
                               .bufferImageHeight = 0,
                               .imageSubresource = {vk::ImageAspectFlagBits::eColor, 0, 0, 1},
                               .imageOffset = {0, 0, 0},
                               .imageExtent = {width, height, 1}};
    commandBuffer.copyBufferToImage(*buffer, *image, vk::ImageLayout::eTransferDstOptimal, region);
    endSingleTimeCommands(commandBuffer);
  }

  vk::raii::ImageView createImageView(const vk::raii::Image& image, vk::Format format, vk::ImageAspectFlagBits aspectFlags) {
    vk::ImageViewCreateInfo viewInfo{
      .image = image,
      .viewType = vk::ImageViewType::e2D,
      .format = format,
      .components = {.r = vk::ComponentSwizzle::eIdentity,
                     .g = vk::ComponentSwizzle::eIdentity,
                     .b = vk::ComponentSwizzle::eIdentity,
                     .a = vk::ComponentSwizzle::eIdentity},
      .subresourceRange = {.aspectMask = aspectFlags, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
    return vk::raii::ImageView(mDevice, viewInfo);
  }

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

  bool hasStencilComponent(vk::Format format) { return format == vk::Format::eD32SfloatS8Uint || format == vk::Format::eD24UnormS8Uint; }
};
}  // namespace maple