#include <engine/file/file.h>
#include <engine/logging/log_macros.h>
#include <engine/renderer/renderer.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <glm/geometric.hpp>
#include <map>
#include <ranges>
#include <vector>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>
#define GLM_FORCE_RADIANS
#include <glm/ext/matrix_clip_space.hpp>
#include <glm/ext/matrix_transform.hpp>
#include <glm/fwd.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define STB_IMAGE_IMPLEMENTATION
#include <engine/stb/stb_image.h>

const std::vector<char const*> validationLayers = {"VK_LAYER_KHRONOS_validation"};

#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

std::vector<const char*> deviceExtensions = {vk::KHRSwapchainExtensionName};

static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                                      vk::DebugUtilsMessageTypeFlagsEXT type,
                                                      const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                      void*) {
  MAPLE_DEBUG("Vulkan validation layer: type {} msg: {}", to_string(type), pCallbackData->pMessage);
  return vk::False;
}

constexpr int MAX_FRAMES_IN_FLIGHT = 2;

bool deviceSupportsExtension(vk::raii::PhysicalDevice device, const char* extensionName) {
  auto extensions = device.enumerateDeviceExtensionProperties();
  for (auto const& ext : extensions)
    if (strcmp(ext.extensionName, extensionName) == 0) return true;
  return false;
}

struct QueueFamilyIndices {
  uint32_t graphics, present, compute;
  bool hasGraphics() { return graphics != VK_QUEUE_FAMILY_IGNORED; }
  bool hasPresent() { return present != VK_QUEUE_FAMILY_IGNORED; }
  bool hasCompute() { return compute != VK_QUEUE_FAMILY_IGNORED; }
  bool complete() { return hasGraphics() && hasPresent() && hasCompute(); }
};

QueueFamilyIndices getDeviceQueueFamilyIndices(vk::raii::PhysicalDevice device, vk::SurfaceKHR surface) {
  QueueFamilyIndices indices = {VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED};

  for (auto [i, qfp] : std::views::enumerate(device.getQueueFamilyProperties())) {
    bool graphics = static_cast<bool>(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
    indices.graphics = graphics && !indices.hasGraphics() ? static_cast<uint32_t>(i) : indices.graphics;

    bool present = device.getSurfaceSupportKHR(i, surface);
    indices.present = present && !indices.hasPresent() ? static_cast<uint32_t>(i) : indices.present;

    bool compute = static_cast<bool>(qfp.queueFlags & vk::QueueFlagBits::eCompute);
    indices.compute = compute && !indices.hasCompute() ? static_cast<uint32_t>(i) : indices.compute;

    if (indices.complete()) break;
  }

  return indices;
}

bool isPhysicalDeviceSuitable(vk::raii::PhysicalDevice device, vk::SurfaceKHR surface) {
  if (device.getProperties().apiVersion < VK_API_VERSION_1_3) return false;

  auto featureChain = device.getFeatures2<vk::PhysicalDeviceFeatures2,
                                          vk::PhysicalDeviceVulkan11Features,
                                          vk::PhysicalDeviceVulkan13Features,
                                          vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
  auto features = featureChain.get<vk::PhysicalDeviceFeatures2>();
  auto vk11Features = featureChain.get<vk::PhysicalDeviceVulkan11Features>();
  auto vk13Features = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
  auto extDynStateFeatures = featureChain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

  if (!features.features.samplerAnisotropy || !vk11Features.shaderDrawParameters || !vk13Features.dynamicRendering ||
      !vk13Features.synchronization2 || !extDynStateFeatures.extendedDynamicState)
    return false;

  auto extensions = device.enumerateDeviceExtensionProperties();
  for (auto ext : deviceExtensions)
    if (!deviceSupportsExtension(device, ext)) return false;

  auto qIndices = getDeviceQueueFamilyIndices(device, surface);
  if (!qIndices.complete()) return false;

  return true;
}

float scorePhysicalDevice(vk::raii::PhysicalDevice device, QueueFamilyIndices qIndices) {
  float score = 0.0f;

  auto deviceProperties = device.getProperties2();
  deviceProperties.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu ? score += 1000.0f : score += 0.0f;

  auto devMemProps = device.getMemoryProperties2();
  uint32_t vramBytes = 0;
  for (auto const& memHeap : devMemProps.memoryProperties.memoryHeaps)
    if (memHeap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) vramBytes += memHeap.size;
  score += (float)vramBytes / 1024 / 1024;  // in MB

  score += static_cast<float>(deviceProperties.properties.limits.maxImageDimension2D);

  if (qIndices.graphics != qIndices.present) score += 500.0f;

  return score;
}

vk::SurfaceFormatKHR chooseSwapSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& availableFormats) {
  for (const auto& availableFormat : availableFormats)
    if (availableFormat.format == vk::Format::eB8G8R8A8Srgb && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
      return availableFormat;

  return availableFormats[0];
}

vk::PresentModeKHR chooseSwapPresentMode(const std::vector<vk::PresentModeKHR>& availablePresentModes) {
  for (const auto& availablePresentMode : availablePresentModes) {
    if (availablePresentMode == vk::PresentModeKHR::eMailbox) {
      return availablePresentMode;
    }
  }
  return vk::PresentModeKHR::eFifo;
}

vk::Extent2D chooseSwapExtent(const vk::SurfaceCapabilitiesKHR& capabilities, FrameBufferSizeCallback fbCallback) {
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
    return capabilities.currentExtent;
  }
  uint32_t width, height;
  fbCallback(width, height);

  return {std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
          std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height)};
}

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

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static vk::VertexInputBindingDescription getBindingDescription() { return {0, sizeof(Vertex), vk::VertexInputRate::eVertex}; }
  static std::array<vk::VertexInputAttributeDescription, 3> getAttributeDescriptions() {
    return {vk::VertexInputAttributeDescription(0, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, pos)),
            vk::VertexInputAttributeDescription(1, 0, vk::Format::eR32G32B32Sfloat, offsetof(Vertex, color)),
            vk::VertexInputAttributeDescription(2, 0, vk::Format::eR32G32Sfloat, offsetof(Vertex, texCoord))};
  }
};

const std::vector<Vertex> vertices = {
  {{-0.5f, -0.5f, 0.0f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
  {{0.5f, -0.5f, 0.0f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
  {{0.5f, 0.5f, 0.0f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
  {{-0.5f, 0.5f, 0.0f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},

  {{-0.5f, -0.5f, -0.5f}, {1.0f, 0.0f, 0.0f}, {1.0f, 0.0f}},
  {{0.5f, -0.5f, -0.5f}, {0.0f, 1.0f, 0.0f}, {0.0f, 0.0f}},
  {{0.5f, 0.5f, -0.5f}, {0.0f, 0.0f, 1.0f}, {0.0f, 1.0f}},
  {{-0.5f, 0.5f, -0.5f}, {1.0f, 1.0f, 1.0f}, {1.0f, 1.0f}},
};

const std::vector<uint16_t> indices = {
  0,
  1,
  2,
  2,
  3,
  0,

  4,
  5,
  6,
  6,
  7,
  4,
};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

namespace maple {
void Renderer::Init(const std::vector<const char*>& glfwExtensions, SurfaceCreateCallback surfaceCallback, FrameBufferSizeCallback fbCallback) {
  mFrameBufferSizeCallback = fbCallback;
  createInstance(glfwExtensions);
  setupDebugMessenger();
  mSurface = vk::raii::SurfaceKHR(mInstance, surfaceCallback(*mInstance));
  pickPhysicalDevice();
  createLogicalDevice();
  createSwapChain();
  createImageViews();
  createDescriptorSetLayout();
  createGraphicsPipeline();
  createCommandPool();
  createDepthResources();
  createTextureImage();
  createTextureImageView();
  createTextureSampler();
  createVertexBuffer();
  createIndexBuffer();
  createUniformBuffers();
  createDescriptorPool();
  createDescriptorSets();
  createCommandBuffers();
  createSyncObjects();
}

void Renderer::DrawFrame() {
  auto fenceResult = mDevice.waitForFences(*mDrawFences[mFrameIdx], vk::True, UINT64_MAX);
  auto [result, imageIdx] = mSwapChain.acquireNextImage(UINT64_MAX, *mPresentCompleteSems[mFrameIdx], nullptr);
  mDevice.resetFences(*mDrawFences[mFrameIdx]);

  mCommandBuffers[mFrameIdx].reset();

  updateUniformBuffer(mFrameIdx);
  recordCommandBuffer(imageIdx);

  vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
  const vk::SubmitInfo submitInfo{
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &*mPresentCompleteSems[mFrameIdx],
    .pWaitDstStageMask = &waitDestinationStageMask,
    .commandBufferCount = 1,
    .pCommandBuffers = &*mCommandBuffers[mFrameIdx],
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &*mRenderCompleteSems[mFrameIdx],
  };

  mGraphicsQueue.submit(submitInfo, *mDrawFences[mFrameIdx]);

  const vk::PresentInfoKHR presentInfoKHR{
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &*mRenderCompleteSems[mFrameIdx],
    .swapchainCount = 1,
    .pSwapchains = &*mSwapChain,
    .pImageIndices = &imageIdx,
  };

  auto presentResult = mPresentQueue.presentKHR(presentInfoKHR);

  if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || mFrameBufferResized) {
    mFrameBufferResized = false;
    recreateSwapChain();
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
    .view = glm::lookAt(glm::vec3(2.0f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f)),
    .proj = glm::perspective(
      glm::radians(45.0f), static_cast<float>(mSwapChainDetails.extent.width) / static_cast<float>(mSwapChainDetails.extent.height), 0.1f, 2000.0f)};
  ubo.proj[1][1] *= -1;  // Invert Y for Vulkan

  memcpy(mUniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
}

void Renderer::createInstance(const std::vector<const char*>& glfwExtensions) {
  std::vector<const char*> requiredExtensions, requiredLayers;

  requiredExtensions.assign(glfwExtensions.begin(), glfwExtensions.end());
  if (enableValidationLayers) {
    requiredExtensions.push_back(vk::EXTDebugUtilsExtensionName);
    requiredLayers.assign(validationLayers.begin(), validationLayers.end());
  }

  // Check required extensions and layers
  auto extensionProperties = mContext.enumerateInstanceExtensionProperties();
  for (auto [i, v] : std::views::enumerate(glfwExtensions))
    if (std::ranges::none_of(extensionProperties, [v](auto const extProps) { return strcmp(extProps.extensionName, v) == 0; }))
      MAPLE_FATAL("Missing required extension {}", v);

  auto layerProperties = mContext.enumerateInstanceLayerProperties();
  if (std::ranges::any_of(requiredLayers, [&layerProperties](auto const& requiredLayer) {
        return std::ranges::none_of(layerProperties,
                                    [requiredLayer](auto const& layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
      }))
    MAPLE_FATAL("One or more required layers not supported");

  constexpr vk::ApplicationInfo appInfo{
    .applicationVersion = VK_MAKE_VERSION(1, 0, 0),
    .pEngineName = "Maple",
    .engineVersion = VK_MAKE_VERSION(1, 0, 0),
    .apiVersion = vk::ApiVersion14,
  };

  vk::InstanceCreateInfo createInfo{
    .pApplicationInfo = &appInfo,
    .enabledLayerCount = static_cast<uint32_t>(requiredLayers.size()),
    .ppEnabledLayerNames = requiredLayers.data(),
    .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
    .ppEnabledExtensionNames = requiredExtensions.data(),
  };

  mInstance = vk::raii::Instance(mContext, createInfo);
}

void Renderer::setupDebugMessenger() {
  if (!enableValidationLayers) return;

  vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose |
                                                      vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                                                      vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
  vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
                                                     vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
                                                     vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);
  vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{
    .messageSeverity = severityFlags, .messageType = messageTypeFlags, .pfnUserCallback = &debugCallback};
  mDebugMessenger = mInstance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
}

void Renderer::pickPhysicalDevice() {
  auto devices = mInstance.enumeratePhysicalDevices();
  std::map<float, vk::raii::PhysicalDevice> scoredDevices;

  for (auto& device : devices) {
    if (!isPhysicalDeviceSuitable(device, *mSurface)) continue;
    auto qIndices = getDeviceQueueFamilyIndices(device, *mSurface);
    float score = scorePhysicalDevice(device, qIndices);
    scoredDevices.insert({score, device});
  }

  if (scoredDevices.empty()) MAPLE_FATAL("Failed to find a suitable GPU");

  mPhysicalDevice = scoredDevices.rbegin()->second;
}

void Renderer::createLogicalDevice() {
  // std::vector<vk::QueueFamilyProperties> queueFamilyProperties = mPhysicalDevice.getQueueFamilyProperties();
  QueueFamilyIndices qIndices = getDeviceQueueFamilyIndices(mPhysicalDevice, *mSurface);

  float queuePriority = 0.5f;
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = qIndices.graphics, .queueCount = 1, .pQueuePriorities = &queuePriority};

  vk::StructureChain<vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceVulkan11Features,
                     vk::PhysicalDeviceVulkan13Features,
                     vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
    featureChain = {
      {.features = {.samplerAnisotropy = true}},             // vk::PhysicalDeviceFeatures2
      {.shaderDrawParameters = true},                        // Enable shader draw parameters from Vulkan 1.1
      {.synchronization2 = true, .dynamicRendering = true},  // Enable dynamic rendering from Vulkan 1.3
      {.extendedDynamicState = true}                         // Enable extended dynamic state from the extension
    };

  vk::DeviceCreateInfo deviceCreateInfo{
    .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
    .queueCreateInfoCount = 1,
    .pQueueCreateInfos = &deviceQueueCreateInfo,
    .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
    .ppEnabledExtensionNames = deviceExtensions.data(),
  };

  mDevice = vk::raii::Device(mPhysicalDevice, deviceCreateInfo);

  VkBool32 presentSupport = mPhysicalDevice.getSurfaceSupportKHR(qIndices.graphics, *mSurface);

  mGraphicsQueue = vk::raii::Queue(mDevice, qIndices.graphics, 0);
  mPresentQueue = qIndices.graphics == qIndices.present ? mGraphicsQueue : vk::raii::Queue(mDevice, qIndices.present, 0);
}

void Renderer::createSwapChain() {
  auto surfaceCapabilities = mPhysicalDevice.getSurfaceCapabilitiesKHR(*mSurface);
  mSwapChainDetails.format = chooseSwapSurfaceFormat(mPhysicalDevice.getSurfaceFormatsKHR(*mSurface));
  mSwapChainDetails.extent = chooseSwapExtent(surfaceCapabilities, mFrameBufferSizeCallback);

  auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
  minImageCount =
    (surfaceCapabilities.maxImageCount > 0 && minImageCount > surfaceCapabilities.maxImageCount) ? surfaceCapabilities.maxImageCount : minImageCount;

  vk::SwapchainCreateInfoKHR swapChainCreateInfo{
    .flags = vk::SwapchainCreateFlagsKHR(),
    .surface = *mSurface,
    .minImageCount = minImageCount,
    .imageFormat = mSwapChainDetails.format.format,
    .imageColorSpace = mSwapChainDetails.format.colorSpace,
    .imageExtent = mSwapChainDetails.extent,
    .imageArrayLayers = 1,
    .imageUsage = vk::ImageUsageFlagBits::eColorAttachment,
    .imageSharingMode = vk::SharingMode::eExclusive,
    .preTransform = surfaceCapabilities.currentTransform,
    .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
    .presentMode = chooseSwapPresentMode(mPhysicalDevice.getSurfacePresentModesKHR(*mSurface)),
    .clipped = true,
    .oldSwapchain = nullptr,
  };

  QueueFamilyIndices qIndices = getDeviceQueueFamilyIndices(mPhysicalDevice, *mSurface);
  uint32_t queueFamilyIndices[] = {qIndices.graphics, qIndices.present};

  if (qIndices.graphics != qIndices.present) {
    swapChainCreateInfo.imageSharingMode = vk::SharingMode::eConcurrent;
    swapChainCreateInfo.queueFamilyIndexCount = 2;
    swapChainCreateInfo.pQueueFamilyIndices = queueFamilyIndices;
  } else {
    swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
    swapChainCreateInfo.queueFamilyIndexCount = 0;      // Optional
    swapChainCreateInfo.pQueueFamilyIndices = nullptr;  // Optional
  }

  mSwapChain = vk::raii::SwapchainKHR(mDevice, swapChainCreateInfo);
  mSwapChainImages = mSwapChain.getImages();
}

void Renderer::createImageViews() {
  mSwapChainImageViews.clear();

  vk::ImageViewCreateInfo imageViewCreateInfo{
    .viewType = vk::ImageViewType::e2D,
    .format = mSwapChainDetails.format.format,
    .components =
      {
        vk::ComponentSwizzle::eIdentity,
        vk::ComponentSwizzle::eIdentity,
        vk::ComponentSwizzle::eIdentity,
        vk::ComponentSwizzle::eIdentity,
      },
    .subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1},
  };

  for (auto& image : mSwapChainImages) {
    imageViewCreateInfo.image = image;
    mSwapChainImageViews.emplace_back(mDevice, imageViewCreateInfo);
  }
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
  mDescriptorSetLayout = vk::raii::DescriptorSetLayout(mDevice, layoutInfo);
}

void Renderer::createGraphicsPipeline() {
  auto shaderModule = createShaderModule(mDevice, file::ReadFile("assets/shaders/slang.spv"));

  vk::PipelineShaderStageCreateInfo vertShaderStageInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = shaderModule, .pName = "vertMain"};
  vk::PipelineShaderStageCreateInfo fragShaderStageInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = shaderModule, .pName = "fragMain"};
  vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo, fragShaderStageInfo};

  std::vector dynamicStates = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};

  vk::PipelineDynamicStateCreateInfo dynamicState{
    .dynamicStateCount = static_cast<uint32_t>(dynamicStates.size()),
    .pDynamicStates = dynamicStates.data(),
  };

  auto bindingDescription = Vertex::getBindingDescription();
  auto attributeDescriptions = Vertex::getAttributeDescriptions();
  vk::PipelineVertexInputStateCreateInfo vertexInputInfo{
    .vertexBindingDescriptionCount = 1,
    .pVertexBindingDescriptions = &bindingDescription,
    .vertexAttributeDescriptionCount = attributeDescriptions.size(),
    .pVertexAttributeDescriptions = attributeDescriptions.data(),
  };

  vk::PipelineInputAssemblyStateCreateInfo inputAssembly{.topology = vk::PrimitiveTopology::eTriangleList};

  vk::Viewport viewport{
    0.0f, 0.0f, static_cast<float>(mSwapChainDetails.extent.width), static_cast<float>(mSwapChainDetails.extent.height), 0.0f, 1.0f};
  vk::Rect2D scissor{{0, 0}, mSwapChainDetails.extent};
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
  mPipelineLayout = vk::raii::PipelineLayout(mDevice, pipelineLayoutInfo);

  vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{.colorAttachmentCount = 1, .pColorAttachmentFormats = &mSwapChainDetails.format.format, .depthAttachmentFormat = findDepthFormat()};
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

  mGraphicsPipeline = vk::raii::Pipeline(mDevice, nullptr, pipelineInfo);
}

void Renderer::createCommandPool() {
  QueueFamilyIndices qIndices = getDeviceQueueFamilyIndices(mPhysicalDevice, *mSurface);

  vk::CommandPoolCreateInfo poolInfo{
    .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
    .queueFamilyIndex = qIndices.graphics,
  };

  mCommandPool = vk::raii::CommandPool(mDevice, poolInfo);
}

void Renderer::createDepthResources() {
  vk::Format depthFormat = findDepthFormat();
  createImage(glm::u32vec2(mSwapChainDetails.extent.width, mSwapChainDetails.extent.height),
              depthFormat,
              vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eDepthStencilAttachment,
              vk::MemoryPropertyFlagBits::eDeviceLocal,
              mDepthImage,
              mDepthImageMemory);

  mDepthImageView = createImageView(mDepthImage, depthFormat, vk::ImageAspectFlagBits::eDepth);
}

void Renderer::createTextureImage() {
  int32_t texWidth, texHeight, texChannels;
  stbi_uc* pixels = stbi_load("assets/textures/texture.jpg", &texWidth, &texHeight, &texChannels, STBI_rgb_alpha);
  vk::DeviceSize imageSize = texWidth * texHeight * 4;

  if (!pixels) MAPLE_FATAL("failed to load texture image");

  vk::raii::Buffer stagingBuffer({});
  vk::raii::DeviceMemory stagingBufferMemory({});
  createBuffer(imageSize,
               vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer,
               stagingBufferMemory);

  void* data = stagingBufferMemory.mapMemory(0, imageSize);
  memcpy(data, pixels, imageSize);
  stagingBufferMemory.unmapMemory();

  stbi_image_free(pixels);

  createImage(glm::u32vec2(texWidth, texHeight),
              vk::Format::eR8G8B8A8Srgb,
              vk::ImageTiling::eOptimal,
              vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
              vk::MemoryPropertyFlagBits::eDeviceLocal,
              mTextureImage,
              mTextureImageMemory);

  transitionImageLayout(mTextureImage, vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal);
  copyBufferToImage(stagingBuffer, mTextureImage, texWidth, texHeight);
  transitionImageLayout(mTextureImage, vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal);
}

void Renderer::createTextureImageView() {
  mTextureImageView = createImageView(mTextureImage, vk::Format::eR8G8B8A8Srgb, vk::ImageAspectFlagBits::eColor);
}

void Renderer::createTextureSampler() {
  vk::PhysicalDeviceProperties properties = mPhysicalDevice.getProperties();

  vk::SamplerCreateInfo samplerInfo{
    .magFilter = vk::Filter::eLinear,
    .minFilter = vk::Filter::eLinear,
    .mipmapMode = vk::SamplerMipmapMode::eLinear,
    .addressModeU = vk::SamplerAddressMode::eRepeat,
    .addressModeV = vk::SamplerAddressMode::eRepeat,
    .addressModeW = vk::SamplerAddressMode::eRepeat,
    .mipLodBias = 0.0f,
    .anisotropyEnable = vk::True,
    .maxAnisotropy = properties.limits.maxSamplerAnisotropy,
    .compareEnable = vk::False,
    .compareOp = vk::CompareOp::eAlways,
    .minLod = 0.0f,
    .maxLod = 1.0f,
    .borderColor = vk::BorderColor::eIntOpaqueBlack,
    .unnormalizedCoordinates = vk::False,
  };

  mTextureSampler = vk::raii::Sampler(mDevice, samplerInfo);
}

void Renderer::createVertexBuffer() {
  vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

  vk::BufferCreateInfo stagingInfo{.size = bufferSize, .usage = vk::BufferUsageFlagBits::eTransferSrc, .sharingMode = vk::SharingMode::eExclusive};
  vk::raii::Buffer stagingBuffer(mDevice, stagingInfo);
  vk::MemoryRequirements memRequirementsStaging = stagingBuffer.getMemoryRequirements();
  vk::MemoryAllocateInfo memoryAllocateInfoStaging{
    .allocationSize = memRequirementsStaging.size,
    .memoryTypeIndex =
      findMemoryType(memRequirementsStaging.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)};
  vk::raii::DeviceMemory stagingBufferMemory(mDevice, memoryAllocateInfoStaging);

  stagingBuffer.bindMemory(stagingBufferMemory, 0);
  void* dataStaging = stagingBufferMemory.mapMemory(0, stagingInfo.size);
  memcpy(dataStaging, vertices.data(), stagingInfo.size);
  stagingBufferMemory.unmapMemory();

  vk::BufferCreateInfo bufferInfo{.size = bufferSize,
                                  .usage = vk::BufferUsageFlagBits::eVertexBuffer | vk::BufferUsageFlagBits::eTransferDst,
                                  .sharingMode = vk::SharingMode::eExclusive};
  mVertexBuffer = vk::raii::Buffer(mDevice, bufferInfo);

  vk::MemoryRequirements memRequirements = mVertexBuffer.getMemoryRequirements();
  vk::MemoryAllocateInfo memoryAllocateInfo{
    .allocationSize = memRequirements.size,
    .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal)};
  mVertexBufferMemory = vk::raii::DeviceMemory(mDevice, memoryAllocateInfo);

  mVertexBuffer.bindMemory(*mVertexBufferMemory, 0);

  copyBuffer(stagingBuffer, mVertexBuffer, stagingInfo.size);
}

void Renderer::createIndexBuffer() {
  vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

  vk::raii::Buffer stagingBuffer({});
  vk::raii::DeviceMemory stagingBufferMemory({});
  createBuffer(bufferSize,
               vk::BufferUsageFlagBits::eTransferSrc,
               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
               stagingBuffer,
               stagingBufferMemory);

  void* data = stagingBufferMemory.mapMemory(0, bufferSize);
  memcpy(data, indices.data(), (size_t)bufferSize);
  stagingBufferMemory.unmapMemory();

  createBuffer(bufferSize,
               vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
               vk::MemoryPropertyFlagBits::eDeviceLocal,
               mIndexBuffer,
               mIndexBufferMemory);

  copyBuffer(stagingBuffer, mIndexBuffer, bufferSize);
}

void Renderer::createUniformBuffers() {
  mUniformBuffers.clear();
  mUniformBuffersMemory.clear();
  mUniformBuffersMapped.clear();

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
    vk::raii::Buffer buffer({});
    vk::raii::DeviceMemory bufferMem({});
    createBuffer(bufferSize,
                 vk::BufferUsageFlagBits::eUniformBuffer,
                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                 buffer,
                 bufferMem);
    mUniformBuffers.emplace_back(std::move(buffer));
    mUniformBuffersMemory.emplace_back(std::move(bufferMem));
    mUniformBuffersMapped.emplace_back(mUniformBuffersMemory[i].mapMemory(0, bufferSize));
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
  mDescriptorPool = vk::raii::DescriptorPool(mDevice, poolInfo);
}

void Renderer::createCommandBuffers() {
  mCommandBuffers.clear();

  vk::CommandBufferAllocateInfo allocInfo{
    .commandPool = mCommandPool,
    .level = vk::CommandBufferLevel::ePrimary,
    .commandBufferCount = static_cast<uint32_t>(mSwapChainImages.size()),
  };

  mCommandBuffers = vk::raii::CommandBuffers(mDevice, allocInfo);
}

void Renderer::createDescriptorSets() {
  std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT, *mDescriptorSetLayout);
  vk::DescriptorSetAllocateInfo allocInfo{
    .descriptorPool = mDescriptorPool, .descriptorSetCount = static_cast<uint32_t>(layouts.size()), .pSetLayouts = layouts.data()};
  mDescriptorSets = vk::raii::DescriptorSets(mDevice, allocInfo);
  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    vk::DescriptorBufferInfo bufferInfo{.buffer = mUniformBuffers[i], .offset = 0, .range = sizeof(UniformBufferObject)};
    vk::DescriptorImageInfo imageInfo{
      .sampler = mTextureSampler,
      .imageView = mTextureImageView,
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

    mDevice.updateDescriptorSets(descriptorWrites, {});
  }
}

void Renderer::recordCommandBuffer(uint32_t imageIdx) {
  mCommandBuffers[mFrameIdx].begin({});

  // Color attachment transition
  transition_image_layout(mSwapChainImages[imageIdx],
                          mCommandBuffers[mFrameIdx],
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eColorAttachmentOptimal,
                          {},
                          vk::AccessFlagBits2::eColorAttachmentWrite,
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                          vk::ImageAspectFlagBits::eColor);

  // Depth attachment transition
  transition_image_layout(mDepthImage,
                          mCommandBuffers[mFrameIdx],
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
    .imageView = mDepthImageView,
    .imageLayout = vk::ImageLayout::eDepthAttachmentOptimal,
    .loadOp = vk::AttachmentLoadOp::eClear,
    .storeOp = vk::AttachmentStoreOp::eDontCare,
    .clearValue = clearDepth,
  };

  vk::RenderingAttachmentInfo colorAttachmentInfo = {
    .imageView = mSwapChainImageViews[imageIdx],
    .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
    .loadOp = vk::AttachmentLoadOp::eClear,
    .storeOp = vk::AttachmentStoreOp::eStore,
    .clearValue = clearColor,
  };

  vk::RenderingInfo renderingInfo = {
    .renderArea = {.offset = {0, 0}, .extent = mSwapChainDetails.extent},
    .layerCount = 1,
    .colorAttachmentCount = 1,
    .pColorAttachments = &colorAttachmentInfo,
    .pDepthAttachment = &depthAttachmentInfo,
  };

  mCommandBuffers[mFrameIdx].beginRendering(renderingInfo);

  mCommandBuffers[mFrameIdx].bindPipeline(vk::PipelineBindPoint::eGraphics, mGraphicsPipeline);

  mCommandBuffers[mFrameIdx].setViewport(
    0, vk::Viewport{0.0f, 0.0f, static_cast<float>(mSwapChainDetails.extent.width), static_cast<float>(mSwapChainDetails.extent.height), 0.0f, 1.0f});
  mCommandBuffers[mFrameIdx].setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), mSwapChainDetails.extent));

  mCommandBuffers[mFrameIdx].bindVertexBuffers(0, *mVertexBuffer, {0});
  mCommandBuffers[mFrameIdx].bindIndexBuffer(mIndexBuffer, 0, vk::IndexType::eUint16);
  mCommandBuffers[mFrameIdx].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, mPipelineLayout, 0, *mDescriptorSets[mFrameIdx], nullptr);
  mCommandBuffers[mFrameIdx].drawIndexed(indices.size(), 1, 0, 0, 0);

  mCommandBuffers[mFrameIdx].endRendering();

  transition_image_layout(mSwapChainImages[imageIdx],
                          mCommandBuffers[mFrameIdx],
                          vk::ImageLayout::eColorAttachmentOptimal,
                          vk::ImageLayout::ePresentSrcKHR,
                          vk::AccessFlagBits2::eColorAttachmentWrite,
                          {},
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                          vk::PipelineStageFlagBits2::eBottomOfPipe,
                          vk::ImageAspectFlagBits::eColor);
  mCommandBuffers[mFrameIdx].end();
}

void Renderer::createSyncObjects() {
  assert(mPresentCompleteSems.empty() && mRenderCompleteSems.empty() && mDrawFences.empty());

  for (size_t i = 0; i < mSwapChainImages.size(); i++) {
    mRenderCompleteSems.emplace_back(mDevice, vk::SemaphoreCreateInfo{});
  }

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
    mPresentCompleteSems.emplace_back(mDevice, vk::SemaphoreCreateInfo{});
    mDrawFences.emplace_back(mDevice, vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled});
  }
}

void Renderer::cleanupSwapChain() {
  mSwapChainImageViews.clear();
  mSwapChain = nullptr;
}

void Renderer::recreateSwapChain() {
  uint32_t width, height;
  mFrameBufferSizeCallback(width, height);
  while (width == 0 || height == 0) {
    MAPLE_DEBUG("minimized...");
    mFrameBufferSizeCallback(width, height);
  }
  mDevice.waitIdle();

  cleanupSwapChain();

  createSwapChain();
  createImageViews();
  createDepthResources();
}

uint32_t Renderer::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
  auto memProperties = mPhysicalDevice.getMemoryProperties();

  for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++)
    if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) return i;

  MAPLE_FATAL("Failed to find suitable memory type");
}

void Renderer::createBuffer(vk::DeviceSize size,
                            vk::BufferUsageFlags usage,
                            vk::MemoryPropertyFlags properties,
                            vk::raii::Buffer& buffer,
                            vk::raii::DeviceMemory& bufferMemory) {
  vk::BufferCreateInfo bufferInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};
  buffer = vk::raii::Buffer(mDevice, bufferInfo);
  vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
  vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size,
                                   .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
  bufferMemory = vk::raii::DeviceMemory(mDevice, allocInfo);
  buffer.bindMemory(*bufferMemory, 0);
}

void Renderer::copyBuffer(vk::raii::Buffer& srcBuffer, vk::raii::Buffer& dstBuffer, vk::DeviceSize size) {
  auto commandCopyBuffer = beginSingleTimeCommands();
  vk::BufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = size};
  commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer, copyRegion);
  endSingleTimeCommands(commandCopyBuffer);
}

void Renderer::createImage(glm::u32vec2 size,
                           vk::Format format,
                           vk::ImageTiling tiling,
                           vk::ImageUsageFlags usage,
                           vk::MemoryPropertyFlags properties,
                           vk::raii::Image& image,
                           vk::raii::DeviceMemory& imageMemory) {
  vk::ImageCreateInfo imageInfo{.imageType = vk::ImageType::e2D,
                                .format = format,
                                .extent = {size.x, size.y, 1},
                                .mipLevels = 1,
                                .arrayLayers = 1,
                                .samples = vk::SampleCountFlagBits::e1,
                                .tiling = tiling,
                                .usage = usage,
                                .sharingMode = vk::SharingMode::eExclusive};

  image = vk::raii::Image(mDevice, imageInfo);

  vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
  vk::MemoryAllocateInfo allocInfo{.allocationSize = memRequirements.size,
                                   .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
  imageMemory = vk::raii::DeviceMemory(mDevice, allocInfo);
  image.bindMemory(imageMemory, 0);
}

}  // namespace maple