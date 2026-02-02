#include <engine/file/file.h>
#include <engine/logging/log_macros.h>
#include <engine/renderer/renderer.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <map>
#include <utility>
#include <vector>
#include <vulkan/vulkan_enums.hpp>
#include <vulkan/vulkan_handles.hpp>
#include <vulkan/vulkan_raii.hpp>
#include <vulkan/vulkan_structs.hpp>
#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <ranges>

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
  auto vk11Features = featureChain.get<vk::PhysicalDeviceVulkan11Features>();
  auto vk13Features = featureChain.get<vk::PhysicalDeviceVulkan13Features>();
  auto extDynStateFeatures = featureChain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

  if (!vk11Features.shaderDrawParameters || !vk13Features.dynamicRendering || !vk13Features.synchronization2 ||
      !extDynStateFeatures.extendedDynamicState)
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
                             vk::PipelineStageFlags2 dstStageMask) {
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
    .subresourceRange = {.aspectMask = vk::ImageAspectFlagBits::eColor, .baseMipLevel = 0, .levelCount = 1, .baseArrayLayer = 0, .layerCount = 1}};
  vk::DependencyInfo dependencyInfo = {.dependencyFlags = {}, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &barrier};
  commandBuffer.pipelineBarrier2(dependencyInfo);
}

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
  createGraphicsPipeline();
  createCommandPool();
  createCommandBuffer();
  createSyncObjects();
}

void Renderer::DrawFrame() {
  auto fenceResult = mDevice.waitForFences(*mDrawFence, vk::True, UINT64_MAX);

  auto [result, imageIdx] = mSwapChain.acquireNextImage(UINT64_MAX, *mPresentCompleteSem, nullptr);
  recordCommandBuffer(imageIdx);

  mDevice.resetFences(*mDrawFence);

  vk::PipelineStageFlags waitDestinationStageMask(vk::PipelineStageFlagBits::eColorAttachmentOutput);
  const vk::SubmitInfo submitInfo{
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &*mPresentCompleteSem,
    .pWaitDstStageMask = &waitDestinationStageMask,
    .commandBufferCount = 1,
    .pCommandBuffers = &*mCommandBuffer,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &*mRenderCompleteSem,
  };

  mGraphicsQueue.submit(submitInfo, *mDrawFence);

  const vk::PresentInfoKHR presentInfoKHR{
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &*mRenderCompleteSem,
    .swapchainCount = 1,
    .pSwapchains = &*mSwapChain,
    .pImageIndices = &imageIdx,
  };

  auto presentResult = mPresentQueue.presentKHR(presentInfoKHR);
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
      {},                                                    // vk::PhysicalDeviceFeatures2 (empty for now)
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

  for (const auto& image : mSwapChainImages) {
    imageViewCreateInfo.image = image;
    mSwapChainImageViews.emplace_back(mDevice.createImageView(imageViewCreateInfo));
  }
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

  vk::PipelineVertexInputStateCreateInfo vertexInputInfo;

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
    .frontFace = vk::FrontFace::eClockwise,
    .depthBiasEnable = vk::False,
    .depthBiasSlopeFactor = 1.0f,
    .lineWidth = 1.0f,
  };

  vk::PipelineMultisampleStateCreateInfo multisampling{.rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = vk::False};

  vk::PipelineColorBlendAttachmentState colorBlendAttachment{
    .blendEnable = vk::False,
    .colorWriteMask =
      vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA,
  };

  vk::PipelineColorBlendStateCreateInfo colorBlending{
    .logicOpEnable = vk::False, .logicOp = vk::LogicOp::eCopy, .attachmentCount = 1, .pAttachments = &colorBlendAttachment};

  vk::PipelineLayoutCreateInfo pipelineLayoutInfo{.setLayoutCount = 0, .pushConstantRangeCount = 0};
  mPipelineLayout = vk::raii::PipelineLayout(mDevice, pipelineLayoutInfo);

  vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{.colorAttachmentCount = 1, .pColorAttachmentFormats = &mSwapChainDetails.format.format};
  vk::GraphicsPipelineCreateInfo pipelineInfo{
    .pNext = &pipelineRenderingCreateInfo,
    .stageCount = 2,
    .pStages = shaderStages,
    .pVertexInputState = &vertexInputInfo,
    .pInputAssemblyState = &inputAssembly,
    .pViewportState = &viewportState,
    .pRasterizationState = &rasterizer,
    .pMultisampleState = &multisampling,
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

void Renderer::createCommandBuffer() {
  vk::CommandBufferAllocateInfo allocInfo{
    .commandPool = mCommandPool,
    .level = vk::CommandBufferLevel::ePrimary,
    .commandBufferCount = 1,
  };

  mCommandBuffer = std::move(vk::raii::CommandBuffers(mDevice, allocInfo).front());
}

void Renderer::recordCommandBuffer(uint32_t imageIdx) {
  mCommandBuffer.begin({});
  transition_image_layout(mSwapChainImages[imageIdx],
                          mCommandBuffer,
                          vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eColorAttachmentOptimal,
                          {},
                          vk::AccessFlagBits2::eColorAttachmentWrite,
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput);

  vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
  vk::RenderingAttachmentInfo attachmentInfo = {
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
    .pColorAttachments = &attachmentInfo,
  };

  mCommandBuffer.beginRendering(renderingInfo);

  mCommandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, mGraphicsPipeline);

  mCommandBuffer.setViewport(
    0, vk::Viewport{0.0f, 0.0f, static_cast<float>(mSwapChainDetails.extent.width), static_cast<float>(mSwapChainDetails.extent.height), 0.0f, 1.0f});
  mCommandBuffer.setScissor(0, vk::Rect2D(vk::Offset2D(0, 0), mSwapChainDetails.extent));

  mCommandBuffer.draw(3, 1, 0, 0);

  mCommandBuffer.endRendering();

  transition_image_layout(mSwapChainImages[imageIdx],
                          mCommandBuffer,
                          vk::ImageLayout::eColorAttachmentOptimal,
                          vk::ImageLayout::ePresentSrcKHR,
                          vk::AccessFlagBits2::eColorAttachmentWrite,
                          {},
                          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                          vk::PipelineStageFlagBits2::eBottomOfPipe);
  mCommandBuffer.end();
}

void Renderer::createSyncObjects() {
  vk::SemaphoreCreateInfo semaphoreInfo;
  vk::FenceCreateInfo fenceInfo{.flags = vk::FenceCreateFlagBits::eSignaled};

  mPresentCompleteSem = vk::raii::Semaphore(mDevice, semaphoreInfo);
  mRenderCompleteSem = vk::raii::Semaphore(mDevice, semaphoreInfo);
  mDrawFence = vk::raii::Fence(mDevice, fenceInfo);
}

}  // namespace maple