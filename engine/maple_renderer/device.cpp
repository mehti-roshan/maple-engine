#include "device.h"

#include <vulkan/vulkan_core.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <ranges>
#include <vector>

#include "chkvk_macro.h"
#include "feature_chain.h"
#include "log_macros.h"

namespace maple {

static bool PhysicalDeviceSuitableAndScore(
  VkPhysicalDevice device, VkSurfaceKHR surface, float& score, uint32_t& graphicsQFamilyIdx, uint32_t& presentQFamilyIdx) {
  // check api version support
  VkPhysicalDeviceProperties properties;
  vkGetPhysicalDeviceProperties(device, &properties);
  if (properties.apiVersion < VK_API_VERSION_1_3) return false;

  // swapchain extension support
  bool swapchain = false;
  uint32_t extensionCount = 0;
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, nullptr);
  std::vector<VkExtensionProperties> extensions(extensionCount);
  vkEnumerateDeviceExtensionProperties(device, nullptr, &extensionCount, extensions.data());
  for (auto& ext : extensions) {
    if (strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0) {
      swapchain = true;
      break;
    }
  }
  if (!swapchain) return false;

  // Check features
  FeatureChain chain{};
  vkGetPhysicalDeviceFeatures2(device, &chain.features);
  chain.CheckRequired();

  // Check graphics and present queue family support
  uint32_t queueCount = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, nullptr);
  std::vector<VkQueueFamilyProperties> qFamilyProps(queueCount);
  vkGetPhysicalDeviceQueueFamilyProperties(device, &queueCount, qFamilyProps.data());

  int32_t graphicsFamilyIdx = -1;
  int32_t presentFamilyIdx = -1;
  for (auto [i, qProps] : std::views::enumerate(qFamilyProps)) {
    if (qProps.queueFlags & VK_QUEUE_GRAPHICS_BIT) graphicsFamilyIdx = i;

    VkBool32 present = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(device, i, surface, &present);
    if (present) presentFamilyIdx = i;
  }
  if (graphicsFamilyIdx == -1 || presentFamilyIdx == -1) return false;
  graphicsQFamilyIdx = graphicsFamilyIdx;
  presentQFamilyIdx = presentFamilyIdx;

  size_t totalVRAMBytes = 0;
  VkPhysicalDeviceMemoryProperties memoryProps{};
  vkGetPhysicalDeviceMemoryProperties(device, &memoryProps);
  auto heaps = std::vector<VkMemoryHeap>(memoryProps.memoryHeaps, memoryProps.memoryHeaps + memoryProps.memoryHeapCount);
  for (const auto& heap : heaps)
    if (heap.flags & VkMemoryHeapFlagBits::VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
      totalVRAMBytes += heap.size;
    }

  auto totalVRAMGB = ((totalVRAMBytes / 1024) / 1024) / 1024;
  if (properties.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 1000.0f;
  score += totalVRAMGB;

  return true;
}

static VkPresentModeKHR ChoosePresentMode(VkPhysicalDevice device, VkSurfaceKHR surface) {
  uint32_t count;
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, nullptr);
  std::vector<VkPresentModeKHR> presentModes(count);
  vkGetPhysicalDeviceSurfacePresentModesKHR(device, surface, &count, presentModes.data());
  for (auto mode : presentModes)
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR) return mode;
  return VK_PRESENT_MODE_FIFO_KHR;
}

static constexpr VkFormat SwapchainFormat = VK_FORMAT_B8G8R8A8_SRGB;

static VkImageViewCreateInfo CreateSwapchainImgViewCI(VkImage img) {
  VkImageViewCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
  ci.viewType = VK_IMAGE_VIEW_TYPE_2D;
  ci.format = SwapchainFormat;
  ci.components = {VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY, VK_COMPONENT_SWIZZLE_IDENTITY};
  ci.subresourceRange = {.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT, .levelCount = 1, .layerCount = 1};
  ci.image = img;
  return ci;
}

bool Device::CreateSwapChain(FrameBufferSizeCallback frameBufferSizeFunc, std::string& err) {
  VkSurfaceCapabilitiesKHR surfaceCaps{};
  chkvk(
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(mPhysicalDevice, mSurface, &surfaceCaps), err, "failed to get physical device surface capabilities");

  // if fullscreen, exclusive fullscreen etc, the surface itself forces us to use a specific size
  // else we must query the window size via callback and set it correctly
  mSwapchainExtent = surfaceCaps.currentExtent;
  if (surfaceCaps.currentExtent.width == std::numeric_limits<uint32_t>::max()) {
    int32_t sizeX, sizeY;
    frameBufferSizeFunc(sizeX, sizeY);
    mSwapchainExtent = {
      .width = std::clamp(uint32_t(sizeX), surfaceCaps.minImageExtent.width, surfaceCaps.maxImageExtent.width),
      .height = std::clamp(uint32_t(sizeY), surfaceCaps.minImageExtent.height, surfaceCaps.maxImageExtent.height),
    };
  }

  VkSwapchainCreateInfoKHR swapchainCI{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  swapchainCI.surface = mSurface;
  swapchainCI.imageFormat = SwapchainFormat;
  swapchainCI.imageColorSpace = VK_COLORSPACE_SRGB_NONLINEAR_KHR;
  swapchainCI.imageExtent = mSwapchainExtent;
  swapchainCI.imageArrayLayers = 1;
  swapchainCI.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
  swapchainCI.preTransform = VK_SURFACE_TRANSFORM_IDENTITY_BIT_KHR;
  swapchainCI.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
  swapchainCI.presentMode = ChoosePresentMode(mPhysicalDevice, mSurface);

  // 0 on surfaceCaps.maxImageCount means unlimited
  uint32_t maxImageCount = surfaceCaps.maxImageCount == 0 ? std::numeric_limits<uint32_t>::max() : surfaceCaps.maxImageCount;
  swapchainCI.minImageCount = std::clamp(3u, surfaceCaps.minImageCount, maxImageCount);

  if (mGraphicsQueue == mPresentQueue) {
    swapchainCI.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
  } else {
    swapchainCI.imageSharingMode = VK_SHARING_MODE_CONCURRENT;
    uint32_t indices[2] = {mGraphicsQFamilyIdx, mPresentQFamilyIdx};
    swapchainCI.queueFamilyIndexCount = 2;
    swapchainCI.pQueueFamilyIndices = indices;
  }

  chkvk(vkCreateSwapchainKHR(mDevice, &swapchainCI, nullptr, &mSwapchain), err, "failed to create swapchain");

  uint32_t count = 0;
  chkvk(vkGetSwapchainImagesKHR(mDevice, mSwapchain, &count, nullptr), err, "failed to get swapchain images");
  std::vector<VkImage> images(count);
  chkvk(vkGetSwapchainImagesKHR(mDevice, mSwapchain, &count, images.data()), err, "failed to get swapchain images");

  mSwapchainData.resize(count);
  for (uint32_t i = 0; i < count; i++) {
    mSwapchainData[i].img = images[i];

    auto imgViewCI = CreateSwapchainImgViewCI(images[i]);
    chkvk(vkCreateImageView(mDevice, &imgViewCI, nullptr, &mSwapchainData[i].view), err, "failed to create swapchain image view");

    VkSemaphoreCreateInfo semaphoreCI{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    auto result = vkCreateSemaphore(mDevice, &semaphoreCI, nullptr, &mSwapchainData[i].renderCompleteSemaphore);
    chkvk(result, err, "failed to create render complete semaphore");
  }

  return true;
}

bool Device::UpdateSwapChain(FrameBufferSizeCallback frameBufferSizeFunc, std::string& err) {
  chkvk(vkDeviceWaitIdle(mDevice), err, "failed wait for device idle");
  VkSurfaceCapabilitiesKHR surfaceCaps{};
  chkvk(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(mPhysicalDevice, mSurface, &surfaceCaps), err, "failed to get surface capabilities");

  int32_t x, y;
  frameBufferSizeFunc(x, y);
  while (x == 0 || y == 0) {
    MAPLE_DEBUG("minimized...");
    frameBufferSizeFunc(x, y);
  }
  VkSwapchainCreateInfoKHR swapchainCI{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
  swapchainCI.oldSwapchain = mSwapchain;
  swapchainCI.imageExtent = {.width = static_cast<uint32_t>(x), .height = static_cast<uint32_t>(y)};

  chkvk(vkCreateSwapchainKHR(mDevice, &swapchainCI, nullptr, &mSwapchain), err, "failed to recreate swapchain");
  for (auto& data : mSwapchainData) {
    vkDestroyImageView(mDevice, data.view, nullptr);
    vkDestroySemaphore(mDevice, data.renderCompleteSemaphore, nullptr);
  }

  uint32_t imageCount = 0;
  chkvk(vkGetSwapchainImagesKHR(mDevice, mSwapchain, &imageCount, nullptr), err, "failed to get swapchain images");
  std::vector<VkImage> images(imageCount);
  chkvk(vkGetSwapchainImagesKHR(mDevice, mSwapchain, &imageCount, images.data()), err, "failed to get swapchain images");

  mSwapchainData.resize(imageCount);
  for (auto i = 0; i < imageCount; i++) {
    mSwapchainData[i].img = images[i];

    auto imgViewCI = CreateSwapchainImgViewCI(images[i]);
    chkvk(vkCreateImageView(mDevice, &imgViewCI, nullptr, &mSwapchainData[i].view), err, "failed to create swapchain image view");

    VkSemaphoreCreateInfo semaphoreCI{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    auto result = vkCreateSemaphore(mDevice, &semaphoreCI, nullptr, &mSwapchainData[i].renderCompleteSemaphore);
    chkvk(result, err, "failed to recreate render complete semaphore");
  }

  vkDestroySwapchainKHR(mDevice, swapchainCI.oldSwapchain, nullptr);

  return true;
}

bool Device::Init(VkInstance instance, const VkSurfaceKHR surface, FrameBufferSizeCallback frameBufferSizeFunc, std::string& err) {
  mSurface = surface;

  // Select physical device
  uint32_t count = 0;
  chkvk(vkEnumeratePhysicalDevices(instance, &count, nullptr), err, "failed to enumerate physical devices");
  std::vector<VkPhysicalDevice> devices(count);
  chkvk(vkEnumeratePhysicalDevices(instance, &count, devices.data()), err, "failed to enumerate physical devices");

  float selectedScore = -std::numeric_limits<float>::infinity();
  for (VkPhysicalDevice gpu : devices) {
    float score = 0.0f;
    if (!PhysicalDeviceSuitableAndScore(gpu, mSurface, score, mGraphicsQFamilyIdx, mPresentQFamilyIdx)) continue;
    if (score > selectedScore) mPhysicalDevice = gpu;
  }

  if (!mPhysicalDevice) {
    err = "failed to find suitable physical device";
    return false;
  }

  // Create device
  std::vector<VkDeviceQueueCreateInfo> queueInfos;

  float qPriority = 1.0f;
  VkDeviceQueueCreateInfo graphicsQueue{
    .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
    .queueFamilyIndex = mGraphicsQFamilyIdx,
    .queueCount = 1,
    .pQueuePriorities = &qPriority,
  };
  queueInfos.push_back(graphicsQueue);

  if (mPresentQFamilyIdx != mGraphicsQFamilyIdx) {
    VkDeviceQueueCreateInfo presentQueue{
      .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
      .queueFamilyIndex = mPresentQFamilyIdx,
      .queueCount = 1,
      .pQueuePriorities = &qPriority,
    };
    queueInfos.push_back(presentQueue);
  }

  std::vector<const char*> extensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};

  FeatureChain chain{};
  chain.EnableRequired();
  VkDeviceCreateInfo deviceCI{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
  deviceCI.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
  deviceCI.pQueueCreateInfos = queueInfos.data();
  deviceCI.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
  deviceCI.ppEnabledExtensionNames = extensions.data();
  deviceCI.pEnabledFeatures = nullptr;  // leave nullptr because using VkPhysicalDeviceFeatures2
  deviceCI.pNext = &chain.features;

  chkvk(vkCreateDevice(mPhysicalDevice, &deviceCI, nullptr, &mDevice), err, "failed to create vulkan device");

  // Get queues
  vkGetDeviceQueue(mDevice, mGraphicsQFamilyIdx, 0, &mGraphicsQueue);
  vkGetDeviceQueue(mDevice, mPresentQFamilyIdx, 0, &mPresentQueue);

  if (!CreateSwapChain(frameBufferSizeFunc, err)) return false;

  // Create allocator
  VmaAllocatorCreateInfo allocatorCI{};
  allocatorCI.instance = instance;
  allocatorCI.physicalDevice = mPhysicalDevice;
  allocatorCI.device = mDevice;
  allocatorCI.vulkanApiVersion = VK_API_VERSION_1_3;
  allocatorCI.flags = VK_KHR_buffer_device_address;
  chkvk(vmaCreateAllocator(&allocatorCI, &mAllocator), err, "failed to create vulkan memory allocator");

  return true;
}

void Device::Destroy() {
  if (mDevice) vkDeviceWaitIdle(mDevice);
  if (mAllocator) vmaDestroyAllocator(mAllocator);
  for (auto& data : mSwapchainData) {
    vkDestroyImageView(mDevice, data.view, nullptr);
    vkDestroySemaphore(mDevice, data.renderCompleteSemaphore, nullptr);
  }
  mSwapchainData.clear();

  if (mSwapchain) vkDestroySwapchainKHR(mDevice, mSwapchain, nullptr);
  if (mDevice) vkDestroyDevice(mDevice, nullptr);
}

}  // namespace maple