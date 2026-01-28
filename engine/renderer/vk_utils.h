#pragma once
#include <vulkan/vulkan.h>

#include <optional>
#include <ranges>

#define VK_CASE_STR(x) \
  case x:              \
    return #x

VkShaderModule CreateShaderModule(VkDevice device, const std::vector<char>& code) {
  VkShaderModuleCreateInfo createInfo{
      .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
      .codeSize = code.size(),
      .pCode = reinterpret_cast<const uint32_t*>(code.data()),
  };

  VkShaderModule shaderModule;
  if (vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) MAPLE_FATAL("Failed to create shader module");

  return shaderModule;
}

VkExtent2D ChooseOptimalSwapExtent(const VkSurfaceCapabilitiesKHR& capabilities, uint32_t framebufferWidth, uint32_t framebufferHeight) {
  if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max()) return capabilities.currentExtent;

  return VkExtent2D{
      .width = std::clamp(framebufferWidth, capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
      .height = std::clamp(framebufferHeight, capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
  };
}

uint32_t ChooseOptimalPresentMode(const std::vector<VkPresentModeKHR>& presentModes) {
  uint32_t fifoIdx;

  for (auto [i, mode] : std::views::enumerate(presentModes)) {
    if (mode == VK_PRESENT_MODE_MAILBOX_KHR) return i;
    if (mode == VK_PRESENT_MODE_FIFO_KHR) fifoIdx = i;
  }

  // VK_PRESENT_MODE_FIFO_KHR above is guaranteed by the spec to be available (hopefully :))
  return fifoIdx;
}

uint32_t ChooseOptimalSurfaceFormat(const std::vector<VkSurfaceFormatKHR>& availableFormats) {
  for (auto [i, format] : std::views::enumerate(availableFormats))
    if (format.format == VK_FORMAT_B8G8R8A8_SRGB && format.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) return i;
  return 0;
}

struct QueueCapabilities {
  uint32_t QueueCount;
  bool Graphics, Compute, Transfer, Sparse_binding, Protected, Video_decode, Video_encode, Optical_flow, Present;
};

enum class GraphicsQueueCapabilityType : uint8_t {
  GRAPHICS,
  COMPUTE,
  TRANSFER,
  SPARSE_BINDING,
  PROTECTED,
  VIDEO_DECODE,
  VIDEO_ENCODE,
  OPTICAL_FLOW,
  PRESENT,
};

QueueCapabilities GetQueueCapabilities(VkPhysicalDevice dev, VkSurfaceKHR surface, uint32_t queueFamilyIdx,
                                       VkQueueFamilyProperties queueFamilyProperties) {
  VkBool32 presentSupport = false;
  vkGetPhysicalDeviceSurfaceSupportKHR(dev, queueFamilyIdx, surface, &presentSupport);

  const auto flags = queueFamilyProperties.queueFlags;
  return QueueCapabilities{
      .QueueCount = queueFamilyProperties.queueCount,
      .Graphics = static_cast<bool>(flags & VK_QUEUE_GRAPHICS_BIT),
      .Compute = static_cast<bool>(flags & VK_QUEUE_COMPUTE_BIT),
      .Transfer = static_cast<bool>(flags & VK_QUEUE_TRANSFER_BIT),
      .Sparse_binding = static_cast<bool>(flags & VK_QUEUE_SPARSE_BINDING_BIT),
      .Protected = static_cast<bool>(flags & VK_QUEUE_PROTECTED_BIT),
      .Video_decode = static_cast<bool>(flags & VK_QUEUE_VIDEO_DECODE_BIT_KHR),
      .Video_encode = static_cast<bool>(flags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR),
      .Optical_flow = static_cast<bool>(flags & VK_QUEUE_OPTICAL_FLOW_BIT_NV),
      .Present = static_cast<bool>(presentSupport),
  };
}

struct PhysicalDevice {
  VkPhysicalDevice dev;
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceFeatures features;
  std::vector<VkExtensionProperties> extensions;

  std::vector<VkQueueFamilyProperties> queueFamiliesProperties;
  std::vector<QueueCapabilities> queueFamiliesCapabilities;

  VkSurfaceCapabilitiesKHR surfaceCapabilities;
  std::vector<VkSurfaceFormatKHR> surfaceFormats;
  std::vector<VkPresentModeKHR> presentModes;
};

std::optional<size_t> GetQueueFamilyIdxWithCapability(std::vector<QueueCapabilities> familyCapabilities, GraphicsQueueCapabilityType filter) {
  for (auto [i, caps] : std::views::enumerate(familyCapabilities)) {
    switch (filter) {
      case GraphicsQueueCapabilityType::GRAPHICS:
        if (caps.Graphics) return i;
      case GraphicsQueueCapabilityType::COMPUTE:
        if (caps.Compute) return i;
      case GraphicsQueueCapabilityType::TRANSFER:
        if (caps.Transfer) return i;
      case GraphicsQueueCapabilityType::SPARSE_BINDING:
        if (caps.Sparse_binding) return i;
      case GraphicsQueueCapabilityType::PROTECTED:
        if (caps.Protected) return i;
      case GraphicsQueueCapabilityType::VIDEO_DECODE:
        if (caps.Video_decode) return i;
      case GraphicsQueueCapabilityType::VIDEO_ENCODE:
        if (caps.Video_encode) return i;
      case GraphicsQueueCapabilityType::OPTICAL_FLOW:
        if (caps.Optical_flow) return i;
      case GraphicsQueueCapabilityType::PRESENT:
        if (caps.Present) return i;
    }
  }
  return std::nullopt;
}

std::vector<PhysicalDevice> GetPhysicalDevices(VkInstance instance, VkSurfaceKHR surface) {
  std::vector<PhysicalDevice> devices;

  {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    devices.resize(deviceCount);
    std::vector<VkPhysicalDevice> dev(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, dev.data());
    for (auto [i, _] : std::views::enumerate(dev)) devices[i].dev = dev[i];
  }

  for (auto [i, dev] : std::views::enumerate(devices)) {
    uint32_t count = 0;

    vkGetPhysicalDeviceProperties(dev.dev, &dev.properties);
    vkGetPhysicalDeviceFeatures(dev.dev, &dev.features);

    vkEnumerateDeviceExtensionProperties(dev.dev, nullptr, &count, nullptr);
    dev.extensions.resize(count);
    vkEnumerateDeviceExtensionProperties(dev.dev, nullptr, &count, dev.extensions.data());

    vkGetPhysicalDeviceQueueFamilyProperties(dev.dev, &count, nullptr);
    dev.queueFamiliesProperties.resize(count);
    dev.queueFamiliesCapabilities.resize(count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev.dev, &count, dev.queueFamiliesProperties.data());
    for (size_t j = 0; j < count; j++) dev.queueFamiliesCapabilities[j] = GetQueueCapabilities(dev.dev, surface, j, dev.queueFamiliesProperties[j]);

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(dev.dev, surface, &dev.surfaceCapabilities);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev.dev, surface, &count, nullptr);
    dev.surfaceFormats.resize(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(dev.dev, surface, &count, dev.surfaceFormats.data());

    vkGetPhysicalDeviceSurfacePresentModesKHR(dev.dev, surface, &count, nullptr);
    dev.presentModes.resize(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(dev.dev, surface, &count, dev.presentModes.data());
  }

  return devices;
}

const char* vkPhysicalDeviceTypeToString(VkPhysicalDeviceType type) {
  switch (type) {
    VK_CASE_STR(VK_PHYSICAL_DEVICE_TYPE_OTHER);
    VK_CASE_STR(VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU);
    VK_CASE_STR(VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU);
    VK_CASE_STR(VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU);
    VK_CASE_STR(VK_PHYSICAL_DEVICE_TYPE_CPU);
    default:
      return "UNKNOWN";
  }
}

VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo,
                                      const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pDebugMessenger) {
  auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
  if (func != nullptr) {
    return func(instance, pCreateInfo, pAllocator, pDebugMessenger);
  } else {
    return VK_ERROR_EXTENSION_NOT_PRESENT;
  }
}

void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT debugMessenger, const VkAllocationCallbacks* pAllocator) {
  auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");
  if (func != nullptr) {
    func(instance, debugMessenger, pAllocator);
  }
}

void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo, PFN_vkDebugUtilsMessengerCallbackEXT callback) {
  createInfo = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = callback,
  };
}