#pragma once
#include <vulkan/vulkan.h>

#include <optional>

#define VK_CASE_STR(x) \
  case x:              \
    return #x

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

std::optional<size_t> GetQueueFamilyIdxWithCapability(std::vector<QueueCapabilities> familyCapabilities,
                                                      GraphicsQueueCapabilityType filter) {
  for (size_t i = 0; i < familyCapabilities.size(); i++) {
    const auto caps = familyCapabilities[i];
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

  uint32_t deviceCount = 0;
  {
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    devices.resize(deviceCount);
    std::vector<VkPhysicalDevice> dev(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, dev.data());
    for (size_t i = 0; i < deviceCount; i++) devices[i].dev = dev[i];
  }

  for (size_t i = 0; i < deviceCount; i++) {
    uint32_t count = 0;

    vkGetPhysicalDeviceProperties(devices[i].dev, &devices[i].properties);
    vkGetPhysicalDeviceFeatures(devices[i].dev, &devices[i].features);

    vkEnumerateDeviceExtensionProperties(devices[i].dev, nullptr, &count, nullptr);
    devices[i].extensions.resize(count);
    vkEnumerateDeviceExtensionProperties(devices[i].dev, nullptr, &count, devices[i].extensions.data());

    vkGetPhysicalDeviceQueueFamilyProperties(devices[i].dev, &count, nullptr);
    devices[i].queueFamiliesProperties.resize(count);
    devices[i].queueFamiliesCapabilities.resize(count);
    vkGetPhysicalDeviceQueueFamilyProperties(devices[i].dev, &count, devices[i].queueFamiliesProperties.data());
    for (size_t j = 0; j < count; j++)
      devices[i].queueFamiliesCapabilities[j] = GetQueueCapabilities(devices[i].dev, surface, j, devices[i].queueFamiliesProperties[j]);

    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(devices[i].dev, surface, &devices[i].surfaceCapabilities);
    vkGetPhysicalDeviceSurfaceFormatsKHR(devices[i].dev, surface, &count, nullptr);
    devices[i].surfaceFormats.resize(count);
    vkGetPhysicalDeviceSurfaceFormatsKHR(devices[i].dev, surface, &count, devices[i].surfaceFormats.data());

    vkGetPhysicalDeviceSurfacePresentModesKHR(devices[i].dev, surface, &count, nullptr);
    devices[i].presentModes.resize(count);
    vkGetPhysicalDeviceSurfacePresentModesKHR(devices[i].dev, surface, &count, devices[i].presentModes.data());
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