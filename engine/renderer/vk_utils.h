#pragma once
#include <vulkan/vulkan.h>

#include <optional>

#define VK_CASE_STR(x) \
  case x:              \
    return #x

struct GraphicsQueueCapabilities {
  bool Graphics, Compute, Transfer, Sparse_binding, Protected, Video_decode, Video_encode, Optical_flow;
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
};

GraphicsQueueCapabilities GetGraphicsQueueCapabilities(VkQueueFlags flags) {
  return GraphicsQueueCapabilities{
      .Graphics = static_cast<bool>(flags & VK_QUEUE_GRAPHICS_BIT),
      .Compute = static_cast<bool>(flags & VK_QUEUE_COMPUTE_BIT),
      .Transfer = static_cast<bool>(flags & VK_QUEUE_TRANSFER_BIT),
      .Sparse_binding = static_cast<bool>(flags & VK_QUEUE_SPARSE_BINDING_BIT),
      .Protected = static_cast<bool>(flags & VK_QUEUE_PROTECTED_BIT),
      .Video_decode = static_cast<bool>(flags & VK_QUEUE_VIDEO_DECODE_BIT_KHR),
      .Video_encode = static_cast<bool>(flags & VK_QUEUE_VIDEO_ENCODE_BIT_KHR),
      .Optical_flow = static_cast<bool>(flags & VK_QUEUE_OPTICAL_FLOW_BIT_NV),
  };
}

struct PhysicalDeviceData {
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceFeatures features;
  std::vector<VkQueueFamilyProperties> queueFamilies;
};

std::optional<uint32_t> GetGraphicsQueueIdxWithCapability(const PhysicalDeviceData& d, GraphicsQueueCapabilityType c) {
  for (uint32_t i = 0; i < d.queueFamilies.size(); i++) {
    const auto caps = GetGraphicsQueueCapabilities(d.queueFamilies[i].queueFlags);
    switch (c) {
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
    }
  }
  return std::nullopt;
}

PhysicalDeviceData GetPhysicalDeviceData(VkPhysicalDevice dev) {
  PhysicalDeviceData data{};
  vkGetPhysicalDeviceProperties(dev, &data.properties);
  vkGetPhysicalDeviceFeatures(dev, &data.features);
  uint32_t numQueueFamilies = 0;
  vkGetPhysicalDeviceQueueFamilyProperties(dev, &numQueueFamilies, nullptr);
  data.queueFamilies.resize(numQueueFamilies);
  vkGetPhysicalDeviceQueueFamilyProperties(dev, &numQueueFamilies, data.queueFamilies.data());
  return data;
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