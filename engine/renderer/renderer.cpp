#include <engine/logging/log_macros.h>
#include <engine/renderer/renderer.h>
#include <vulkan/vulkan.h>

#include <map>

#include "vk_utils.h"

const std::vector<const char*> validationLayers = {
#ifndef NDEBUG
    "VK_LAYER_KHRONOS_validation"
#endif
};

static VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                          VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                          const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
  MAPLE_DEBUG("Vulkan validation layer: {}", pCallbackData->pMessage);

  return VK_FALSE;
}

namespace maple {

struct Renderer::Impl {
  VkInstance mVkInstance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT mDebugMessenger = VK_NULL_HANDLE;
  size_t mSelectedDeviceIdx = 0;

  std::vector<VkExtensionProperties> mAvailableInstanceExtensions;
  std::vector<VkLayerProperties> mAvailableInstanceLayers;
  std::vector<VkPhysicalDevice> mPhysicalDevices;
  std::vector<PhysicalDeviceData> mPhysicalDevicesData;

  void Init(const uint32_t requiredExtensionsCount, const char* const* requiredExtensions) {
    MAPLE_INFO("Initializing Renderer...");
    probeInstanceExtensions();
    probeInstanceLayers();
    createVulkanInstance(requiredExtensionsCount, requiredExtensions);
    setupDebugCallback();
    probePhysicalDevices();
    selectPhysicalDevice();
  }

  void Destroy() {
    MAPLE_INFO("Cleaning Renderer...");
    if (validationLayers.size() > 0) DestroyDebugUtilsMessengerEXT(mVkInstance, mDebugMessenger, nullptr);

    vkDestroyInstance(mVkInstance, nullptr);
  }

  void createVulkanInstance(const uint32_t requiredExtensionsCount, const char* const* windowRequiredExtensions) {
    for (const char* layerName : validationLayers) {
      bool found = false;

      for (const auto& l : mAvailableInstanceLayers) {
        if (strcmp(layerName, l.layerName) == 0) {
          found = true;
          break;
        }
      }

      if (!found) MAPLE_FATAL("Failed to find required Vulkan instance layer \"{}\"", layerName);
    }

    std::vector<const char*> requiredExtensions(windowRequiredExtensions, windowRequiredExtensions + requiredExtensionsCount);
    if (validationLayers.size() > 0) requiredExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);

    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Application name";                 // TODO: replace
    appInfo.applicationVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);  // TODO: replace
    appInfo.pEngineName = "Maple Engine";
    appInfo.engineVersion = VK_MAKE_API_VERSION(0, 1, 0, 0);  // TODO: replace
    appInfo.apiVersion = VK_API_VERSION_1_0;

    VkDebugUtilsMessengerCreateInfoEXT debugCreateInfo{};
    populateDebugMessengerCreateInfo(debugCreateInfo, vulkanDebugCallback);

    VkInstanceCreateInfo createInfo{
        .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        .pNext = validationLayers.size() > 0 ? &debugCreateInfo : nullptr,
        .pApplicationInfo = &appInfo,
        .enabledLayerCount = static_cast<uint32_t>(validationLayers.size()),
        .ppEnabledLayerNames = validationLayers.size() > 0 ? validationLayers.data() : nullptr,
        .enabledExtensionCount = static_cast<uint32_t>(requiredExtensions.size()),
        .ppEnabledExtensionNames = requiredExtensions.data(),
    };

    if (vkCreateInstance(&createInfo, nullptr, &mVkInstance) != VK_SUCCESS) MAPLE_FATAL("Failed to create vulkan instance");
  }

  void setupDebugCallback() {
    if (validationLayers.size() < 1) return;

    VkDebugUtilsMessengerCreateInfoEXT createInfo;
    populateDebugMessengerCreateInfo(createInfo, vulkanDebugCallback);

    if (CreateDebugUtilsMessengerEXT(mVkInstance, &createInfo, nullptr, &mDebugMessenger) != VK_SUCCESS)
      MAPLE_FATAL("Failed to create a Vulkan debug messenger");
  }

  void selectPhysicalDevice() {
    std::multimap<int32_t, size_t> candidates;

    for (size_t i = 0; i < mPhysicalDevices.size(); i++) {
      int32_t score = 0;

      switch (mPhysicalDevicesData[i].properties.deviceType) {
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:
          score += 2000;
          break;
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU:
          score += 500;
          break;
        case VK_PHYSICAL_DEVICE_TYPE_CPU:
          score += 50;
          break;
      }

      score += mPhysicalDevicesData[i].properties.limits.maxImageDimension2D;

      candidates.insert(std::make_pair(score, i));
    }

    // TODO: disqualify devices with no graphics queue (maybe compute queue aswell)
    // TODO: weigh memory heaps, push constants, and maxFramebuffer dimensions

    for (const auto& e : candidates) MAPLE_INFO("\tScore {}: {}", mPhysicalDevicesData[e.second].properties.deviceName, e.first);

    mSelectedDeviceIdx = candidates.rbegin()->second;
    MAPLE_INFO("Selected Graphics Device {}", mPhysicalDevicesData[mSelectedDeviceIdx].properties.deviceName);
  }

  void probePhysicalDevices() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(mVkInstance, &count, nullptr);
    if (count == 0) MAPLE_FATAL("Failed to find graphics device with Vulkan support");

    mPhysicalDevices.resize(count);
    mPhysicalDevicesData.resize(count);
    vkEnumeratePhysicalDevices(mVkInstance, &count, mPhysicalDevices.data());

    MAPLE_INFO("Available Vulkan devices ({}):", count);
    for (size_t i = 0; i < mPhysicalDevices.size(); i++) {
      mPhysicalDevicesData[i] = GetPhysicalDeviceData(mPhysicalDevices[i]);

      MAPLE_INFO("\t{}: {}", mPhysicalDevicesData[i].properties.deviceName,
                 vkPhysicalDeviceTypeToString(mPhysicalDevicesData[i].properties.deviceType));

      for (const auto& v : mPhysicalDevicesData[i].queueFamilies) {
        const auto caps = GetGraphicsQueueCapabilities(v.queueFlags);
        MAPLE_INFO(
            "\t\tQueue Family Capabilites (Count: {}): Compute: {} Graphics: {} Optical_flow: {} Protected: {} Sparse_binding: {} "
            "Transfer: {} Video_decode: {} Video_encode: {}",
            v.queueCount, caps.Compute, caps.Graphics, caps.Optical_flow, caps.Protected, caps.Sparse_binding, caps.Transfer,
            caps.Video_decode, caps.Video_encode);
      }
    }
  }

  void probeInstanceExtensions() {
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    mAvailableInstanceExtensions.resize(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, mAvailableInstanceExtensions.data());

    MAPLE_INFO("Available Vulkan instance extensions ({}):", count);
    for (const auto& e : mAvailableInstanceExtensions) MAPLE_INFO("\t{}: {}", e.extensionName, e.specVersion);
  }

  void probeInstanceLayers() {
    uint32_t count = 0;
    vkEnumerateInstanceLayerProperties(&count, nullptr);
    mAvailableInstanceLayers.resize(count);
    vkEnumerateInstanceLayerProperties(&count, mAvailableInstanceLayers.data());

    MAPLE_INFO("Available Vulkan instance layers ({}):", count);
    for (const auto& l : mAvailableInstanceLayers)
      MAPLE_INFO("\t{}: {}, {}, ({})", l.layerName, l.specVersion, l.implementationVersion, l.description);
  }
};  // namespace maple

Renderer::Renderer() : mPimpl(std::make_unique<Impl>()) {}
Renderer::~Renderer() {}
void Renderer::Init(const uint32_t requiredExtensionsCount, const char* const* requiredExtensions) {
  mPimpl->Init(requiredExtensionsCount, requiredExtensions);
}
void Renderer::Destroy() { mPimpl->Destroy(); }

}  // namespace maple
