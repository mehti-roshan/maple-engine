#include <engine/logging/log_macros.h>
#include <engine/renderer/renderer.h>
#include <vulkan/vulkan.h>

const std::vector<const char*> validationLayers = {
#ifndef NDEBUG
    "VK_LAYER_KHRONOS_validation"
#endif
};

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

static VKAPI_ATTR VkBool32 VKAPI_CALL vulkanDebugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                          VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                          const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData) {
  MAPLE_DEBUG("Vulkan validation layer: {}", pCallbackData->pMessage);

  return VK_FALSE;
}

void populateDebugMessengerCreateInfo(VkDebugUtilsMessengerCreateInfoEXT& createInfo) {
  createInfo = {
      .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
      .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
      .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
      .pfnUserCallback = vulkanDebugCallback,
  };
}

namespace maple {

struct Renderer::Impl {
  VkInstance mVkInstance = VK_NULL_HANDLE;
  VkDebugUtilsMessengerEXT mDebugMessenger = VK_NULL_HANDLE;

  std::vector<VkExtensionProperties> availableInstanceExtensions;
  std::vector<VkLayerProperties> availableInstanceLayers;

  void Init(const uint32_t requiredExtensionsCount, const char* const* requiredExtensions) {
    MAPLE_INFO("Initializing Renderer...");
    createVulkanInstance(requiredExtensionsCount, requiredExtensions);
    setupDebugCallback();
  }
  void Destroy() {
    MAPLE_INFO("Cleaning Renderer...");
    if (validationLayers.size() > 0) DestroyDebugUtilsMessengerEXT(mVkInstance, mDebugMessenger, nullptr);

    vkDestroyInstance(mVkInstance, nullptr);
  }

  void createVulkanInstance(const uint32_t requiredExtensionsCount, const char* const* windowRequiredExtensions) {
    uint32_t numExtensions = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &numExtensions, nullptr);
    availableInstanceExtensions.resize(numExtensions);
    vkEnumerateInstanceExtensionProperties(nullptr, &numExtensions, availableInstanceExtensions.data());

    uint32_t numLayers = 0;
    vkEnumerateInstanceLayerProperties(&numLayers, nullptr);
    availableInstanceLayers.resize(numLayers);
    vkEnumerateInstanceLayerProperties(&numLayers, availableInstanceLayers.data());

    MAPLE_INFO("Available Vulkan instance extensions:");
    for (const auto& e : availableInstanceExtensions) {
      MAPLE_INFO("\t{}: {}", e.extensionName, e.specVersion);
    }

    MAPLE_INFO("Available Vulkan instance layers:");
    for (const auto& l : availableInstanceLayers) {
      MAPLE_INFO("\t{}: {}, {}, ({})", l.layerName, l.specVersion, l.implementationVersion, l.description);
    }

    for (const char* layerName : validationLayers) {
      bool found = false;

      for (const auto& l : availableInstanceLayers) {
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
    populateDebugMessengerCreateInfo(debugCreateInfo);

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
    populateDebugMessengerCreateInfo(createInfo);

    if (CreateDebugUtilsMessengerEXT(mVkInstance, &createInfo, nullptr, &mDebugMessenger) != VK_SUCCESS)
      MAPLE_FATAL("Failed to create a Vulkan debug messenger");
  }
};  // namespace maple

Renderer::Renderer() : mPimpl(std::make_unique<Impl>()) {}
Renderer::~Renderer() {}
void Renderer::Init(const uint32_t requiredExtensionsCount, const char* const* requiredExtensions) {
  mPimpl->Init(requiredExtensionsCount, requiredExtensions);
}
void Renderer::Destroy() { mPimpl->Destroy(); }

}  // namespace maple
