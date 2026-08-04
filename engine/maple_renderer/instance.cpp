#include "instance.h"

#include <engine/maple_logging/log_macros.h>

#include <vector>

#include "chkvk_macro.h"

namespace maple {

static const char* severityToStr(VkDebugUtilsMessageSeverityFlagBitsEXT severity) {
  switch (severity) {
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT:
      return "VERBOSE";
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT:
      return "INFO";
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT:
      return "WARNING";
    case VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT:
      return "ERROR";
    default:
      return "UNKNOWN";
  }
}

static const char* typeToStr(VkDebugUtilsMessageTypeFlagsEXT type) {
  switch (type) {
    case VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT:
      return "GENERAL";
    case VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT:
      return "VALIDATION";
    case VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT:
      return "PERFORMANCE";
    case VK_DEBUG_UTILS_MESSAGE_TYPE_DEVICE_ADDRESS_BINDING_BIT_EXT:
      return "DEVICE_ADDRESS_BINDING";
    default:
      return "UNKNOWN";
  }
}

static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                    VkDebugUtilsMessageTypeFlagsEXT type,
                                                    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                    void* pUserData) {
  MAPLE_DEBUG("[VK Debug] [{} {}] {}", severityToStr(severity), typeToStr(type), pCallbackData->pMessage);
  return VK_FALSE;
}

VkDebugUtilsMessengerCreateInfoEXT CreateDebugMessengerCI() {
  VkDebugUtilsMessageSeverityFlagsEXT severityFlags(VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT |
                                                    VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT);
  VkDebugUtilsMessageTypeFlagsEXT messageTypeFlags(VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT |
                                                   VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT);
  VkDebugUtilsMessengerCreateInfoEXT debugCI{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
  debugCI.messageSeverity = severityFlags;
  debugCI.messageType = messageTypeFlags;
  debugCI.pfnUserCallback = &debugCallback;
  return debugCI;
}

static constexpr const char* ValidationLayerName = "VK_LAYER_KHRONOS_validation";

static bool HasValidationLayer() {
  uint32_t count = 0;
  vkEnumerateInstanceLayerProperties(&count, nullptr);

  std::vector<VkLayerProperties> layers(count);
  vkEnumerateInstanceLayerProperties(&count, layers.data());

  for (const auto& layer : layers) {
    if (strcmp(layer.layerName, ValidationLayerName) == 0) return true;
  }

  return false;
}

bool Instance::Init(const std::vector<const char*>& requiredExtensions,
                    SurfaceCreateCallback surfaceFunc,
                    FrameBufferSizeCallback frameBufferSizeFunc,
                    bool debug,
                    std::string& err) {
  std::vector<const char*> instanceExtensions = requiredExtensions;
  std::vector<const char*> layers;
  VkDebugUtilsMessengerCreateInfoEXT debugCI{};
  if (debug) {
    if (!HasValidationLayer()) {
      err = "VK_LAYER_KHRONOS_validation not available";
      return false;
    }
    instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    layers.push_back(ValidationLayerName);
    debugCI = CreateDebugMessengerCI();
  }

  VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
  appInfo.pEngineName = "Maple Engine";
  appInfo.apiVersion = VK_API_VERSION_1_3;
  appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
  appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);

  VkInstanceCreateInfo instanceCI{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
  instanceCI.pApplicationInfo = &appInfo;
  instanceCI.enabledExtensionCount = static_cast<uint32_t>(instanceExtensions.size());
  instanceCI.ppEnabledExtensionNames = instanceExtensions.data();
  instanceCI.enabledLayerCount = static_cast<uint32_t>(layers.size());
  instanceCI.ppEnabledLayerNames = layers.data();
  if (debug) {
    instanceCI.pNext = &debugCI;
  }

  chkvk(vkCreateInstance(&instanceCI, nullptr, &mInstance), err, "failed to create vk instance");

  if (debug) {
    chkvk(vkCreateDebugUtilsMessengerEXT(mInstance, &debugCI, nullptr, &mDebugMessenger), err, "failed to create debug messenger");
  }

  mSurface = (VkSurfaceKHR)surfaceFunc(mInstance);
  if (!mSurface) {
    err = "failed to create VkSurfaceKHR";
    return false;
  }

  if (!mDevice.Init(mInstance, mSurface, frameBufferSizeFunc, err)) return false;

  return true;
}

void Instance::Destroy() {
  mDevice.Destroy();
  if (mDebugMessenger) vkDestroyDebugUtilsMessengerEXT(mInstance, mDebugMessenger, nullptr);
  if (mSurface) vkDestroySurfaceKHR(mInstance, mSurface, nullptr);
  if (mInstance) vkDestroyInstance(mInstance, nullptr);
}
}  // namespace maple