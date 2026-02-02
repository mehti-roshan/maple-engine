#include <engine/file/file.h>
#include <engine/logging/log_macros.h>
#include <engine/renderer/renderer.h>

#include <algorithm>
#include <array>
#include <chrono>
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
                                                      const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData, void*) {
  MAPLE_DEBUG("Vulkan validation layer: type {} msg: {}", to_string(type), pCallbackData->pMessage);
  return vk::False;
}

namespace maple {
void Renderer::Init(const std::vector<const char*>& glfwExtensions, SurfaceCreateCallback surfaceCallback,
                    FrameBufferSizeCallback fbCallback) {
  mFrameBufferSizeCallback = fbCallback;
  createInstance(glfwExtensions);
  setupDebugMessenger();
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
        return std::ranges::none_of(
            layerProperties, [requiredLayer](auto const& layerProperty) { return strcmp(layerProperty.layerName, requiredLayer) == 0; });
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
  std::vector<vk::raii::PhysicalDevice> devices = mInstance.enumeratePhysicalDevices();
  const auto devIter = std::ranges::find_if(devices, [&](auto const& device) {
    auto queueFamilies = device.getQueueFamilyProperties();
    bool isSuitable = device.getProperties().apiVersion >= VK_API_VERSION_1_3;

    const auto qfpIter = std::ranges::find_if(queueFamilies, [](vk::QueueFamilyProperties const& qfp) {
      return (qfp.queueFlags & vk::QueueFlagBits::eGraphics) != static_cast<vk::QueueFlags>(0);
    });
    isSuitable = isSuitable && (qfpIter != queueFamilies.end());
    auto extensions = device.enumerateDeviceExtensionProperties();
    bool found = true;
    for (auto const& extension : deviceExtensions) {
      auto extensionIter =
          std::ranges::find_if(extensions, [extension](auto const& ext) { return strcmp(ext.extensionName, extension) == 0; });
      found = found && extensionIter != extensions.end();
    }
    isSuitable = isSuitable && found;
    if (isSuitable) mPhysicalDevice = device;

    return isSuitable;
  });
  if (devIter == devices.end()) MAPLE_FATAL("Failed to find suitable GPU");
}

void Renderer::createLogicalDevice() {
  vk::StructureChain<vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan13Features, vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
      featureChain = {
          {},                             // vk::PhysicalDeviceFeatures2 (empty for now)
          {.dynamicRendering = true},     // Enable dynamic rendering from Vulkan 1.3
          {.extendedDynamicState = true}  // Enable extended dynamic state from the extension
      };

  std::vector<vk::QueueFamilyProperties> queueFamilyProperties = mPhysicalDevice.getQueueFamilyProperties();
  uint32_t graphicsIndex = findQueueFamilies(mPhysicalDevice);
  vk::DeviceQueueCreateInfo deviceQueueCreateInfo{.queueFamilyIndex = graphicsIndex};

  vk::DeviceCreateInfo deviceCreateInfo{
      .pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>(),
      .queueCreateInfoCount = 1,
      .pQueueCreateInfos = &deviceQueueCreateInfo,
      .enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size()),
      .ppEnabledExtensionNames = deviceExtensions.data(),
  };

  mDevice = vk::raii::Device(mPhysicalDevice, deviceCreateInfo);
  mGraphicsQueue = vk::raii::Queue(mDevice, graphicsIndex, 0);
}

uint32_t Renderer::findQueueFamilies(vk::raii::PhysicalDevice physicalDevice) {
  // find the index of the first queue family that supports graphics
  std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

  // get the first index into queueFamilyProperties which supports graphics
  auto graphicsQueueFamilyProperty =
      std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(),
                   [](vk::QueueFamilyProperties const& qfp) { return qfp.queueFlags & vk::QueueFlagBits::eGraphics; });

  return static_cast<uint32_t>(std::distance(queueFamilyProperties.begin(), graphicsQueueFamilyProperty));
}
}  // namespace maple
