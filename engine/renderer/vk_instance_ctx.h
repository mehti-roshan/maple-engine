#pragma once

#include <engine/logging/log_macros.h>

#include <vector>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

namespace maple {

struct VulkanInstanceContext {
  vk::raii::Context mContext;
  vk::raii::Instance mInstance;
  vk::raii::DebugUtilsMessengerEXT mDebugMessenger;

  VulkanInstanceContext() : mInstance(nullptr), mDebugMessenger(nullptr) {}
  VulkanInstanceContext(std::vector<const char*> requiredExtensions, bool debug) : mInstance(nullptr), mDebugMessenger(nullptr) {
    std::vector<const char*> requiredLayers;

    if (debug) {
      requiredExtensions.push_back(vk::EXTDebugUtilsExtensionName);
      const std::vector<char const*> validationLayers = {"VK_LAYER_KHRONOS_validation"};
      requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    // Check required extensions and layers
    auto extProps = mContext.enumerateInstanceExtensionProperties();
    checkMissingExtensions(requiredExtensions, extProps);
    auto layerProperties = mContext.enumerateInstanceLayerProperties();
    checkMissingValidationLayers(requiredLayers, layerProperties);

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

    if (debug) createDebugMessenger();
  }

 private:
  void checkMissingExtensions(const std::vector<const char*>& requiredExtensions, std::vector<vk::ExtensionProperties>& extensionProperties) {
    for (size_t i = 0; i < requiredExtensions.size(); ++i) {
      auto v = requiredExtensions[i];
      bool found = false;
      for (auto const& extProps : extensionProperties) {
        if (strcmp(extProps.extensionName, v) == 0) {
          found = true;
          break;
        }
      }
      if (!found) MAPLE_FATAL("Missing required extension {}", v);
    }
  }

  void checkMissingValidationLayers(const std::vector<const char*>& requiredLayers, std::vector<vk::LayerProperties>& layerProps) {
    for (auto const& requiredLayer : requiredLayers) {
      bool found = false;

      for (auto const& layerProperty : layerProps) {
        if (strcmp(layerProperty.layerName, requiredLayer) == 0) {
          found = true;
          break;
        }
      }
      if (!found) MAPLE_FATAL("Missing required validation layer {}", requiredLayer);
    }
  }

  void createDebugMessenger() {
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

  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
                                                        vk::DebugUtilsMessageTypeFlagsEXT type,
                                                        const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
                                                        void*) {
    MAPLE_DEBUG("Vulkan [{} {}] {}", vk::to_string(severity), vk::to_string(type), pCallbackData->pMessage);
    return vk::False;
  }
};
}  // namespace maple