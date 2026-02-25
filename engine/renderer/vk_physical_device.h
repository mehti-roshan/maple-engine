#pragma once

#include <engine/logging/log_macros.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#include "vk_device_features.h"
#include "vk_queue_family_indices.h"

/**
 * @class VulkanPhysicalDevice
 * @brief Manages physical GPU device selection and queue family configuration for Vulkan rendering.
 *
 * This class encapsulates the logic for selecting the most suitable physical GPU device from
 * available devices and configuring its queue families for graphics, presentation, compute,
 * and transfer operations.
 *
 * @details
 * **Device Selection Strategy:**
 * - Evaluates all available physical devices based on suitability criteria
 * - Ranks suitable devices by a scoring system and selects the highest-scoring device
 * - Ensures the selected device supports all required features and extensions
 *
 * **Suitability Criteria:**
 * - Minimum Vulkan API version 1.4
 * - Required features: samplerAnisotropy, shaderDrawParameters, dynamicRendering,
 *   synchronization2, and extendedDynamicState
 * - Support for VK_KHR_swapchain extension
 * - Complete queue family configuration (graphics, present, compute, transfer)
 *
 * **Scoring System:**
 * Devices are ranked by the following factors (in order of influence):
 * - Discrete GPU advantage: +1000.0 points
 * - VRAM size: score += VRAM in MB
 * - Maximum 2D image dimension supported
 * - Separate graphics and present queues: +500.0 points
 *
 * **Queue Family Assignment Logic:**
 * - Prefers dedicated queue families for each operation type
 * - Graphics queue: First available family with graphics capability
 * - Present queue: First family supporting surface presentation
 * - Compute queue: Prioritizes dedicated compute families without graphics;
 *   falls back to graphics family if dedicated compute is unavailable
 * - Transfer queue: Prefers dedicated transfer families; falls back to compute,
 *   then graphics family if necessary
 *
 * @member device The selected Vulkan physical device
 * @member queueFamilyIndices Queue family indices for graphics, present, compute, and transfer operations
 *
 * @throws Logs MAPLE_FATAL if no suitable GPU device is found
 */
namespace maple {
struct VulkanPhysicalDevice {
  vk::raii::PhysicalDevice device;
  QueueFamilyIndices queueFamilyIndices;

  struct CreateInfo {
    const vk::raii::SurfaceKHR& surface;
    const std::vector<vk::raii::PhysicalDevice>& availableDevices;
    const std::vector<const char*>& requiredDeviceExtensions;
    DeviceFeatureMask requiredFeatureMask{};
  };

  // Default constructor
  VulkanPhysicalDevice() : device(nullptr), queueFamilyIndices({}), mSurface(nullptr) {}

  VulkanPhysicalDevice(const CreateInfo& info) : device(nullptr), mSurface(&info.surface) {
    std::map<float, uint32_t> scoredDevices;

    for (uint32_t i = 0; i < info.availableDevices.size(); ++i) {
      auto& device = info.availableDevices[i];
      if (!isSuitable(device, info.surface, info.requiredDeviceExtensions, info.requiredFeatureMask)) continue;

      auto qIndices = getQueueFamilyIndices(device, info.surface);
      float score = calculateScore(device, qIndices);
      scoredDevices.insert({score, i});
    }

    if (scoredDevices.empty()) MAPLE_FATAL("Failed to find suitable GPU");

    auto idx = scoredDevices.rbegin()->second;

    device = info.availableDevices[idx];
    queueFamilyIndices = getQueueFamilyIndices(device, *mSurface);
  }

  vk::SurfaceCapabilitiesKHR SurfaceCapabilities() const { return device.getSurfaceCapabilitiesKHR(*mSurface); }
  std::vector<vk::SurfaceFormatKHR> SurfaceFormats() { return device.getSurfaceFormatsKHR(*mSurface); }

 private:
  const vk::raii::SurfaceKHR* mSurface;

  static QueueFamilyIndices getQueueFamilyIndices(const vk::raii::PhysicalDevice& device, const vk::raii::SurfaceKHR& surface) {
    QueueFamilyIndices indices{};

    auto qFamilyProps = device.getQueueFamilyProperties();
    auto graphicsBit = vk::QueueFlagBits::eGraphics;
    auto computeBit = vk::QueueFlagBits::eCompute;
    auto transferBit = vk::QueueFlagBits::eTransfer;

    for (uint32_t i = 0; i < qFamilyProps.size(); ++i) {
      auto& qfp = qFamilyProps[i];
      bool graphics = static_cast<bool>(qfp.queueFlags & graphicsBit);
      indices.graphics = graphics && !indices.hasGraphics() ? i : indices.graphics;

      bool present = device.getSurfaceSupportKHR(i, surface);
      indices.present = present && !indices.hasPresent() ? i : indices.present;

      // Ideally select a dedicated compute queue that doesn't have graphics
      bool compute = static_cast<bool>((qfp.queueFlags & computeBit) && !(qfp.queueFlags & graphicsBit));
      indices.compute = compute && !indices.hasCompute() ? i : indices.compute;

      // Ideally select a dedicated transfer queue that doesn't have graphics or compute
      bool transfer = static_cast<bool>((qfp.queueFlags & transferBit) && !(qfp.queueFlags & (graphicsBit | computeBit)));
      indices.transfer = transfer && !indices.hasTransfer() ? i : indices.transfer;

      if (indices.complete()) break;
    }

    // If didn't find a dedicated compute family, but graphics family also has compute, fallback to graphics family for compute
    if (!indices.hasCompute() && indices.hasGraphics() && (qFamilyProps[indices.graphics].queueFlags & computeBit))
      indices.compute = indices.graphics;

    // If didn't find a dedicated transfer family, fallback to compute, then graphics
    if (!indices.hasTransfer() && indices.hasCompute()) indices.transfer = indices.compute;
    if (!indices.hasTransfer() && indices.hasGraphics()) indices.transfer = indices.graphics;

    return indices;
  }

  static bool isSuitable(const vk::raii::PhysicalDevice& device,
                         const vk::raii::SurfaceKHR& surface,
                         const std::vector<const char*>& requiredDeviceExtensions,
                         DeviceFeatureMask featureMask) {
    auto properties = device.getProperties();
    if (properties.apiVersion < VK_API_VERSION_1_4) return false;

    DeviceFeatures features;

    features.chain = device.getFeatures2<vk::PhysicalDeviceFeatures2,
                                         vk::PhysicalDeviceVulkan11Features,
                                         vk::PhysicalDeviceVulkan13Features,
                                         vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

    if (!features.supports(featureMask)) return false;

    for (const auto& ext : requiredDeviceExtensions)
      if (!deviceSupportsExtension(device, ext)) return false;

    auto qIndices = getQueueFamilyIndices(device, surface);
    if (!qIndices.complete()) return false;

    return true;
  }

  static float calculateScore(const vk::raii::PhysicalDevice& device, const QueueFamilyIndices& queueFamilyIndices) {
    float score = 0.0f;

    auto deviceProperties = device.getProperties2();
    deviceProperties.properties.deviceType == vk::PhysicalDeviceType::eDiscreteGpu ? score += 1000.0f : score += 0.0f;

    auto devMemProps = device.getMemoryProperties2();
    uint32_t vramBytes = 0;
    for (auto const& memHeap : devMemProps.memoryProperties.memoryHeaps)
      if (memHeap.flags & vk::MemoryHeapFlagBits::eDeviceLocal) vramBytes += memHeap.size;

    score += (float)vramBytes / 1024 / 1024;  // in MB

    score += static_cast<float>(deviceProperties.properties.limits.maxImageDimension2D);

    if (queueFamilyIndices.graphics != queueFamilyIndices.present) score += 500.0f;  // TODO: ???

    return score;
  }

  static bool deviceSupportsExtension(const vk::raii::PhysicalDevice& device, const char* extensionName) {
    auto deviceExtensions = device.enumerateDeviceExtensionProperties();
    for (auto const& devExt : deviceExtensions)
      if (strcmp(devExt.extensionName, extensionName) == 0) return true;

    return false;
  }
};
}  // namespace maple