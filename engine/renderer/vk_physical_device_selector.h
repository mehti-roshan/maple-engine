#pragma once

#include <engine/logging/log_macros.h>

#include <cstdint>
#include <cstring>
#include <map>
#include <vector>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

namespace maple {
struct VulkanPhysicalDeviceSelector {
  struct QueueFamilyIndices {
    uint32_t graphics, present, compute, transfer;
    bool hasGraphics() const { return graphics != VK_QUEUE_FAMILY_IGNORED; }
    bool hasPresent() const { return present != VK_QUEUE_FAMILY_IGNORED; }
    bool hasCompute() const { return compute != VK_QUEUE_FAMILY_IGNORED; }
    bool hasTransfer() const { return transfer != VK_QUEUE_FAMILY_IGNORED; }
    bool complete() const { return hasGraphics() && hasPresent() && hasCompute() && hasTransfer(); }
  };

  struct DeviceSelectionResult {
    vk::raii::PhysicalDevice& device;
    QueueFamilyIndices queueFamilyIndices;
  };

  static vk::raii::PhysicalDevice& PickBestDevice(std::vector<vk::raii::PhysicalDevice>& devices, const vk::raii::SurfaceKHR& surface) {
    std::map<float, vk::raii::PhysicalDevice&> scoredDevices;

    for (auto& device : devices) {
      if (!isSuitable(device, surface)) continue;

      auto qIndices = getQueueFamilyIndices(device, surface);
      float score = calculateScore(device, qIndices);
      scoredDevices.insert({score, device});
    }

    if (scoredDevices.empty()) MAPLE_FATAL("Failed to find suitable GPU");

    return scoredDevices.rbegin()->second;
  }

 private:
  static bool isSuitable(const vk::raii::PhysicalDevice& device, const vk::raii::SurfaceKHR& surface) {
    auto properties = device.getProperties();
    if (properties.apiVersion < VK_API_VERSION_1_4) return false;

    auto features = device.getFeatures2<vk::PhysicalDeviceFeatures2,
                                            vk::PhysicalDeviceVulkan11Features,
                                            vk::PhysicalDeviceVulkan13Features,
                                            vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
    auto deviceFeatures = features.get<vk::PhysicalDeviceFeatures2>();
    auto vk11Features = features.get<vk::PhysicalDeviceVulkan11Features>();
    auto vk13Features = features.get<vk::PhysicalDeviceVulkan13Features>();
    auto extDynStateFeatures = features.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();

    if (!deviceFeatures.features.samplerAnisotropy || !vk11Features.shaderDrawParameters || !vk13Features.dynamicRendering ||
        !vk13Features.synchronization2 || !extDynStateFeatures.extendedDynamicState)
      return false;

    std::vector<const char*> requiredDeviceExtensions = {vk::KHRSwapchainExtensionName}; // TODO: move out

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

  static QueueFamilyIndices getQueueFamilyIndices(const vk::raii::PhysicalDevice& device, const vk::raii::SurfaceKHR& surface) {
    QueueFamilyIndices indices = {VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED};

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
};
}  // namespace maple