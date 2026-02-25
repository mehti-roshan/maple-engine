#pragma once

#include <cstdint>
#include <ranges>
#include <set>
#include <vector>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

#include "engine/renderer/vk_device_features.h"
#include "engine/renderer/vk_queue_family_indices.h"
#include "vk_physical_device.h"

namespace maple {
struct VulkanLogicalDevice {
  struct Queues {
    vk::raii::Queue graphics = nullptr;
    vk::raii::Queue present = nullptr;
    vk::raii::Queue transfer = nullptr;
    vk::raii::Queue compute = nullptr;
  };
  vk::raii::Device device;
  Queues queues;

  struct CreateInfo {
    const VulkanPhysicalDevice& physicalDevice;
    const std::vector<const char*> requiredDeviceExtensions;
    DeviceFeatureMask requiredFeatures;
  };

  VulkanLogicalDevice() : device(nullptr) {}

  VulkanLogicalDevice(const CreateInfo& info) : device(nullptr) {
    auto& qIndices = info.physicalDevice.queueFamilyIndices;

    std::set<uint32_t> uniqueFamilies = {qIndices.graphics, qIndices.present, qIndices.transfer, qIndices.compute};
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos(uniqueFamilies.size());

    float priority = 1.0f;
    for (auto [i, family] : std::views::enumerate(uniqueFamilies))
      queueCreateInfos[i] = vk::DeviceQueueCreateInfo{.queueFamilyIndex = family, .queueCount = 1, .pQueuePriorities = &priority};

    DeviceFeatures features;
    features.enable(info.requiredFeatures);

    vk::DeviceCreateInfo deviceCreateInfo{
      .pNext = &features.getCore(),
      .queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size()),
      .pQueueCreateInfos = queueCreateInfos.data(),
      .enabledExtensionCount = static_cast<uint32_t>(info.requiredDeviceExtensions.size()),
      .ppEnabledExtensionNames = info.requiredDeviceExtensions.data(),
    };

    device = vk::raii::Device(info.physicalDevice.device, deviceCreateInfo);

    queues = {
      .graphics = vk::raii::Queue(device, qIndices.graphics, 0),
      .present = vk::raii::Queue(device, qIndices.present, 0),
      .transfer = vk::raii::Queue(device, qIndices.transfer, 0),
      .compute = vk::raii::Queue(device, qIndices.compute, 0),
    };
  }
};
}  // namespace maple