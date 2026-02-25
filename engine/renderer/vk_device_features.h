#pragma once

#include <cstdint>

#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

namespace maple {
enum DeviceFeature : uint64_t {
  SamplerAnisotropy = 1ull << 0,
  ShaderDrawParameters = 1ull << 1,
  DynamicRendering = 1ull << 2,
  Synchronization2 = 1ull << 3,
  ExtendedDynamicState = 1ull << 4,
};

using DeviceFeatureMask = uint64_t;

struct DeviceFeatures {
  vk::StructureChain<vk::PhysicalDeviceFeatures2,
                     vk::PhysicalDeviceVulkan11Features,
                     vk::PhysicalDeviceVulkan13Features,
                     vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>
    chain{};

  DeviceFeatures() {}

  bool supports(DeviceFeatureMask mask) const {
    if (mask & (uint64_t)DeviceFeature::SamplerAnisotropy)
      if (!getCore().features.samplerAnisotropy) return false;

    if (mask & (uint64_t)DeviceFeature::ShaderDrawParameters)
      if (!getVk11().shaderDrawParameters) return false;

    if (mask & (uint64_t)DeviceFeature::DynamicRendering)
      if (!getVk13().dynamicRendering) return false;

    if (mask & (uint64_t)DeviceFeature::Synchronization2)
      if (!getVk13().synchronization2) return false;

    if (mask & (uint64_t)DeviceFeature::ExtendedDynamicState)
      if (!getExtDynState().extendedDynamicState) return false;

    return true;
  }

  void enable(DeviceFeatureMask mask) {
    if (mask & (uint64_t)DeviceFeature::SamplerAnisotropy) getCore().features.samplerAnisotropy = VK_TRUE;

    if (mask & (uint64_t)DeviceFeature::ShaderDrawParameters) getVk11().shaderDrawParameters = VK_TRUE;

    if (mask & (uint64_t)DeviceFeature::DynamicRendering) getVk13().dynamicRendering = VK_TRUE;

    if (mask & (uint64_t)DeviceFeature::Synchronization2) getVk13().synchronization2 = VK_TRUE;

    if (mask & (uint64_t)DeviceFeature::ExtendedDynamicState) getExtDynState().extendedDynamicState = VK_TRUE;
  }

  vk::PhysicalDeviceFeatures2& getCore() { return chain.get<vk::PhysicalDeviceFeatures2>(); }

 private:
  const vk::PhysicalDeviceFeatures2& getCore() const { return chain.get<vk::PhysicalDeviceFeatures2>(); }

  vk::PhysicalDeviceVulkan11Features& getVk11() { return chain.get<vk::PhysicalDeviceVulkan11Features>(); }
  const vk::PhysicalDeviceVulkan11Features& getVk11() const { return chain.get<vk::PhysicalDeviceVulkan11Features>(); }

  vk::PhysicalDeviceVulkan13Features& getVk13() { return chain.get<vk::PhysicalDeviceVulkan13Features>(); }
  const vk::PhysicalDeviceVulkan13Features& getVk13() const { return chain.get<vk::PhysicalDeviceVulkan13Features>(); }

  vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT& getExtDynState() { return chain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>(); }
  const vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT& getExtDynState() const {
    return chain.get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>();
  }
};
}  // namespace maple