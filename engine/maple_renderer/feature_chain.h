#pragma once

#include <vulkan/vulkan_core.h>

namespace maple {
struct FeatureChain {
  VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
  VkPhysicalDeviceShaderDrawParametersFeatures shaderDraw{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_DRAW_PARAMETERS_FEATURES};
  VkPhysicalDeviceDynamicRenderingFeatures dynamicRendering{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DYNAMIC_RENDERING_FEATURES};
  VkPhysicalDeviceSynchronization2Features synchronization2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SYNCHRONIZATION_2_FEATURES};
  VkPhysicalDeviceBufferDeviceAddressFeatures bufferDeviceAddress{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES};
  VkPhysicalDeviceDescriptorIndexingFeatures descriptorIndexing{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_FEATURES};
  VkPhysicalDeviceScalarBlockLayoutFeatures scalarBlockLayout{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SCALAR_BLOCK_LAYOUT_FEATURES};

  // Keep it non copyable to prevent the pNext pointers from being invalid
  FeatureChain(const FeatureChain&) = delete;
  FeatureChain& operator=(const FeatureChain&) = delete;

  FeatureChain() {
    // set all the pNext pointers to build the chain
    features.pNext = &shaderDraw;
    shaderDraw.pNext = &dynamicRendering;
    dynamicRendering.pNext = &synchronization2;
    synchronization2.pNext = &bufferDeviceAddress;
    bufferDeviceAddress.pNext = &descriptorIndexing;
    descriptorIndexing.pNext = &scalarBlockLayout;
  };

  bool CheckRequired() const {
    if (!features.features.samplerAnisotropy) return false;
    if (!features.features.shaderInt64) return false;
    if (!shaderDraw.shaderDrawParameters) return false;
    if (!dynamicRendering.dynamicRendering) return false;
    if (!synchronization2.synchronization2) return false;
    if (!bufferDeviceAddress.bufferDeviceAddress) return false;
    if (!descriptorIndexing.shaderSampledImageArrayNonUniformIndexing) return false;
    if (!descriptorIndexing.descriptorBindingPartiallyBound) return false;
    if (!descriptorIndexing.runtimeDescriptorArray) return false;
    if (!scalarBlockLayout.scalarBlockLayout) return false;
    return true;
  }

  void EnableRequired() {
    features.features.samplerAnisotropy = VK_TRUE;
    features.features.shaderInt64 = VK_TRUE;
    shaderDraw.shaderDrawParameters = VK_TRUE;
    dynamicRendering.dynamicRendering = VK_TRUE;
    synchronization2.synchronization2 = VK_TRUE;
    bufferDeviceAddress.bufferDeviceAddress = VK_TRUE;
    descriptorIndexing.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    descriptorIndexing.descriptorBindingPartiallyBound = VK_TRUE;
    descriptorIndexing.runtimeDescriptorArray = VK_TRUE;
    scalarBlockLayout.scalarBlockLayout = VK_TRUE;
  }
};
}  // namespace maple