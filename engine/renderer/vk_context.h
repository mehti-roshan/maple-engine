#pragma once

#include "VkBootstrap.h"
#include <engine/logging/log_macros.h>
#define VULKAN_HPP_NO_STRUCT_CONSTRUCTORS
#include <vulkan/vulkan_raii.hpp>

class VulkanContext {
 public:
  void Init() {
    vkb::InstanceBuilder instance_builder;
    auto instance_ret = instance_builder.set_app_name("Maple")
                          .set_engine_name("Maple")
                          .require_api_version(1, 4, 0)
                          .build();

    if (!instance_ret) MAPLE_FATAL("Failed to create Vulkan instance. Error: {}", instance_ret.error().message());

    VkInstance instance = instance_ret.value();
  }

 private:
};