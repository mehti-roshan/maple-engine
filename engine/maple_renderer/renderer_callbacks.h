#pragma once
#include <cstdint>
#include <functional>

// creates a VkSurfaceKHR using the provided VkInstance
using SurfaceCreateCallback = std::function<void*(void*)>;
// query the framebuffer size from the window library
using FrameBufferSizeCallback = std::function<void(uint32_t&, uint32_t&)>;