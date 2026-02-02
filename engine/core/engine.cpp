#include <engine/core/engine.h>
#include <engine/logging/log_macros.h>
#include "engine/renderer/renderer.h"

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>

void glfw_err_callback(int error, const char* description) { MAPLE_ERROR("GLFW error {}: {}", error, description); }

namespace maple {

void Engine::Init() {
  logging::Log::init();
  MAPLE_INFO("Initializing...");

  initGLFW();
  uint32_t count = 0;
  const char** exts = glfwGetRequiredInstanceExtensions(&count);
  std::vector<const char*> requiredGLFWExtensions(exts, exts + count);

  mRenderer.Init(
    requiredGLFWExtensions,
    [&](VkInstance instance) {
      VkSurfaceKHR surface = VK_NULL_HANDLE;
      if (glfwCreateWindowSurface(instance, mGLFWContext.mWindow, nullptr, &surface) != VK_SUCCESS) MAPLE_FATAL("Failed to create window surface");
      return surface;
    },
    [&](uint32_t& width, uint32_t& height) { glfwGetFramebufferSize(mGLFWContext.mWindow, (int32_t*)&width, (int32_t*)&height); });
}

void Engine::Run() {
  while (!glfwWindowShouldClose(mGLFWContext.mWindow)) {
    glfwPollEvents();
    mRenderer.DrawFrame();
  }
}

void Engine::initGLFW() {
  glfwSetErrorCallback(glfw_err_callback);
  if (!glfwInit()) MAPLE_FATAL("Failed to initialize glfw");
  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  mGLFWContext.mWindow = glfwCreateWindow(1280, 720, "Maple", nullptr, nullptr);
  if (!mGLFWContext.mWindow) MAPLE_FATAL("Failed to create window");

  glfwSetWindowUserPointer(mGLFWContext.mWindow, this);
  glfwSetFramebufferSizeCallback(mGLFWContext.mWindow, [](GLFWwindow* window, int width, int height) {
    reinterpret_cast<Engine*>(glfwGetWindowUserPointer(window))->mRenderer.SetFrameBufferResized();
  });
}

Engine::~Engine() {
  MAPLE_INFO("Shutting down...");
}

}  // namespace maple