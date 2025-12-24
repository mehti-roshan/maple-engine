#include <engine/core/engine.h>
#include <engine/logging/log_macros.h>
#include <engine/renderer/renderer.h>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/mat4x4.hpp>
#include <glm/vec4.hpp>
#include <iostream>

void glfw_err_callback(int error, const char* description) { MAPLE_ERROR("GLFW error {}: {}", error, description); }

namespace maple {

struct Engine::Impl {
  GLFWwindow* mWindow = nullptr;
  Renderer mRenderer;

  void Init() {
    logging::Log::init();
    MAPLE_INFO("Initializing...");

    initGLFW();
    uint32_t glfwRequiredExtensionsCount = 0;
    const char** glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwRequiredExtensionsCount);
    mRenderer.Init(glfwRequiredExtensionsCount, glfwExtensions);

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(mRenderer.GetInstance(), mWindow, nullptr, &surface) != VK_SUCCESS)
      MAPLE_FATAL("Failed to create window surface");
    
    mRenderer.SetSurface(surface);
  }

  void Run() {
    int i = 1000000;
    while (i != 0) {
      glfwPollEvents();
      i--;
    }
  }

  void initGLFW() {
    glfwSetErrorCallback(glfw_err_callback);
    if (!glfwInit()) MAPLE_FATAL("Failed to initialize glfw");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    // to simplify vulkan logic disable window resizing for now
    // TODO: handle window resizing
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    mWindow = glfwCreateWindow(1280, 720, "Maple", nullptr, nullptr);
    if (!mWindow) MAPLE_FATAL("Failed to create window");
  }

  ~Impl() {
    MAPLE_INFO("Shutting down...");

    mRenderer.Destroy();

    glfwDestroyWindow(mWindow);
    glfwTerminate();
  }
};

Engine::Engine() : mPimpl(std::make_unique<Impl>()) {}
Engine::~Engine() {}

void Engine::Init() { mPimpl->Init(); }
void Engine::Run() { mPimpl->Run(); }

}  // namespace maple