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
    uint32_t count = 0;
    const char** exts = glfwGetRequiredInstanceExtensions(&count);
    std::vector<const char*> requiredGlfwExtensions(exts, exts + count);
    mRenderer.Init(
        requiredGlfwExtensions,
        [&](VkInstance instance) {
          VkSurfaceKHR surface = VK_NULL_HANDLE;
          if (glfwCreateWindowSurface(instance, mWindow, nullptr, &surface) != VK_SUCCESS) MAPLE_FATAL("Failed to create window surface");
          return surface;
        },
        [&](uint32_t& width, uint32_t& height) { glfwGetFramebufferSize(mWindow, (int32_t*)&width, (int32_t*)&height); });
  }

  void Run() {
    uint32_t frameTimeSize = 1000;
    std::vector<double> frameTimes;
    frameTimes.reserve(frameTimeSize);
    while (!glfwWindowShouldClose(mWindow)) {
      glfwPollEvents();
      auto start = glfwGetTime();
      mRenderer.DrawFrame();
      auto dt = glfwGetTime() - start;
      frameTimes.push_back(dt);
      if (frameTimes.size() == frameTimeSize) {
        double avgDt = 0.0f;
        for (auto t : frameTimes) avgDt += t;
        avgDt /= (double)frameTimeSize;
        MAPLE_INFO("Average frame time over last {} frames: {:.2f} ms ({:.2f} FPS)", frameTimeSize, avgDt * 1000.0, 1.0 / avgDt);
        frameTimes.clear();
      }
    }
  }

  void initGLFW() {
    glfwSetErrorCallback(glfw_err_callback);
    if (!glfwInit()) MAPLE_FATAL("Failed to initialize glfw");
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    mWindow = glfwCreateWindow(1280, 720, "Maple", nullptr, nullptr);
    if (!mWindow) MAPLE_FATAL("Failed to create window");

    glfwSetWindowUserPointer(mWindow, this);
    glfwSetFramebufferSizeCallback(mWindow, [](GLFWwindow* window, int width, int height) {
      reinterpret_cast<Engine::Impl*>(glfwGetWindowUserPointer(window))->mRenderer.SetFrameBufferResized();
    });
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