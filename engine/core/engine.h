#pragma once
#include <engine/renderer/renderer.h>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

struct GLFWContext {
  GLFWwindow* mWindow = nullptr;

  ~GLFWContext() {
    if (mWindow) {
      glfwDestroyWindow(mWindow);
      mWindow = nullptr;
    }
    glfwTerminate();
  }
};

namespace maple {
class Engine {
 public:
  ~Engine();

  void Init();
  void Run();

 private:
 GLFWContext mGLFWContext;
 Renderer mRenderer;

  void initGLFW();
};
}  // namespace maple