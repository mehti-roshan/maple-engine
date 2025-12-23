#pragma once
#include <memory>

struct GLFWwindow;
struct VkInstance;

namespace maple {
class Engine {
 public:
  Engine();
  ~Engine();

  void Init();
  void Run();

 private:
  struct Impl;
  std::unique_ptr<Impl> mPimpl;
};
}  // namespace maple