#pragma once
#include <memory>

namespace maple {

class Renderer {
 public:
  Renderer();
  ~Renderer();

  void Init(const uint32_t requiredExtensionsCount, const char* const* requiredExtensions);
  void Destroy();

 private:
  struct Impl;
  std::unique_ptr<Impl> mPimpl;
};
}  // namespace maple