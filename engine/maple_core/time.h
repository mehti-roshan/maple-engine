#pragma once

#include <memory>
namespace maple {
class Time {
 public:
  Time();
  ~Time();

  void Initialize();

  void BeginFrame();

  float DeltaTime() const;
  float TimeSinceStart() const;

  // uint64_t Epoch();  // time since epoch

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}  // namespace maple