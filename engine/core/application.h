#pragma once
#include <SDL.h>

namespace maple {
class Application {
 public:
  Application();
  ~Application();

 private:
  SDL_Window* mWindow;
  bool mRunning;
};
}  // namespace maple