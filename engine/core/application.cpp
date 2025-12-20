#include <SDL.h>
#include <engine/core/application.h>
#include <engine/logging/log_macros.h>

maple::Application::Application() {
  logging::Log::init();

  MAPLE_INFO("Initializing SDL...");

  if (SDL_Init(SDL_INIT_VIDEO) < 0) MAPLE_FATAL("Failed to initialize SDL: {}", SDL_GetError());

  mWindow = SDL_CreateWindow("Maple Engine", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1280, 720, SDL_WINDOW_SHOWN);
  if (!mWindow) MAPLE_FATAL("Failed to create window: {}", SDL_GetError());

  auto screenSurface = SDL_GetWindowSurface(mWindow);
  SDL_FillRect(screenSurface, nullptr, SDL_MapRGB(screenSurface->format, 0xFF, 0xFF, 0xFF));
  SDL_UpdateWindowSurface(mWindow);

  MAPLE_INFO("SDL initialized successfully.");

  SDL_Event e;
  mRunning = true;

  while (mRunning) {
    SDL_PollEvent(&e);
    if (e.type == SDL_QUIT) {
      mRunning = false;
    }
  }
}

maple::Application::~Application() {
  MAPLE_INFO("Shutting down...");
  SDL_DestroyWindow(mWindow);
  SDL_Quit();
}