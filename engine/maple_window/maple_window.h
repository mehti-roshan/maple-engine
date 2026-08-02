#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "engine/maple_window/enums.h"

union SDL_Event;

namespace maple {
class Window {
 public:
  struct GamePadState {
    std::array<bool, static_cast<size_t>(GamePadButton::Count)> buttons;
    std::array<float, static_cast<size_t>(GamePadAxis::Count)> axes;
  };

  using CallbackHndl = uint32_t;
  using FrameBufferSizeCallback = std::function<void(int x, int y)>;
  using KeyCallback = std::function<void(Key key, bool down)>;
  using MouseButtonCallback = std::function<void(MouseButton button, bool down)>;
  using ScrollCallback = std::function<void(double xOffset, double yOffset)>;
  using CursorPosCallback = std::function<void(double xPos, double yPos)>;
  using GamePadsCallback = std::function<void(const std::vector<std::pair<int32_t, GamePadState>>&)>;

  enum RendererAPI { OpenGL, Vulkan };
  struct CreateInfo {
    const std::string& title;
    RendererAPI rendererAPI;
  };

  Window();
  ~Window();
  Window(Window&&) noexcept;
  Window& operator=(Window&&) noexcept;

  bool Init(const CreateInfo& info, std::string& err);
  void Destroy();

  void SetTitle(const std::string& title) const;
  bool ShouldClose() const;
  void SetShouldClose(bool shouldClose);
  void PollEvents() const;

  bool GetFrameBufferSize(int32_t& x, int32_t& y, std::string& err) const;

  void LockCursor() const;
  void UnlockCursor() const;
  std::string GetJoyStickName(int32_t jid) const;

  /// OpenGL functions
  using GLProc = void (*)();
  bool SetGlAttrib(int attrib, int v, std::string& err);
  bool CreateGlContext(void* ctx, std::string& err);
  static GLProc GetGlProcAddress(const char* name);
  bool DestroyGlContext(void* ctx, std::string& err);
  bool GlPresent(std::string& err);

  /// Vulkan functions
  std::vector<const char*> RequiredVkInstanceExtensions() const;
  void* CreateWindowSurface(void* pVkInstance) const;  // returns VkSurfaceKHR*

  CallbackHndl AddFramebufferSizeCallback(FrameBufferSizeCallback callback);
  void RemoveFramebufferSizeCallback(CallbackHndl hndl);
  CallbackHndl AddKeyCallback(const KeyCallback& callback);
  void RemoveKeyCallback(CallbackHndl hndl);
  CallbackHndl AddMouseButtonCallback(MouseButtonCallback callback);
  void RemoveMouseButtonCallback(CallbackHndl hndl);
  CallbackHndl AddScrollCallback(ScrollCallback callback);
  void RemoveScrollCallback(CallbackHndl hndl);
  CallbackHndl AddCursorPosCallback(CursorPosCallback callback);
  void RemoveCursorPosCallback(CallbackHndl hndl);
  CallbackHndl AddGamePadsCallback(GamePadsCallback callback);
  void RemoveGamePadsCallback(CallbackHndl hndl);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;

  void DispatchEvent(const SDL_Event& event) const;
  void UpdateJoySticks() const;
};
}  // namespace maple