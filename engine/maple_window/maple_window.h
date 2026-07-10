#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

class GLFWwindow;
struct GLFWgamepadstate;

namespace maple {
class MapleWindow {
 public:
  struct CreateInfo {
    const std::string& title;
  };

  MapleWindow();
  MapleWindow(const CreateInfo& info);
  ~MapleWindow();

  MapleWindow(MapleWindow&&) noexcept;
  MapleWindow& operator=(MapleWindow&&) noexcept;

  void SetTitle(const std::string& title) const;
  bool ShouldClose() const;
  void SetShouldClose(bool shouldClose) const;
  void PollEvents() const;

  std::pair<int32_t, int32_t> GetFrameBufferSize() const;  // X and Y

  void LockCursor() const;
  void UnlockCursor() const;
  bool RawMouseMotionSupported() const;
  void SetRawMouseMotion(bool value) const;

  std::vector<const char*> RequiredVkInstanceExtensions() const;
  void* CreateWindowSurface(void* pVkInstance) const;  // returns VkSurfaceKHR*

  std::string GetJoyStickName(int32_t jid) const;

  using CallbackHndl = uint32_t;

  using FrameBufferSizeCallback = std::function<void(int x, int y)>;
  CallbackHndl AddFramebufferSizeCallback(FrameBufferSizeCallback callback);
  void RemoveFramebufferSizeCallback(CallbackHndl hndl);

  using KeyCallback = std::function<void(int key, int scancode, int action, int mods)>;
  CallbackHndl AddKeyCallback(const KeyCallback& callback);
  void RemoveKeyCallback(CallbackHndl hndl);

  using MouseButtonCallback = std::function<void(int button, int action, int mods)>;
  CallbackHndl AddMouseButtonCallback(MouseButtonCallback callback);
  void RemoveMouseButtonCallback(CallbackHndl hndl);

  using ScrollCallback = std::function<void(double xOffset, double yOffset)>;
  CallbackHndl AddScrollCallback(ScrollCallback callback);
  void RemoveScrollCallback(CallbackHndl hndl);

  using CursorPosCallback = std::function<void(double xPos, double yPos)>;
  CallbackHndl AddCursorPosCallback(CursorPosCallback callback);
  void RemoveCursorPosCallback(CallbackHndl hndl);

  struct JoystickState {
    unsigned char buttons[15];
    float axes[6];
  };
  using JoySticksCallback = std::function<void(const std::vector<std::pair<int32_t, JoystickState>>&)>;
  CallbackHndl AddJoySticksCallback(JoySticksCallback callback);
  void RemoveJoySticksCallback(CallbackHndl hndl);

 private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;

  void UpdateJoySticks() const;

  static void OnFrameBufferResize(GLFWwindow* window, int x, int y);
  static void OnJoyConnectDisconnect(int jid, int event);
  static void OnKey(GLFWwindow* window, int key, int scancode, int action, int mods);
  static void OnMouseButtons(GLFWwindow* window, int button, int action, int mods);
  static void OnMouseScroll(GLFWwindow* window, double xoffset, double yoffset);
  static void OnCursorPos(GLFWwindow* window, double xpos, double ypos);
  void OnJoyStickssUpdate(const std::vector<std::pair<int32_t, JoystickState>>& pads) const;
};
}  // namespace maple