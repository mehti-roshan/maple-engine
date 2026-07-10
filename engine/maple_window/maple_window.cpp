#include "maple_window.h"

#include <engine/maple_logging/log_macros.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

static void glfwErrorCallback(int error, const char* description) { MAPLE_ERROR("GLFW error {}: {}", error, description); }

namespace maple {
static std::vector<MapleWindow*> sWindows;
static std::unordered_map<int32_t, MapleWindow::JoystickState> sJoysticks;

struct MapleWindow::Impl {
  GLFWwindow* window = nullptr;
  bool glfwInitialized = false;
  std::unordered_map<CallbackHndl, FrameBufferSizeCallback> frameBufferSizeCallbacks;
  std::unordered_map<CallbackHndl, KeyCallback> keyCallbacks;
  std::unordered_map<CallbackHndl, MouseButtonCallback> mouseButtonCallbacks;
  std::unordered_map<CallbackHndl, ScrollCallback> scrollCallbacks;
  std::unordered_map<CallbackHndl, CursorPosCallback> cursorPosCallbacks;
  std::unordered_map<CallbackHndl, JoySticksCallback> joysticksCallback;
};

MapleWindow::MapleWindow() = default;
MapleWindow::MapleWindow(MapleWindow&& other) noexcept : mImpl(std::move(other.mImpl)) {
  if (mImpl && mImpl->window) glfwSetWindowUserPointer(mImpl->window, this);
  for (auto& ptr : sWindows) {
    if (ptr == &other) ptr = this;
  }
}
MapleWindow& MapleWindow::operator=(MapleWindow&& other) noexcept {
  if (this != &other) {
    mImpl = std::move(other.mImpl);

    if (mImpl && mImpl->window) glfwSetWindowUserPointer(mImpl->window, this);
    for (auto& ptr : sWindows) {
      if (ptr == &other) ptr = this;
    }
  }
  return *this;
}

MapleWindow::MapleWindow(const CreateInfo& info) {
  mImpl = std::make_unique<Impl>();

  glfwSetErrorCallback(glfwErrorCallback);
  if (!glfwInit()) MAPLE_FATAL("Failed to initialize glfw");
  mImpl->glfwInitialized = true;

  glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

  mImpl->window = glfwCreateWindow(1920, 1080, info.title.c_str(), nullptr, nullptr);
  if (!mImpl->window) MAPLE_FATAL("Failed to create window");

  glfwSetWindowUserPointer(mImpl->window, this);
  sWindows.push_back(this);

  glfwSetJoystickCallback(OnJoyConnectDisconnect);
  glfwSetFramebufferSizeCallback(mImpl->window, OnFrameBufferResize);
  glfwSetKeyCallback(mImpl->window, OnKey);
  glfwSetMouseButtonCallback(mImpl->window, OnMouseButtons);
  glfwSetScrollCallback(mImpl->window, OnMouseScroll);
  glfwSetCursorPosCallback(mImpl->window, OnCursorPos);

  GLFWgamepadstate state;
  for (int32_t id = 0; id <= GLFW_JOYSTICK_LAST; id++) {
    if (glfwGetGamepadState(id, &state)) {
      sJoysticks[id] = std::bit_cast<JoystickState>(state);
    }
  }
}

MapleWindow::~MapleWindow() {
  if (!mImpl) return;

  if (mImpl->window) {
    glfwSetFramebufferSizeCallback(mImpl->window, nullptr);
    glfwSetKeyCallback(mImpl->window, nullptr);
    glfwSetMouseButtonCallback(mImpl->window, nullptr);
    glfwSetScrollCallback(mImpl->window, nullptr);
    glfwSetCursorPosCallback(mImpl->window, nullptr);

    glfwDestroyWindow(mImpl->window);
    mImpl->window = nullptr;
  }

  if (mImpl->glfwInitialized) {
    glfwTerminate();
    mImpl->glfwInitialized = false;
  }

  auto it = std::find(sWindows.begin(), sWindows.end(), this);
  if (it != sWindows.end()) sWindows.erase(it);
}

std::vector<const char*> MapleWindow::RequiredVkInstanceExtensions() const {
  uint32_t count = 0;
  const char** exts = glfwGetRequiredInstanceExtensions(&count);
  return std::vector<const char*>(exts, exts + count);
}

void* MapleWindow::CreateWindowSurface(void* pVkInstance) const {
  VkSurfaceKHR surface = VK_NULL_HANDLE;
  if (glfwCreateWindowSurface(static_cast<VkInstance>(pVkInstance), mImpl->window, nullptr, &surface) != VK_SUCCESS)
    MAPLE_FATAL("Failed to create window surface");
  return static_cast<void*>(surface);
}

void MapleWindow::SetTitle(const std::string& title) const { glfwSetWindowTitle(mImpl->window, title.c_str()); }

bool MapleWindow::ShouldClose() const { return glfwWindowShouldClose(mImpl->window); }
void MapleWindow::SetShouldClose(bool shouldClose) const { glfwSetWindowShouldClose(mImpl->window, shouldClose); }

void MapleWindow::PollEvents() const {
  glfwPollEvents();
  UpdateJoySticks();
}

void MapleWindow::UpdateJoySticks() const {
  std::vector<std::pair<int32_t, JoystickState>> pads;
  pads.reserve(GLFW_JOYSTICK_LAST);

  GLFWgamepadstate state;
  for (auto it : sJoysticks) {
    if (glfwGetGamepadState(it.first, &state)) {
      pads.push_back(std::make_pair(it.first, std::bit_cast<JoystickState>(state)));
    }
  }

  OnJoyStickssUpdate(pads);
}

std::pair<int32_t, int32_t> MapleWindow::GetFrameBufferSize() const {
  int32_t x, y;
  glfwGetFramebufferSize(mImpl->window, &x, &y);
  return std::make_pair(x, y);
}

void MapleWindow::LockCursor() const { glfwSetInputMode(mImpl->window, GLFW_CURSOR, GLFW_CURSOR_DISABLED); }
void MapleWindow::UnlockCursor() const { glfwSetInputMode(mImpl->window, GLFW_CURSOR, GLFW_CURSOR_NORMAL); }
bool MapleWindow::RawMouseMotionSupported() const { return glfwRawMouseMotionSupported(); }
void MapleWindow::SetRawMouseMotion(bool value) const { glfwSetInputMode(mImpl->window, GLFW_RAW_MOUSE_MOTION, value); }

template <typename T>
MapleWindow::CallbackHndl findNextFreeCallbackSlot() {
  static MapleWindow::CallbackHndl hndl = 0;
  return hndl++;
}

MapleWindow::CallbackHndl MapleWindow::AddFramebufferSizeCallback(FrameBufferSizeCallback callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  mImpl->frameBufferSizeCallbacks[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveFramebufferSizeCallback(MapleWindow::CallbackHndl hndl) {
  auto it = mImpl->frameBufferSizeCallbacks.find(hndl);
  if (it == mImpl->frameBufferSizeCallbacks.end()) return;
  mImpl->frameBufferSizeCallbacks.erase(it);
}

MapleWindow::CallbackHndl MapleWindow::AddKeyCallback(const KeyCallback& callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  mImpl->keyCallbacks[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveKeyCallback(MapleWindow::CallbackHndl hndl) {
  auto it = mImpl->keyCallbacks.find(hndl);
  if (it == mImpl->keyCallbacks.end()) return;
  mImpl->keyCallbacks.erase(it);
}

MapleWindow::CallbackHndl MapleWindow::AddMouseButtonCallback(MouseButtonCallback callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  mImpl->mouseButtonCallbacks[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveMouseButtonCallback(MapleWindow::CallbackHndl hndl) {
  auto it = mImpl->mouseButtonCallbacks.find(hndl);
  if (it == mImpl->mouseButtonCallbacks.end()) return;
  mImpl->mouseButtonCallbacks.erase(it);
}

MapleWindow::CallbackHndl MapleWindow::AddScrollCallback(ScrollCallback callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  mImpl->scrollCallbacks[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveScrollCallback(MapleWindow::CallbackHndl hndl) {
  auto it = mImpl->scrollCallbacks.find(hndl);
  if (it == mImpl->scrollCallbacks.end()) return;
  mImpl->scrollCallbacks.erase(it);
}

MapleWindow::CallbackHndl MapleWindow::AddJoySticksCallback(JoySticksCallback callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  mImpl->joysticksCallback[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveJoySticksCallback(MapleWindow::CallbackHndl hndl) {
  auto it = mImpl->joysticksCallback.find(hndl);
  if (it == mImpl->joysticksCallback.end()) return;
  mImpl->joysticksCallback.erase(it);
}

MapleWindow::CallbackHndl MapleWindow::AddCursorPosCallback(CursorPosCallback callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  mImpl->cursorPosCallbacks[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveCursorPosCallback(MapleWindow::CallbackHndl hndl) {
  auto it = mImpl->cursorPosCallbacks.find(hndl);
  if (it == mImpl->cursorPosCallbacks.end()) return;
  mImpl->cursorPosCallbacks.erase(it);
}

void MapleWindow::OnFrameBufferResize(GLFWwindow* window, int x, int y) {
  auto pWindow = static_cast<MapleWindow*>(glfwGetWindowUserPointer(window));
  for (auto& f : pWindow->mImpl->frameBufferSizeCallbacks) {
    if (f.second) f.second(x, y);
  }
}
void MapleWindow::OnJoyConnectDisconnect(int jid, int event) {
  if (event == GLFW_CONNECTED) {
    sJoysticks[jid] = {};
  } else if (event == GLFW_DISCONNECTED) {
    auto it = sJoysticks.find(jid);
    if (it != sJoysticks.end()) sJoysticks.erase(it);
  } else {
    // unreachable
  }
}
void MapleWindow::OnKey(GLFWwindow* window, int key, int scancode, int action, int mods) {
  auto pWindow = static_cast<MapleWindow*>(glfwGetWindowUserPointer(window));
  for (auto& f : pWindow->mImpl->keyCallbacks) {
    if (f.second) f.second(key, scancode, action, mods);
  }
}
void MapleWindow::OnMouseButtons(GLFWwindow* window, int button, int action, int mods) {
  auto pWindow = static_cast<MapleWindow*>(glfwGetWindowUserPointer(window));
  for (auto& f : pWindow->mImpl->mouseButtonCallbacks) {
    if (f.second) f.second(button, action, mods);
  }
}
void MapleWindow::OnMouseScroll(GLFWwindow* window, double xoffset, double yoffset) {
  auto pWindow = static_cast<MapleWindow*>(glfwGetWindowUserPointer(window));
  for (auto& f : pWindow->mImpl->scrollCallbacks) {
    if (f.second) f.second(xoffset, yoffset);
  }
}
void MapleWindow::OnCursorPos(GLFWwindow* window, double xpos, double ypos) {
  auto pWindow = static_cast<MapleWindow*>(glfwGetWindowUserPointer(window));
  for (auto& f : pWindow->mImpl->cursorPosCallbacks) {
    if (f.second) f.second(xpos, ypos);
  }
}

void MapleWindow::OnJoyStickssUpdate(const std::vector<std::pair<int32_t, JoystickState>>& pads) const {
  for (auto& f : mImpl->joysticksCallback) {
    if (f.second) f.second(pads);
  }
}

}  // namespace maple