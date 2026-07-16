#include "maple_window.h"

#include <SDL3/SDL.h>
#include <SDL3/SDL_events.h>
#include <SDL3/SDL_gamepad.h>
#include <SDL3/SDL_joystick.h>
#include <SDL3/SDL_video.h>
#include <SDL3/SDL_vulkan.h>
#include <engine/maple_logging/log_macros.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

namespace maple {
MouseButton ConvertMouseBtnSDL(uint8_t btn);
Key ConvertKeyBtnSDL(SDL_Keycode btn);

struct MapleWindow::Impl {
  SDL_Window* window = nullptr;
  bool sdlInitialized = false;
  bool shouldClose = false;

  std::unordered_map<SDL_JoystickID, SDL_Gamepad*> gamepads;

  std::unordered_map<CallbackHndl, FrameBufferSizeCallback> frameBufferSizeCallbacks;
  std::unordered_map<CallbackHndl, KeyCallback> keyCallbacks;
  std::unordered_map<CallbackHndl, MouseButtonCallback> mouseButtonCallbacks;
  std::unordered_map<CallbackHndl, ScrollCallback> scrollCallbacks;
  std::unordered_map<CallbackHndl, CursorPosCallback> cursorPosCallbacks;
  std::unordered_map<CallbackHndl, GamePadsCallback> gamePadsCallback;
};

MapleWindow::MapleWindow(MapleWindow&&) noexcept = default;
MapleWindow& MapleWindow::operator=(MapleWindow&&) noexcept = default;

MapleWindow::MapleWindow() = default;
MapleWindow::MapleWindow(const CreateInfo& info) : impl(std::make_unique<Impl>()) {
  if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_GAMEPAD)) MAPLE_FATAL("SDL init failed: {}", SDL_GetError());

  impl->sdlInitialized = true;

  impl->window = SDL_CreateWindow(info.title.c_str(), 1920, 1080, SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE);

  if (!impl->window) MAPLE_FATAL("SDL window creation failed: {}", SDL_GetError());

  int count = 0;
  SDL_JoystickID* joysticks = SDL_GetJoysticks(&count);

  for (int i = 0; i < count; i++) {
    impl->gamepads[joysticks[i]] = {};
  }

  SDL_free(joysticks);
}

MapleWindow::~MapleWindow() {
  if (!impl) return;

  if (impl->window) {
    SDL_DestroyWindow(impl->window);
    impl->window = nullptr;
  }

  if (impl->sdlInitialized) {
    SDL_Quit();
    impl->sdlInitialized = false;
  }
}

std::vector<const char*> MapleWindow::RequiredVkInstanceExtensions() const {
  uint32_t count = 0;

  const char* const* extensions = SDL_Vulkan_GetInstanceExtensions(&count);

  return {extensions, extensions + count};
}

void* MapleWindow::CreateWindowSurface(void* instance) const {
  VkSurfaceKHR surface{};

  if (!SDL_Vulkan_CreateSurface(impl->window, static_cast<VkInstance>(instance), nullptr, &surface)) {
    MAPLE_FATAL("SDL Vulkan surface failed: {}", SDL_GetError());
  }

  return surface;
}

void MapleWindow::SetTitle(const std::string& title) const { SDL_SetWindowTitle(impl->window, title.c_str()); }

bool MapleWindow::ShouldClose() const { return impl->shouldClose; }
void MapleWindow::SetShouldClose(bool shouldClose) { impl->shouldClose = shouldClose; }

void MapleWindow::PollEvents() const {
  SDL_Event event;

  while (SDL_PollEvent(&event)) {
    DispatchEvent(event);
  }

  UpdateJoySticks();
}

void MapleWindow::DispatchEvent(const SDL_Event& e) const {
  static std::optional<std::pair<int32_t, int32_t>> mouseCoords = std::nullopt;
  switch (e.type) {
    case SDL_EVENT_QUIT:
      impl->shouldClose = true;
      break;

    case SDL_EVENT_WINDOW_RESIZED: {
      auto size = GetFrameBufferSize();

      for (auto& [_, cb] : impl->frameBufferSizeCallbacks) {
        cb(size.first, size.second);
      }

      break;
    }

    case SDL_EVENT_KEY_DOWN:
    case SDL_EVENT_KEY_UP: {
      bool down = e.type == SDL_EVENT_KEY_DOWN ? 1 : 0;

      for (auto& [_, cb] : impl->keyCallbacks) {
        cb(ConvertKeyBtnSDL(e.key.key), down);
      }

      break;
    }

    case SDL_EVENT_MOUSE_BUTTON_DOWN:
    case SDL_EVENT_MOUSE_BUTTON_UP: {
      // int action = e.type == SDL_EVENT_MOUSE_BUTTON_DOWN ? 1 : 0;
      bool down = e.type == SDL_EVENT_MOUSE_BUTTON_DOWN ? 1 : 0;

      for (auto& [_, cb] : impl->mouseButtonCallbacks) {
        cb(ConvertMouseBtnSDL(e.button.button), down);
      }

      break;
    }

    case SDL_EVENT_MOUSE_MOTION: {
      if (!mouseCoords.has_value()) {
        mouseCoords = std::make_pair(0, 0);
        break;
      } else {
        mouseCoords->first += e.motion.xrel;
        mouseCoords->second += e.motion.yrel;
      }

      for (auto& [_, cb] : impl->cursorPosCallbacks) {
        cb(mouseCoords->first, mouseCoords->second);
      }

      break;
    }

    case SDL_EVENT_MOUSE_WHEEL: {
      for (auto& [_, cb] : impl->scrollCallbacks) {
        cb(e.wheel.x, e.wheel.y);
      }

      break;
    }

    case SDL_EVENT_GAMEPAD_ADDED: {
      impl->gamepads[e.gdevice.which] = SDL_OpenGamepad(e.gdevice.which);
      break;
    }

    case SDL_EVENT_GAMEPAD_REMOVED: {
      auto it = impl->gamepads.find(e.gdevice.which);
      if (it != impl->gamepads.end()) SDL_CloseGamepad(it->second);
      impl->gamepads.erase(e.gdevice.which);
      break;
    }
  }
}

void MapleWindow::UpdateJoySticks() const {
  std::vector<std::pair<int32_t, GamePadState>> states;
  states.reserve(impl->gamepads.size());

  for (auto& [id, pPad] : impl->gamepads) {
    GamePadState state{};
    state.buttons[static_cast<size_t>(GamePadButton::A)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_SOUTH);
    state.buttons[static_cast<size_t>(GamePadButton::B)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_WEST);
    state.buttons[static_cast<size_t>(GamePadButton::X)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_EAST);
    state.buttons[static_cast<size_t>(GamePadButton::Y)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_NORTH);
    state.buttons[static_cast<size_t>(GamePadButton::LeftBumper)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER);
    state.buttons[static_cast<size_t>(GamePadButton::RightBumper)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER);
    state.buttons[static_cast<size_t>(GamePadButton::Back)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_BACK);
    state.buttons[static_cast<size_t>(GamePadButton::Start)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_START);
    state.buttons[static_cast<size_t>(GamePadButton::Guide)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_GUIDE);
    state.buttons[static_cast<size_t>(GamePadButton::LeftThumb)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_LEFT_STICK);
    state.buttons[static_cast<size_t>(GamePadButton::RightThumb)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_RIGHT_STICK);
    state.buttons[static_cast<size_t>(GamePadButton::DPadUp)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_DPAD_UP);
    state.buttons[static_cast<size_t>(GamePadButton::DPadRight)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_DPAD_RIGHT);
    state.buttons[static_cast<size_t>(GamePadButton::DPadDown)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_DPAD_DOWN);
    state.buttons[static_cast<size_t>(GamePadButton::DPadLeft)] = SDL_GetGamepadButton(pPad, SDL_GAMEPAD_BUTTON_DPAD_LEFT);

    state.axes[static_cast<size_t>(GamePadAxis::LeftX)] = std::max(-1.0f, SDL_GetGamepadAxis(pPad, SDL_GAMEPAD_AXIS_LEFTX) / 32767.0f);
    state.axes[static_cast<size_t>(GamePadAxis::LeftY)] = std::max(-1.0f, SDL_GetGamepadAxis(pPad, SDL_GAMEPAD_AXIS_LEFTY) / 32767.0f);
    state.axes[static_cast<size_t>(GamePadAxis::RightX)] = std::max(-1.0f, SDL_GetGamepadAxis(pPad, SDL_GAMEPAD_AXIS_RIGHTX) / 32767.0f);
    state.axes[static_cast<size_t>(GamePadAxis::RightY)] = std::max(-1.0f, SDL_GetGamepadAxis(pPad, SDL_GAMEPAD_AXIS_RIGHTY) / 32767.0f);
    state.axes[static_cast<size_t>(GamePadAxis::LeftTrigger)] = SDL_GetGamepadAxis(pPad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER) / 32767.0f;
    state.axes[static_cast<size_t>(GamePadAxis::RightTrigger)] = SDL_GetGamepadAxis(pPad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) / 32767.0f;

    states.push_back(std::make_pair(id, state));
  }

  for (auto& [_, cb] : impl->gamePadsCallback) {
    cb(states);
  }
}

std::pair<int32_t, int32_t> MapleWindow::GetFrameBufferSize() const {
  int w, h;

  SDL_GetWindowSizeInPixels(impl->window, &w, &h);

  return {w, h};
}

std::string MapleWindow::GetJoyStickName(int32_t jid) const {
  auto it = impl->gamepads.find(jid);
  if (it == impl->gamepads.end()) {
    MAPLE_WARN("invalid gamepad id '{}'", jid);
    return "";
  }
  return SDL_GetGamepadName(it->second);
}

void MapleWindow::LockCursor() const { SDL_SetWindowRelativeMouseMode(impl->window, true); }
void MapleWindow::UnlockCursor() const { SDL_SetWindowRelativeMouseMode(impl->window, false); }

template <typename T>
MapleWindow::CallbackHndl findNextFreeCallbackSlot() {
  static MapleWindow::CallbackHndl hndl = 0;
  return hndl++;
}

MapleWindow::CallbackHndl MapleWindow::AddFramebufferSizeCallback(FrameBufferSizeCallback callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  impl->frameBufferSizeCallbacks[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveFramebufferSizeCallback(MapleWindow::CallbackHndl hndl) {
  auto it = impl->frameBufferSizeCallbacks.find(hndl);
  if (it == impl->frameBufferSizeCallbacks.end()) return;
  impl->frameBufferSizeCallbacks.erase(it);
}

MapleWindow::CallbackHndl MapleWindow::AddKeyCallback(const KeyCallback& callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  impl->keyCallbacks[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveKeyCallback(MapleWindow::CallbackHndl hndl) {
  auto it = impl->keyCallbacks.find(hndl);
  if (it == impl->keyCallbacks.end()) return;
  impl->keyCallbacks.erase(it);
}

MapleWindow::CallbackHndl MapleWindow::AddMouseButtonCallback(MouseButtonCallback callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  impl->mouseButtonCallbacks[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveMouseButtonCallback(MapleWindow::CallbackHndl hndl) {
  auto it = impl->mouseButtonCallbacks.find(hndl);
  if (it == impl->mouseButtonCallbacks.end()) return;
  impl->mouseButtonCallbacks.erase(it);
}

MapleWindow::CallbackHndl MapleWindow::AddScrollCallback(ScrollCallback callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  impl->scrollCallbacks[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveScrollCallback(MapleWindow::CallbackHndl hndl) {
  auto it = impl->scrollCallbacks.find(hndl);
  if (it == impl->scrollCallbacks.end()) return;
  impl->scrollCallbacks.erase(it);
}

MapleWindow::CallbackHndl MapleWindow::AddGamePadsCallback(GamePadsCallback callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  impl->gamePadsCallback[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveGamePadsCallback(MapleWindow::CallbackHndl hndl) {
  auto it = impl->gamePadsCallback.find(hndl);
  if (it == impl->gamePadsCallback.end()) return;
  impl->gamePadsCallback.erase(it);
}

MapleWindow::CallbackHndl MapleWindow::AddCursorPosCallback(CursorPosCallback callback) {
  auto hndl = findNextFreeCallbackSlot<typeof(callback)>();
  impl->cursorPosCallbacks[hndl] = callback;
  return hndl;
}

void MapleWindow::RemoveCursorPosCallback(MapleWindow::CallbackHndl hndl) {
  auto it = impl->cursorPosCallbacks.find(hndl);
  if (it == impl->cursorPosCallbacks.end()) return;
  impl->cursorPosCallbacks.erase(it);
}

MouseButton ConvertMouseBtnSDL(uint8_t btn) {
  switch (btn) {
    case SDL_BUTTON_LEFT:
      return MouseButton::Left;
    case SDL_BUTTON_MIDDLE:
      return MouseButton::Middle;
    case SDL_BUTTON_RIGHT:
      return MouseButton::Right;
    case SDL_BUTTON_X1:
      return MouseButton::Side1;
    case SDL_BUTTON_X2:
      return MouseButton::Side2;
    default:
      return MouseButton::Unknown;
  }
}

Key ConvertKeyBtnSDL(SDL_Keycode key) {
  switch (key) {
    // Letters
    case SDLK_A:
      return Key::A;
    case SDLK_B:
      return Key::B;
    case SDLK_C:
      return Key::C;
    case SDLK_D:
      return Key::D;
    case SDLK_E:
      return Key::E;
    case SDLK_F:
      return Key::F;
    case SDLK_G:
      return Key::G;
    case SDLK_H:
      return Key::H;
    case SDLK_I:
      return Key::I;
    case SDLK_J:
      return Key::J;
    case SDLK_K:
      return Key::K;
    case SDLK_L:
      return Key::L;
    case SDLK_M:
      return Key::M;
    case SDLK_N:
      return Key::N;
    case SDLK_O:
      return Key::O;
    case SDLK_P:
      return Key::P;
    case SDLK_Q:
      return Key::Q;
    case SDLK_R:
      return Key::R;
    case SDLK_S:
      return Key::S;
    case SDLK_T:
      return Key::T;
    case SDLK_U:
      return Key::U;
    case SDLK_V:
      return Key::V;
    case SDLK_W:
      return Key::W;
    case SDLK_X:
      return Key::X;
    case SDLK_Y:
      return Key::Y;
    case SDLK_Z:
      return Key::Z;

    // Numbers
    case SDLK_0:
      return Key::Num0;
    case SDLK_1:
      return Key::Num1;
    case SDLK_2:
      return Key::Num2;
    case SDLK_3:
      return Key::Num3;
    case SDLK_4:
      return Key::Num4;
    case SDLK_5:
      return Key::Num5;
    case SDLK_6:
      return Key::Num6;
    case SDLK_7:
      return Key::Num7;
    case SDLK_8:
      return Key::Num8;
    case SDLK_9:
      return Key::Num9;

    // Function keys
    case SDLK_F1:
      return Key::F1;
    case SDLK_F2:
      return Key::F2;
    case SDLK_F3:
      return Key::F3;
    case SDLK_F4:
      return Key::F4;
    case SDLK_F5:
      return Key::F5;
    case SDLK_F6:
      return Key::F6;
    case SDLK_F7:
      return Key::F7;
    case SDLK_F8:
      return Key::F8;
    case SDLK_F9:
      return Key::F9;
    case SDLK_F10:
      return Key::F10;
    case SDLK_F11:
      return Key::F11;
    case SDLK_F12:
      return Key::F12;

    // Arrows
    case SDLK_UP:
      return Key::Up;
    case SDLK_DOWN:
      return Key::Down;
    case SDLK_LEFT:
      return Key::Left;
    case SDLK_RIGHT:
      return Key::Right;

    // Modifiers
    case SDLK_LSHIFT:
      return Key::LeftShift;
    case SDLK_RSHIFT:
      return Key::RightShift;
    case SDLK_LCTRL:
      return Key::LeftControl;
    case SDLK_RCTRL:
      return Key::RightControl;
    case SDLK_LALT:
      return Key::LeftAlt;
    case SDLK_RALT:
      return Key::RightAlt;
    case SDLK_LGUI:
      return Key::LeftSuper;
    case SDLK_RGUI:
      return Key::RightSuper;

    // Navigation
    case SDLK_TAB:
      return Key::Tab;
    case SDLK_RETURN:
      return Key::Enter;
    case SDLK_ESCAPE:
      return Key::Escape;
    case SDLK_BACKSPACE:
      return Key::Backspace;
    case SDLK_SPACE:
      return Key::Space;
    case SDLK_INSERT:
      return Key::Insert;
    case SDLK_DELETE:
      return Key::Delete;
    case SDLK_HOME:
      return Key::Home;
    case SDLK_END:
      return Key::End;
    case SDLK_PAGEUP:
      return Key::PageUp;
    case SDLK_PAGEDOWN:
      return Key::PageDown;

    // Symbols
    case SDLK_GRAVE:
      return Key::Grave;
    case SDLK_MINUS:
      return Key::Minus;
    case SDLK_EQUALS:
      return Key::Equal;
    case SDLK_LEFTBRACKET:
      return Key::LeftBracket;
    case SDLK_RIGHTBRACKET:
      return Key::RightBracket;
    case SDLK_BACKSLASH:
      return Key::Backslash;
    case SDLK_SEMICOLON:
      return Key::Semicolon;
    case SDLK_APOSTROPHE:
      return Key::Apostrophe;
    case SDLK_COMMA:
      return Key::Comma;
    case SDLK_PERIOD:
      return Key::Period;
    case SDLK_SLASH:
      return Key::Slash;

    // Locks
    case SDLK_CAPSLOCK:
      return Key::CapsLock;
    case SDLK_NUMLOCKCLEAR:
      return Key::NumLock;
    case SDLK_SCROLLLOCK:
      return Key::ScrollLock;

    // Misc
    case SDLK_PRINTSCREEN:
      return Key::PrintScreen;
    case SDLK_PAUSE:
      return Key::Pause;
    case SDLK_MENU:
      return Key::Menu;

    // Keypad
    case SDLK_KP_0:
      return Key::Keypad0;
    case SDLK_KP_1:
      return Key::Keypad1;
    case SDLK_KP_2:
      return Key::Keypad2;
    case SDLK_KP_3:
      return Key::Keypad3;
    case SDLK_KP_4:
      return Key::Keypad4;
    case SDLK_KP_5:
      return Key::Keypad5;
    case SDLK_KP_6:
      return Key::Keypad6;
    case SDLK_KP_7:
      return Key::Keypad7;
    case SDLK_KP_8:
      return Key::Keypad8;
    case SDLK_KP_9:
      return Key::Keypad9;
    case SDLK_KP_DECIMAL:
      return Key::KeypadDecimal;
    case SDLK_KP_DIVIDE:
      return Key::KeypadDivide;
    case SDLK_KP_MULTIPLY:
      return Key::KeypadMultiply;
    case SDLK_KP_MINUS:
      return Key::KeypadSubtract;
    case SDLK_KP_PLUS:
      return Key::KeypadAdd;
    case SDLK_KP_ENTER:
      return Key::KeypadEnter;
    case SDLK_KP_EQUALS:
      return Key::KeypadEqual;

    default:
      return Key::Unknown;
  }
}

}  // namespace maple