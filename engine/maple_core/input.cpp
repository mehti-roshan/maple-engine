#include "input.h"

#include <GLFW/glfw3.h>

#include <cstddef>
#include <glm/common.hpp>
#include <glm/geometric.hpp>
#include <variant>

#include "input_enums.h"
#include "log_macros.h"

namespace maple {

// TODO: update ActiveInputDevice

void Input::BeginFrame() {
  // Set previous to current
  for (auto& v : mKeys) v.Advance();
  for (auto& v : mMouseKeys) v.Advance();
  for (auto& v : mJoyStates) v.second.Advance();
  mMousePos.Advance();
  mMouseScroll.Advance();
}

void Input::Bind(const std::string& name, const Binding& binding) { mBindings[name].push_back(binding); }

bool Input::Pressed(const std::string& name) const {
  auto it = mBindings.find(name);
  MAPLE_ASSERT(it != mBindings.end(), "failed to find binding for '{}'", name);

  for (const auto& binding : it->second) {
    if (auto* t = std::get_if<Key>(&binding.type)) {
      if (mKeys[static_cast<size_t>(*t)].current && !mKeys[static_cast<size_t>(*t)].previous) return true;
    } else if (auto* t = std::get_if<MouseButton>(&binding.type)) {
      if (mMouseKeys[static_cast<size_t>(*t)].current && !mMouseKeys[static_cast<size_t>(*t)].previous) return true;
    } else if (auto* t = std::get_if<GamepadButton>(&binding.type)) {
      for (auto [id, state] : mJoyStates) {
        if (state.current.buttons[static_cast<size_t>(*t)] && !state.previous.buttons[static_cast<size_t>(*t)]) return true;
      }
    } else if (auto* t = std::get_if<GamepadAxis>(&binding.type)) {
      for (auto [id, state] : mJoyStates) {
        if (state.current.axes[static_cast<size_t>(*t)] > binding.threshold && state.previous.axes[static_cast<size_t>(*t)] <= binding.threshold)
          return true;
      }
    }
  }

  return false;
}
bool Input::Released(const std::string& name) const {
  auto it = mBindings.find(name);
  MAPLE_ASSERT(it != mBindings.end(), "failed to find binding for '{}'", name);

  for (const auto& binding : it->second) {
    if (auto* t = std::get_if<Key>(&binding.type)) {
      if (!mKeys[static_cast<size_t>(*t)].current && mKeys[static_cast<size_t>(*t)].previous) return true;
    } else if (auto* t = std::get_if<MouseButton>(&binding.type)) {
      if (!mMouseKeys[static_cast<size_t>(*t)].current && mMouseKeys[static_cast<size_t>(*t)].previous) return true;
    } else if (auto* t = std::get_if<GamepadButton>(&binding.type)) {
      for (auto [id, state] : mJoyStates) {
        if (!state.current.buttons[static_cast<size_t>(*t)] && state.previous.buttons[static_cast<size_t>(*t)]) return true;
      }
    } else if (auto* t = std::get_if<GamepadAxis>(&binding.type)) {
      for (auto [id, state] : mJoyStates) {
        if (state.current.axes[static_cast<size_t>(*t)] <= binding.threshold && state.previous.axes[static_cast<size_t>(*t)] > binding.threshold)
          return true;
      }
    }
  }

  return false;
}

float Input::Value(const std::string& name) const {
  auto it = mBindings.find(name);
  MAPLE_ASSERT(it != mBindings.end(), "failed to find binding for '{}'", name);

  float clamped = 0.0f;
  for (const auto& binding : it->second) {
    if (auto* t = std::get_if<Key>(&binding.type)) {
      if (mKeys[static_cast<size_t>(*t)].current) clamped += binding.positive ? 1.0f : -1.0f;
    } else if (auto* t = std::get_if<MouseButton>(&binding.type)) {
      if (mMouseKeys[static_cast<size_t>(*t)].current) clamped += binding.positive ? 1.0f : -1.0f;
    } else if (auto* t = std::get_if<GamepadButton>(&binding.type)) {
      for (auto [id, state] : mJoyStates) {
        if (state.current.buttons[static_cast<size_t>(*t)]) clamped += binding.positive ? 1.0f : -1.0f;
      }
    } else if (auto* t = std::get_if<GamepadAxis>(&binding.type)) {
      for (auto [id, state] : mJoyStates) {
        float v = state.current.axes[static_cast<size_t>(*t)];
        v *= binding.positive ? 1.0f : -1.0f;
        clamped += v;
      }
    }
  }

  return glm::clamp<float>(clamped, -1.0f, 1.0);
}

glm::vec2 Input::GetMousePos() const { return mMousePos.current; }
glm::vec2 Input::GetMouseDelta() const { return mMousePos.current - mMousePos.previous; }
glm::vec2 Input::GetScrollDelta() const { return mMouseScroll.current; }

glm::vec2 ApplyDeadZone(glm::vec2 v, float deadZone) {
  auto len = glm::length(v);
  if (len < deadZone) return glm::vec2(0);
  float scaled = (len - deadZone) / (1.0f - deadZone);
  return glm::normalize(v) * scaled;
}

Input::JoystickState ConvertFromGlfwGamePad(const Input::JoystickState& glfw, float leftStickDeadZone, float rightStickDeadZone) {
  Input::JoystickState v{};

  auto leftStickDead = ApplyDeadZone(glm::vec2(glfw.axes[GLFW_GAMEPAD_AXIS_LEFT_X], glfw.axes[GLFW_GAMEPAD_AXIS_LEFT_Y]), leftStickDeadZone);
  auto rightStickDead = ApplyDeadZone(glm::vec2(glfw.axes[GLFW_GAMEPAD_AXIS_RIGHT_X], glfw.axes[GLFW_GAMEPAD_AXIS_RIGHT_Y]), rightStickDeadZone);

  v.axes[static_cast<size_t>(GamepadAxis::LeftX)] = leftStickDead.x;
  v.axes[static_cast<size_t>(GamepadAxis::LeftY)] = leftStickDead.y;
  v.axes[static_cast<size_t>(GamepadAxis::RightX)] = rightStickDead.x;
  v.axes[static_cast<size_t>(GamepadAxis::RightY)] = rightStickDead.y;
  v.axes[static_cast<size_t>(GamepadAxis::LeftTrigger)] = glfw.axes[GLFW_GAMEPAD_AXIS_LEFT_TRIGGER];
  v.axes[static_cast<size_t>(GamepadAxis::RightTrigger)] = glfw.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER];

  v.buttons[static_cast<size_t>(GamepadButton::A)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_A] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::B)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_B] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::X)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_X] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::Y)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_Y] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::LeftBumper)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_LEFT_BUMPER] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::RightBumper)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_BUMPER] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::Back)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_BACK] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::Start)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_START] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::Guide)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_GUIDE] != GLFW_RELEASE;  
  v.buttons[static_cast<size_t>(GamepadButton::LeftThumb)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_LEFT_THUMB] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::RightThumb)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_RIGHT_THUMB] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::DPadUp)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_DPAD_UP] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::DPadRight)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_DPAD_RIGHT] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::DPadDown)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_DPAD_DOWN] != GLFW_RELEASE;
  v.buttons[static_cast<size_t>(GamepadButton::DPadLeft)] = glfw.buttons[GLFW_GAMEPAD_BUTTON_DPAD_LEFT] != GLFW_RELEASE;

  return v;
}

void Input::OnJoySticks(const std::vector<std::pair<int32_t, JoystickState>>& joysticks) {
  for (auto& joy : joysticks) {
    mJoyStates[joy.first].current = ConvertFromGlfwGamePad(joy.second, mLeftStickDeadZone, mRightStickDeadZone);
  }
}

void Input::OnMouseScroll(double xoffset, double yoffset) { mMouseScroll.current = glm::vec2(xoffset, yoffset); }

void Input::OnCursorPos(double xpos, double ypos) { mMousePos.current = glm::vec2(xpos, ypos); }

void Input::OnMouseButtons(int glfwKey, int action, int mods) {
  MouseButton key;
  switch (glfwKey) {
    case GLFW_MOUSE_BUTTON_1:
      key = MouseButton::Left;
      break;
    case GLFW_MOUSE_BUTTON_2:
      key = MouseButton::Right;
      break;
    case GLFW_MOUSE_BUTTON_3:
      key = MouseButton::Middle;
      break;
    case GLFW_MOUSE_BUTTON_4:
      key = MouseButton::Button4;
      break;
    case GLFW_MOUSE_BUTTON_5:
      key = MouseButton::Button5;
      break;
    case GLFW_MOUSE_BUTTON_6:
      key = MouseButton::Button6;
      break;
    case GLFW_MOUSE_BUTTON_7:
      key = MouseButton::Button7;
      break;
    case GLFW_MOUSE_BUTTON_8:
      key = MouseButton::Button8;
      break;
    default:
      return;  // Unknown mouse key
  }
  mMouseKeys[static_cast<size_t>(key)].current = action != GLFW_RELEASE;
}

ActiveInputDevice Input::GetInputDevice() const { return mActiveInputDevice; }

void Input::SetRightStickDeadZone(float v) { mRightStickDeadZone = glm::clamp<float>(v, 0.0f, 1.0f); }
void Input::SetLeftStickDeadZone(float v) { mLeftStickDeadZone = glm::clamp<float>(v, 0.0f, 1.0f); }

void Input::OnKey(int glfwKey, int scancode, int action, int mods) {
  Key key;

  switch (glfwKey) {
    case GLFW_KEY_SPACE:
      key = Key::Space;
      break;
    case GLFW_KEY_APOSTROPHE:
      key = Key::Apostrophe;
      break;
    case GLFW_KEY_COMMA:
      key = Key::Comma;
      break;
    case GLFW_KEY_MINUS:
      key = Key::Minus;
      break;
    case GLFW_KEY_PERIOD:
      key = Key::Period;
      break;
    case GLFW_KEY_SLASH:
      key = Key::Slash;
      break;
    case GLFW_KEY_0:
      key = Key::Num0;
      break;
    case GLFW_KEY_1:
      key = Key::Num1;
      break;
    case GLFW_KEY_2:
      key = Key::Num2;
      break;
    case GLFW_KEY_3:
      key = Key::Num3;
      break;
    case GLFW_KEY_4:
      key = Key::Num4;
      break;
    case GLFW_KEY_5:
      key = Key::Num5;
      break;
    case GLFW_KEY_6:
      key = Key::Num6;
      break;
    case GLFW_KEY_7:
      key = Key::Num7;
      break;
    case GLFW_KEY_8:
      key = Key::Num8;
      break;
    case GLFW_KEY_9:
      key = Key::Num9;
      break;
    case GLFW_KEY_SEMICOLON:
      key = Key::Semicolon;
      break;
    case GLFW_KEY_EQUAL:
      key = Key::Equal;
      break;
    case GLFW_KEY_A:
      key = Key::A;
      break;
    case GLFW_KEY_B:
      key = Key::B;
      break;
    case GLFW_KEY_C:
      key = Key::C;
      break;
    case GLFW_KEY_D:
      key = Key::D;
      break;
    case GLFW_KEY_E:
      key = Key::E;
      break;
    case GLFW_KEY_F:
      key = Key::F;
      break;
    case GLFW_KEY_G:
      key = Key::G;
      break;
    case GLFW_KEY_H:
      key = Key::H;
      break;
    case GLFW_KEY_I:
      key = Key::I;
      break;
    case GLFW_KEY_J:
      key = Key::J;
      break;
    case GLFW_KEY_K:
      key = Key::K;
      break;
    case GLFW_KEY_L:
      key = Key::L;
      break;
    case GLFW_KEY_M:
      key = Key::M;
      break;
    case GLFW_KEY_N:
      key = Key::N;
      break;
    case GLFW_KEY_O:
      key = Key::O;
      break;
    case GLFW_KEY_P:
      key = Key::P;
      break;
    case GLFW_KEY_Q:
      key = Key::Q;
      break;
    case GLFW_KEY_R:
      key = Key::R;
      break;
    case GLFW_KEY_S:
      key = Key::S;
      break;
    case GLFW_KEY_T:
      key = Key::T;
      break;
    case GLFW_KEY_U:
      key = Key::U;
      break;
    case GLFW_KEY_V:
      key = Key::V;
      break;
    case GLFW_KEY_W:
      key = Key::W;
      break;
    case GLFW_KEY_X:
      key = Key::X;
      break;
    case GLFW_KEY_Y:
      key = Key::Y;
      break;
    case GLFW_KEY_Z:
      key = Key::Z;
      break;
    case GLFW_KEY_LEFT_BRACKET:
      key = Key::LeftBracket;
      break;
    case GLFW_KEY_BACKSLASH:
      key = Key::Backslash;
      break;
    case GLFW_KEY_RIGHT_BRACKET:
      key = Key::RightBracket;
      break;
    case GLFW_KEY_GRAVE_ACCENT:
      key = Key::Grave;
      break;
    // case GLFW_KEY_WORLD_1:
    //   key = Key::International1;
    //   break;
    // case GLFW_KEY_WORLD_2:
    //   key = Key::International2;
    //   break;
    case GLFW_KEY_ESCAPE:
      key = Key::Escape;
      break;
    case GLFW_KEY_ENTER:
      key = Key::Enter;
      break;
    case GLFW_KEY_TAB:
      key = Key::Tab;
      break;
    case GLFW_KEY_BACKSPACE:
      key = Key::Backspace;
      break;
    case GLFW_KEY_INSERT:
      key = Key::Insert;
      break;
    case GLFW_KEY_DELETE:
      key = Key::Delete;
      break;
    case GLFW_KEY_RIGHT:
      key = Key::Right;
      break;
    case GLFW_KEY_LEFT:
      key = Key::Left;
      break;
    case GLFW_KEY_DOWN:
      key = Key::Down;
      break;
    case GLFW_KEY_UP:
      key = Key::Up;
      break;
    case GLFW_KEY_PAGE_UP:
      key = Key::PageUp;
      break;
    case GLFW_KEY_PAGE_DOWN:
      key = Key::PageDown;
      break;
    case GLFW_KEY_HOME:
      key = Key::Home;
      break;
    case GLFW_KEY_END:
      key = Key::End;
      break;
    case GLFW_KEY_CAPS_LOCK:
      key = Key::CapsLock;
      break;
    case GLFW_KEY_SCROLL_LOCK:
      key = Key::ScrollLock;
      break;
    case GLFW_KEY_NUM_LOCK:
      key = Key::NumLock;
      break;
    case GLFW_KEY_PRINT_SCREEN:
      key = Key::PrintScreen;
      break;
    case GLFW_KEY_PAUSE:
      key = Key::Pause;
      break;
    case GLFW_KEY_F1:
      key = Key::F1;
      break;
    case GLFW_KEY_F2:
      key = Key::F2;
      break;
    case GLFW_KEY_F3:
      key = Key::F3;
      break;
    case GLFW_KEY_F4:
      key = Key::F4;
      break;
    case GLFW_KEY_F5:
      key = Key::F5;
      break;
    case GLFW_KEY_F6:
      key = Key::F6;
      break;
    case GLFW_KEY_F7:
      key = Key::F7;
      break;
    case GLFW_KEY_F8:
      key = Key::F8;
      break;
    case GLFW_KEY_F9:
      key = Key::F9;
      break;
    case GLFW_KEY_F10:
      key = Key::F10;
      break;
    case GLFW_KEY_F11:
      key = Key::F11;
      break;
    case GLFW_KEY_F12:
      key = Key::F12;
      break;
    // case GLFW_KEY_F13:
    //   key = Key::F13;
    //   break;
    // case GLFW_KEY_F14:
    //   key = Key::F14;
    //   break;
    // case GLFW_KEY_F15:
    //   key = Key::F15;
    //   break;
    // case GLFW_KEY_F16:
    //   key = Key::F16;
    //   break;
    // case GLFW_KEY_F17:
    //   key = Key::F17;
    //   break;
    // case GLFW_KEY_F18:
    //   key = Key::F18;
    //   break;
    // case GLFW_KEY_F19:
    //   key = Key::F19;
    //   break;
    // case GLFW_KEY_F20:
    //   key = Key::F20;
    //   break;
    // case GLFW_KEY_F21:
    //   key = Key::F21;
    //   break;
    // case GLFW_KEY_F22:
    //   key = Key::F22;
    //   break;
    // case GLFW_KEY_F23:
    //   key = Key::F23;
    //   break;
    // case GLFW_KEY_F24:
    //   key = Key::F24;
    //   break;
    // case GLFW_KEY_F25:
    //   key = Key::F25;
    //   break;
    case GLFW_KEY_KP_0:
      key = Key::Keypad0;
      break;
    case GLFW_KEY_KP_1:
      key = Key::Keypad1;
      break;
    case GLFW_KEY_KP_2:
      key = Key::Keypad2;
      break;
    case GLFW_KEY_KP_3:
      key = Key::Keypad3;
      break;
    case GLFW_KEY_KP_4:
      key = Key::Keypad4;
      break;
    case GLFW_KEY_KP_5:
      key = Key::Keypad5;
      break;
    case GLFW_KEY_KP_6:
      key = Key::Keypad6;
      break;
    case GLFW_KEY_KP_7:
      key = Key::Keypad7;
      break;
    case GLFW_KEY_KP_8:
      key = Key::Keypad8;
      break;
    case GLFW_KEY_KP_9:
      key = Key::Keypad9;
      break;
    case GLFW_KEY_KP_DECIMAL:
      key = Key::KeypadDecimal;
      break;
    case GLFW_KEY_KP_DIVIDE:
      key = Key::KeypadDivide;
      break;
    case GLFW_KEY_KP_MULTIPLY:
      key = Key::KeypadMultiply;
      break;
    case GLFW_KEY_KP_SUBTRACT:
      key = Key::KeypadSubtract;
      break;
    case GLFW_KEY_KP_ADD:
      key = Key::KeypadAdd;
      break;
    case GLFW_KEY_KP_ENTER:
      key = Key::KeypadEnter;
      break;
    case GLFW_KEY_KP_EQUAL:
      key = Key::Equal;
      break;
    case GLFW_KEY_LEFT_SHIFT:
      key = Key::LeftShift;
      break;
    case GLFW_KEY_LEFT_CONTROL:
      key = Key::LeftControl;
      break;
    case GLFW_KEY_LEFT_ALT:
      key = Key::LeftAlt;
      break;
    case GLFW_KEY_LEFT_SUPER:
      key = Key::LeftSuper;
      break;
    case GLFW_KEY_RIGHT_SHIFT:
      key = Key::RightShift;
      break;
    case GLFW_KEY_RIGHT_CONTROL:
      key = Key::RightControl;
      break;
    case GLFW_KEY_RIGHT_ALT:
      key = Key::RightAlt;
      break;
    case GLFW_KEY_RIGHT_SUPER:
      key = Key::RightSuper;
      break;
    case GLFW_KEY_MENU:
      key = Key::Menu;
      break;
    default:
      return;  // Unknown key
  }

  mKeys[static_cast<size_t>(key)].current = action != GLFW_RELEASE;
}

}  // namespace maple