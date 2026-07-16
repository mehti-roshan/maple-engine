#pragma once

#include <cstdint>

namespace maple {

enum class Key : uint8_t {
  Unknown = 0,

  // Letters
  A,
  B,
  C,
  D,
  E,
  F,
  G,
  H,
  I,
  J,
  K,
  L,
  M,
  N,
  O,
  P,
  Q,
  R,
  S,
  T,
  U,
  V,
  W,
  X,
  Y,
  Z,

  // Numbers
  Num0,
  Num1,
  Num2,
  Num3,
  Num4,
  Num5,
  Num6,
  Num7,
  Num8,
  Num9,

  // Function keys
  F1,
  F2,
  F3,
  F4,
  F5,
  F6,
  F7,
  F8,
  F9,
  F10,
  F11,
  F12,

  // Arrows
  Up,
  Down,
  Left,
  Right,

  // Modifiers
  LeftShift,
  RightShift,
  LeftControl,
  RightControl,
  LeftAlt,
  RightAlt,
  LeftSuper,
  RightSuper,

  // Navigation
  Tab,
  Enter,
  Escape,
  Backspace,
  Space,
  Insert,
  Delete,
  Home,
  End,
  PageUp,
  PageDown,

  // Symbols
  Grave,
  Minus,
  Equal,
  LeftBracket,
  RightBracket,
  Backslash,
  Semicolon,
  Apostrophe,
  Comma,
  Period,
  Slash,

  // Lock keys
  CapsLock,
  NumLock,
  ScrollLock,

  // Misc
  PrintScreen,
  Pause,
  Menu,

  // Keypad
  Keypad0,
  Keypad1,
  Keypad2,
  Keypad3,
  Keypad4,
  Keypad5,
  Keypad6,
  Keypad7,
  Keypad8,
  Keypad9,
  KeypadDecimal,
  KeypadDivide,
  KeypadMultiply,
  KeypadSubtract,
  KeypadAdd,
  KeypadEnter,
  KeypadEqual,

  Count,
};

enum class MouseButton : uint8_t {
  Unknown = 0,

  Left,
  Right,
  Middle,

  Side1,
  Side2,

  Count,
};

enum class GamePadButton : uint8_t {
  A,
  B,
  X,
  Y,

  LeftBumper,
  RightBumper,

  Back,
  Start,
  Guide,

  LeftThumb,
  RightThumb,

  DPadUp,
  DPadRight,
  DPadDown,
  DPadLeft,

  Count,
};

enum class GamePadAxis : uint8_t {
  LeftX,
  LeftY,

  RightX,
  RightY,

  LeftTrigger,
  RightTrigger,

  Count,
};
}  // namespace maple