#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <glm/glm.hpp>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "input_enums.h"

struct GLFWgamepadstate;
struct GLFWwindow;

namespace maple {

class Input {
 public:
  void BeginFrame();

  struct Binding {
    std::variant<Key, MouseButton, GamepadButton, GamepadAxis> type;
    bool positive = true;    // whether input will get flipped
    float threshold = 0.5f;  // if Axes are used as buttons, will be considered true when exceeding this threshold
  };

  void Bind(const std::string& name, const Binding& binding);

  bool Pressed(const std::string& name) const;
  bool Released(const std::string& name) const;

  float Value(const std::string& name) const;

  glm::vec2 GetMousePos() const;
  glm::vec2 GetMouseDelta() const;
  glm::vec2 GetScrollDelta() const;

  ActiveInputDevice GetInputDevice() const;

  void SetRightStickDeadZone(float v);
  void SetLeftStickDeadZone(float v);

  struct JoystickState {
    unsigned char buttons[15];
    float axes[6];
  };

  void OnJoySticks(const std::vector<std::pair<int32_t, JoystickState>>&);
  void OnKey(int key, int scancode, int action, int mods);
  void OnMouseButtons(int button, int action, int mods);
  void OnMouseScroll(double xOffset, double yOffset);
  void OnCursorPos(double xPos, double yPos);

 private:
  std::unordered_map<std::string, std::vector<Binding>> mBindings;
  float mRightStickDeadZone = 0.01f;
  float mLeftStickDeadZone = 0.01f;

  template <typename T>
  struct State {
    T previous{};
    T current{};

    void Advance() { previous = current; }
  };

  std::unordered_map<int32_t, State<JoystickState>> mJoyStates;
  std::array<State<bool>, static_cast<size_t>(Key::Count)> mKeys;
  std::array<State<bool>, static_cast<size_t>(MouseButton::Count)> mMouseKeys;
  State<glm::vec2> mMousePos;
  State<glm::vec2> mMouseScroll;

  struct GamePad {
    std::array<bool, static_cast<size_t>(GamepadButton::Count)> mKeys;
    std::array<float, static_cast<size_t>(GamepadAxis::Count)> mAxes;
  };

  ActiveInputDevice mActiveInputDevice = ActiveInputDevice::MouseKeyboard;
};

}  // namespace maple