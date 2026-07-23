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

namespace maple {

class Input {
 public:
  struct Binding {
    std::variant<InputKey, InputMouseButton, InputGamePadButton, InputGamePadAxis> type;
    bool positive = true;    // whether input will get flipped
    float threshold = 0.5f;  // if Axes are used as buttons, will be considered true when exceeding this threshold
  };

  struct GamePadState {
    std::array<bool, static_cast<size_t>(InputGamePadButton::Count)> buttons;
    std::array<float, static_cast<size_t>(InputGamePadAxis::Count)> axes;
  };

  void BeginFrame();
  void Bind(const std::string& name, const Binding& binding);
  void Bind(const std::string& name, const std::vector<Binding>& bindings);

  bool Pressed(const std::string& name) const;
  bool Released(const std::string& name) const;

  float Value(const std::string& name) const;

  glm::vec2 GetMousePos() const;
  glm::vec2 GetMouseDelta() const;
  glm::vec2 GetScrollDelta() const;

  ActiveInputDevice GetInputDevice() const;

  void SetRightStickDeadZone(float v);
  void SetLeftStickDeadZone(float v);

  void OnJoySticks(const std::vector<std::pair<int32_t, GamePadState>>&);
  void OnKey(InputKey key, bool pressed);
  void OnMouseButtons(InputMouseButton button, bool pressed);
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

  std::unordered_map<int32_t, State<GamePadState>> mGamePads;
  std::array<State<bool>, static_cast<size_t>(InputKey::Count)> mKeys;
  std::array<State<bool>, static_cast<size_t>(InputMouseButton::Count)> mMouseKeys;
  State<glm::vec2> mMousePos;
  State<glm::vec2> mMouseScroll;

  ActiveInputDevice mActiveInputDevice = ActiveInputDevice::MouseKeyboard;
};

}  // namespace maple