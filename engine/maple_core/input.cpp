#include "input.h"

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
  // for (auto& v : mMouseKeys) v.Advance();
  for (auto& v : mGamePads) v.second.Advance();
  mMousePos.Advance();
  mMouseScroll.Advance();
}

void Input::Bind(const std::string& name, const Binding& binding) { mBindings[name].push_back(binding); }

bool Input::Pressed(const std::string& name) const {
  auto it = mBindings.find(name);
  MAPLE_ASSERT(it != mBindings.end(), "failed to find binding for '{}'", name);

  for (const auto& binding : it->second) {
    if (auto* t = std::get_if<InputKey>(&binding.type)) {
      if (mKeys[static_cast<size_t>(*t)].current && !mKeys[static_cast<size_t>(*t)].previous) return true;
    } else if (auto* t = std::get_if<InputMouseButton>(&binding.type)) {
      if (mMouseKeys[static_cast<size_t>(*t)].current && !mMouseKeys[static_cast<size_t>(*t)].previous) return true;
    } else if (auto* t = std::get_if<InputGamePadButton>(&binding.type)) {
      for (auto [id, state] : mGamePads) {
        if (state.current.buttons[static_cast<size_t>(*t)] && !state.previous.buttons[static_cast<size_t>(*t)]) return true;
      }
    } else if (auto* t = std::get_if<InputGamePadAxis>(&binding.type)) {
      for (auto [id, state] : mGamePads) {
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
    if (auto* t = std::get_if<InputKey>(&binding.type)) {
      if (!mKeys[static_cast<size_t>(*t)].current && mKeys[static_cast<size_t>(*t)].previous) return true;
    } else if (auto* t = std::get_if<InputMouseButton>(&binding.type)) {
      if (!mMouseKeys[static_cast<size_t>(*t)].current && mMouseKeys[static_cast<size_t>(*t)].previous) return true;
    } else if (auto* t = std::get_if<InputGamePadButton>(&binding.type)) {
      for (auto [id, state] : mGamePads) {
        if (!state.current.buttons[static_cast<size_t>(*t)] && state.previous.buttons[static_cast<size_t>(*t)]) return true;
      }
    } else if (auto* t = std::get_if<InputGamePadAxis>(&binding.type)) {
      for (auto [id, state] : mGamePads) {
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
    if (auto* t = std::get_if<InputKey>(&binding.type)) {
      if (mKeys[static_cast<size_t>(*t)].current) clamped += binding.positive ? 1.0f : -1.0f;
    } else if (auto* t = std::get_if<InputMouseButton>(&binding.type)) {
      if (mMouseKeys[static_cast<size_t>(*t)].current) clamped += binding.positive ? 1.0f : -1.0f;
    } else if (auto* t = std::get_if<InputGamePadButton>(&binding.type)) {
      for (auto [id, state] : mGamePads) {
        if (state.current.buttons[static_cast<size_t>(*t)]) clamped += binding.positive ? 1.0f : -1.0f;
      }
    } else if (auto* t = std::get_if<InputGamePadAxis>(&binding.type)) {
      for (auto [id, state] : mGamePads) {
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

Input::GamePadState ApplyDeadZoneToState(const Input::GamePadState& pad, float leftStickDeadZone, float rightStickDeadZone) {
  Input::GamePadState v = pad;

  auto leftStickDead =
    ApplyDeadZone(glm::vec2(v.axes[static_cast<size_t>(InputGamePadAxis::LeftX)], v.axes[static_cast<size_t>(InputGamePadAxis::LeftY)]), leftStickDeadZone);
  auto rightStickDead =
    ApplyDeadZone(glm::vec2(v.axes[static_cast<size_t>(InputGamePadAxis::RightX)], v.axes[static_cast<size_t>(InputGamePadAxis::RightY)]), rightStickDeadZone);

  v.axes[static_cast<size_t>(InputGamePadAxis::LeftX)] = leftStickDead.x;
  v.axes[static_cast<size_t>(InputGamePadAxis::LeftY)] = leftStickDead.y;
  v.axes[static_cast<size_t>(InputGamePadAxis::RightX)] = rightStickDead.x;
  v.axes[static_cast<size_t>(InputGamePadAxis::RightY)] = rightStickDead.y;

  return v;
}

void Input::OnJoySticks(const std::vector<std::pair<int32_t, GamePadState>>& pads) {
  for (auto& pad : pads) {
    auto it = mGamePads.find(pad.first);
    if (it == mGamePads.end()) {
      mGamePads[pad.first] = {};
      it = mGamePads.find(pad.first);
    }

    auto e = ApplyDeadZoneToState(pad.second, mLeftStickDeadZone, mRightStickDeadZone);
    it->second.current = e;
  }
}

void Input::OnMouseScroll(double xoffset, double yoffset) { mMouseScroll.current = glm::vec2(xoffset, yoffset); }

void Input::OnCursorPos(double xpos, double ypos) { mMousePos.current = glm::vec2(xpos, ypos); }

void Input::OnMouseButtons(InputMouseButton btn, bool pressed) {
  mMouseKeys[static_cast<size_t>(btn)].Advance();
  mMouseKeys[static_cast<size_t>(btn)].current = pressed;
}

ActiveInputDevice Input::GetInputDevice() const { return mActiveInputDevice; }

void Input::SetRightStickDeadZone(float v) { mRightStickDeadZone = glm::clamp<float>(v, 0.0f, 1.0f); }
void Input::SetLeftStickDeadZone(float v) { mLeftStickDeadZone = glm::clamp<float>(v, 0.0f, 1.0f); }

void Input::OnKey(InputKey key, bool pressed) {
  mKeys[static_cast<size_t>(key)].current = pressed;
}

}  // namespace maple