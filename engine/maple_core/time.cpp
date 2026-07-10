#include "time.h"

#include <chrono>
#include <memory>

namespace maple {

using Clock = std::chrono::steady_clock;
using TimePoint = Clock::time_point;

struct Time::Impl {
  TimePoint mStart;
  TimePoint mPrevious;
  TimePoint mCurrent;

  float mDeltaTime = 0.0f;
  float mTimeSinceStart = 0.0f;
};

void Time::Initialize() {
  impl = std::make_unique<Impl>();
  impl->mStart = Clock::now();
  impl->mPrevious = Clock::now();
  impl->mCurrent = Clock::now();
};

void Time::BeginFrame() {
  impl->mPrevious = impl->mCurrent;
  impl->mCurrent = Clock::now();

  impl->mDeltaTime = std::chrono::duration<float>(impl->mCurrent - impl->mPrevious).count();
  impl->mTimeSinceStart = std::chrono::duration<float>(impl->mCurrent - impl->mStart).count();
};

float Time::DeltaTime() const { return impl->mDeltaTime; }
float Time::TimeSinceStart() const { return impl->mTimeSinceStart; };

Time::Time() = default;
Time::~Time() = default;
}  // namespace maple