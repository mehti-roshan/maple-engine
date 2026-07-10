#pragma once

#include <cstdint>
#include <memory>

namespace maple {
class PRNG {
 public:
  PRNG() = default;
  PRNG(uint64_t seed);
  ~PRNG();

  float NextFloat(float min = 0.0f, float max = 1.0f) const;
  double NextDouble(double min = 0.0, double max = 1.0) const;
  int64_t NextInt64(int64_t min = INT32_MIN, int64_t max = INT32_MAX) const;
  uint64_t NextUInt64(uint64_t min = 0, uint64_t max = UINT64_MAX) const;
  bool NextChance(double probability) const;  // Bernoulli, probability between 0 -> 1

 private:
  struct Impl;
  std::unique_ptr<Impl> mImpl;
};
}  // namespace maple