#include "read_stream.h"

#include <cassert>
#include <cmath>

#include "utils.h"

namespace maple {

ReadStream::ReadStream(const uint32_t* const buffer, uint32_t numBytes) : mReader(buffer, numBytes) {}

bool ReadStream::SerializeBits(uint64_t& value, uint32_t numBits) {
  assert(numBits > 0);
  assert(numBits <= 64);

  uint32_t lowBits = std::min(numBits, 32u);
  uint32_t highBits = numBits - lowBits;

  uint32_t low = 0;
  uint32_t high = 0;

  bool result = mReader.ReadBits(low, lowBits);
  if (!result) return false;

  if (highBits > 0) {
    bool result = mReader.ReadBits(high, highBits);
    if (!result) return false;
  }

  value = uint64_t(low) | (uint64_t(high) << 32);
  return true;
}

bool ReadStream::SerializeBool(bool& value) {
  uint64_t v = 0;
  bool result = SerializeBits(v, 1);
  if (!result) return false;
  value = v;
  return true;
}

bool ReadStream::SerializeInt(int64_t& value, int64_t minInclusive, int64_t maxExclusive) {
  assert(minInclusive + 1 < maxExclusive);

  const uint32_t bitsRequired = BitsRequired(maxExclusive - minInclusive);
  uint64_t encoded = 0;
  bool result = SerializeBits(encoded, bitsRequired);
  if (!result) return false;

  value = int64_t(encoded) + minInclusive;
  if (value < minInclusive) return false;
  if (value >= maxExclusive) return false;

  return true;
}

bool ReadStream::SerializeFloat(float& value) {
  union {
    uint32_t i;
    float f;
  } v;

  uint64_t uint64 = 0;
  bool result = SerializeBits(uint64, sizeof(value) * 8);
  if (!result) return false;
  v.i = uint32_t(uint64);

  if (!std::isfinite(v.f)) return false;

  value = v.f;
  return true;
}

bool ReadStream::SerializeFloatCompressed(float& value, float minInclusive, float maxInclusive, float res) {
  assert(std::isfinite(minInclusive));
  assert(std::isfinite(maxInclusive));
  assert(std::isfinite(res));

  assert(minInclusive < maxInclusive);
  assert(res > 0.0f);

  const float delta = maxInclusive - minInclusive;
  const float values = delta / res;
  const uint64_t maxIntValue = std::ceil(values);

  int64_t intValue = 0;
  bool result = SerializeInt(intValue, 0, maxIntValue + 1);
  if (!result) return false;

  const float normalizedValue = intValue / float(maxIntValue);
  const float tmp = normalizedValue * delta + minInclusive;

  if (!std::isfinite(tmp)) return false;
  if (tmp < minInclusive) return false;
  if (tmp > maxInclusive) return false;
  value = tmp;
  return true;
}

bool ReadStream::SerializeDouble(double& value) {
  union {
    uint64_t i;
    double f;
  } v;

  bool result = SerializeBits(v.i, sizeof(value) * 8);
  if (!result) return false;

  if (!std::isfinite(v.f)) return false;

  value = v.f;
  return true;
}

}  // namespace maple