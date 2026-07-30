#include "write_stream.h"

#include <algorithm>
#include <cassert>
#include <cmath>

#include "utils.h"

namespace maple {

bool WriteStream::SerializeBits(uint64_t value, uint32_t numBits) {
  assert(numBits > 0);
  assert(numBits <= 64);

  uint32_t lowBits = std::min(numBits, 32u);
  uint32_t highBits = numBits - lowBits;
  mWriter.WriteBits(uint32_t(value), lowBits);

  if (highBits > 0) {
    mWriter.WriteBits(uint32_t(value >> lowBits), highBits);
  }

  return true;
}

bool WriteStream::SerializeBool(bool value) { return SerializeBits(value, 1); }

bool WriteStream::SerializeInt(int64_t value, int64_t minInclusive, int64_t maxExclusive) {
  assert(minInclusive + 1 < maxExclusive);
  assert(value >= minInclusive);
  assert(value < maxExclusive);


  const uint32_t bitsRequired = BitsRequired(maxExclusive - minInclusive);
  const uint64_t encoded = uint64_t(value - minInclusive);

  bool result = SerializeBits(encoded, bitsRequired);
  if (!result) return false;
  return true;
}

bool WriteStream::SerializeFloat(float value) {
  assert(std::isfinite(value));

  union {
    uint32_t i;
    float f;
  } v;

  v.f = value;
  return SerializeBits(v.i, sizeof(v.i) * 8);
}

bool WriteStream::SerializeFloatCompressed(float value, float minInclusive, float maxInclusive, float res) {
  assert(std::isfinite(value));
  assert(std::isfinite(minInclusive));
  assert(std::isfinite(maxInclusive));
  assert(std::isfinite(res));

  assert(minInclusive < maxInclusive);
  assert(value >= minInclusive);
  assert(value <= maxInclusive);
  assert(res > 0.0f);

  const float delta = maxInclusive - minInclusive;
  const float values = delta / res;
  const uint64_t maxIntValue = std::ceil(values);

  const float normalizedValue = std::clamp((values - minInclusive) / delta, 0.0f, 1.0f);
  const uint64_t intValue = uint64_t(std::round(normalizedValue * maxIntValue));

  return SerializeInt(intValue, 0, maxIntValue + 1);
}

bool WriteStream::SerializeDouble(double value) {
  assert(std::isfinite(value));

  union {
    uint64_t i;
    double f;
  } v;

  v.f = value;
  return SerializeBits(v.i, sizeof(v.i) * 8);
}

void WriteStream::Finalize() { return mWriter.Finalize(); }
std::vector<uint32_t>& WriteStream::GetBuffer() { return mWriter.GetBuffer(); }

}  // namespace maple