#pragma once

#include "bit_reader.h"

namespace maple {
class ReadStream {
 public:
  static constexpr bool IsWriting = false;
  static constexpr bool IsReading = true;

  ReadStream() = default;
  ReadStream(const uint32_t* const buffer, uint32_t numBytes);

  [[nodiscard]]
  bool SerializeBits(uint64_t& value, uint32_t numBits);
  [[nodiscard]]
  bool SerializeBool(bool& value);
  [[nodiscard]]
  bool SerializeInt(int64_t& value, int64_t minInclusive, int64_t maxExclusive);
  [[nodiscard]]
  bool SerializeFloat(float& value);
  [[nodiscard]]
  bool SerializeFloatCompressed(float& value, float min, float max, float res);
  [[nodiscard]]
  bool SerializeDouble(double& value);

 private:
  BitReader mReader;
};
}  // namespace maple