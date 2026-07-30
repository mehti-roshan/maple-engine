#pragma once

#include "bit_writer.h"

namespace maple {
class WriteStream {
 public:
  static constexpr bool IsWriting = true;
  static constexpr bool IsReading = false;

  bool SerializeBits(uint64_t value, uint32_t numBits);
  bool SerializeBool(bool value);
  bool SerializeInt(int64_t value, int64_t minInclusive, int64_t maxExclusive);
  bool SerializeFloat(float value);
  bool SerializeFloatCompressed(float value, float minInclusive, float maxInclusive, float res);
  bool SerializeDouble(double value);

  // See bit_writer.h for docs
  void Finalize();
  // See bit_writer.h for docs
  std::vector<uint32_t>& GetBuffer();

 private:
  BitWriter mWriter;
};
}  // namespace maple