#pragma once

#include <cassert>
#include <cstdint>

namespace maple {
class BitReader {
 public:
  BitReader() = default;
  BitReader(const uint32_t* const buffer, uint32_t numBytes);

  bool ReadBits(uint32_t& value, uint32_t bits);

 private:
  uint64_t mNumBitsRead = 0;

  const uint32_t mNumWords = 0;
  const uint32_t* const mBuffer = nullptr;
};
}  // namespace maple