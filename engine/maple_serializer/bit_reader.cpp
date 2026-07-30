#include "bit_reader.h"

#include "utils.h"

namespace maple {
BitReader::BitReader(const uint32_t* const buffer, uint32_t numBytes) : mBuffer(buffer), mNumWords((numBytes + 3) / 4) {
  assert(buffer);
  assert(numBytes > 0);
  assert(numBytes % 4 == 0);
}

bool BitReader::ReadBits(uint32_t& value, uint32_t bits) {
  assert(mBuffer);
  assert(bits > 0);
  assert(bits <= 32);
  if (mNumBitsRead + bits > mNumWords * 4 * 8) return false;

  const uint32_t wordIdx = mNumBitsRead / 32;
  const uint32_t bitOffset = mNumBitsRead % 32;

  uint64_t w1 = ToHostEndian(mBuffer[wordIdx]);
  uint64_t w2 = 0;
  if (wordIdx + 1 < mNumWords) {
    w2 = ToHostEndian(mBuffer[wordIdx + 1]);
  }

  uint64_t v = w1 | (w2 << 32);

  v >>= bitOffset;
  const uint64_t mask = (1ull << bits) - 1;
  value = uint32_t(v & mask);

  mNumBitsRead += bits;

  return true;
}
}  // namespace maple