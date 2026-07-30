#include "bit_writer.h"

#include <cassert>

#include "utils.h"

namespace maple {

void BitWriter::WriteBits(uint32_t value, uint32_t bits) {
  assert(bits > 0);
  assert(bits <= 32);
  assert(!mFinalized);
  // make sure value isn't bigger than the number of bits, otherwise it could lead to silent truncation bugs
  assert(uint64_t(value) <= (1ull << bits) - 1);

  mScratch |= uint64_t(value) << mScratchBits;

  const uint32_t newScratchBits = mScratchBits + bits;

  if (newScratchBits < 32) {
    mScratchBits = newScratchBits;
  } else {
    mBuffer.emplace_back(ToLittleEndian(uint32_t(mScratch)));
    mScratch >>= 32;
    mScratchBits = newScratchBits - 32;
  }
}

size_t BitWriter::GetBytesWritten() const { return mBuffer.size() * sizeof(decltype(mBuffer)::value_type) + (mScratchBits + 7) / 8; }

BitWriter::Checkpoint BitWriter::GetCheckpoint() const {
  return {
    .scratch = mScratch,
    .scratchBits = mScratchBits,
    .bufferSize = mBuffer.size(),
  };
}

void BitWriter::RollBackToCheckpoint(const Checkpoint& cp) {
  assert(cp.scratchBits < 32);
  assert(cp.bufferSize <= mBuffer.size());
  mScratch = cp.scratch;
  mScratchBits = cp.scratchBits;
  mBuffer.resize(cp.bufferSize);
}

// Finalizes the writer buffer
// Writer cannot be written to after flushing
void BitWriter::Finalize() {
  assert(mScratchBits < 32);
  assert(!mFinalized);

  // mScratchBits is always < 32 between WriteBits calls.
  if (mScratchBits > 0) {
    mBuffer.emplace_back(ToLittleEndian(uint32_t(mScratch)));
  }

  mFinalized = true;
}

std::vector<uint32_t>& BitWriter::GetBuffer() {
  assert(mFinalized);
  return mBuffer;
}

}  // namespace maple