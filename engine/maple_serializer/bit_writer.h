#pragma once

#include <cstdint>
#include <vector>

namespace maple {
class BitWriter {
 public:
  void WriteBits(uint32_t value, uint32_t bits);

  // Gets the total number of bytes needed to store the current data
  // this can be used to check if we have exceeded a serialization budget
  // WARNING: to actually use the buffer you need to call Finalize() to flush everything
  uint64_t GetBytesWritten() const;

  struct Checkpoint {
    uint64_t scratch = 0;
    uint32_t scratchBits = 0;
    size_t bufferSize = 0;
  };

  // Gets the current writer checkpoint
  // useful for cases when there is a serialization budget (e.g. writing to a network buffer with an MTU)
  // if after serializing the next item, the budget gets exceeded, we can use this checkpoint to roll back to a valid point
  Checkpoint GetCheckpoint() const;
  void RollBackToCheckpoint(const Checkpoint& cp);

  // Finalizes & flushes the writer buffer
  // Writer cannot be written to after finalizing
  void Finalize();

  // WARNING: Writer must have been finalized before calling this
  std::vector<uint32_t>& GetBuffer();

 private:
  uint64_t mScratch = 0;
  uint32_t mScratchBits = 0;
  std::vector<uint32_t> mBuffer;
  bool mFinalized = false;
};
}  // namespace maple