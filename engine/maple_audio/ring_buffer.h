#pragma once

#include <atomic>
#include <vector>

template <typename T>
class RingBuffer {
 public:
  RingBuffer() = default;
  RingBuffer(size_t capacity) {
    mBuffer.resize(capacity);
  }

  // Producer side
  size_t Write(const T* data, size_t count) {
    const size_t write = mWritePos.load(std::memory_order_relaxed);
    const size_t read = mReadPos.load(std::memory_order_acquire);

    const size_t available = mBuffer.size() - (write - read);
    const size_t amount = std::min(count, available);

    for (size_t i = 0; i < amount; ++i) {
      mBuffer[(write + i) % mBuffer.size()] = data[i];
    }

    mWritePos.store(write + amount, std::memory_order_release);

    return amount;
  }

  // Consumer side
  size_t Read(T* output, size_t count) {
    const size_t read = mReadPos.load(std::memory_order_relaxed);
    const size_t write = mWritePos.load(std::memory_order_acquire);

    const size_t available = write - read;
    const size_t amount = std::min(count, available);

    for (size_t i = 0; i < amount; ++i) {
      output[i] = mBuffer[(read + i) % mBuffer.size()];
    }

    mReadPos.store(read + amount, std::memory_order_release);

    return amount;
  }

  size_t Size() const { return mWritePos.load(std::memory_order_acquire) - mReadPos.load(std::memory_order_acquire); }

  constexpr size_t Capacity() const { return mBuffer.size(); }

 private:
  std::vector<T> mBuffer{};

  // Producer and consumer modify different atomics.
  alignas(64) std::atomic<size_t> mWritePos{0};
  alignas(64) std::atomic<size_t> mReadPos{0};
};