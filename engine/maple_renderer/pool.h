// pool.h
#pragma once
#include <algorithm>
#include <cstdint>
#include <vector>

template <typename T>
class Pool {
 public:
  using Handle = uint32_t;

  // Add a new object (moved), returns a handle
  Handle Add(T&& obj) {
    // Search for a free slot
    for (size_t i = 0; i < mFree.size(); ++i) {
      if (mFree[i]) {
        mData[i] = std::move(obj);
        mFree[i] = false;
        return static_cast<Handle>(i);
      }
    }

    // No free slot → append
    Handle handle = static_cast<Handle>(mData.size());
    mData.push_back(std::move(obj));
    mFree.push_back(false);
    return handle;
  }

  // Release the object at 'handle' (calls its destructor and marks slot free)
  void Remove(Handle handle) {
    assert(handle < mData.size() && !mFree[handle]);  // must be in use
    mData[handle] = T();                              // reset to default – releases resources
    mFree[handle] = true;
  }

  // Check if a handle is currently in use
  bool IsValid(Handle handle) const { return handle < mData.size() && !mFree[handle]; }

  // Direct access to the stored object (use only after IsValid check)
  T& Get(Handle handle) {
    assert(IsValid(handle));
    return mData[handle];
  }
  const T& Get(Handle handle) const {
    assert(IsValid(handle));
    return mData[handle];
  }

  // Number of active objects (not the capacity)
  size_t ActiveCount() const { return mData.size() - std::count(mFree.begin(), mFree.end(), true); }

  // Total capacity (including freed slots)
  size_t Capacity() const { return mData.size(); }

 private:
  std::vector<T> mData;     // contiguous storage
  std::vector<bool> mFree;  // true = slot is free
};