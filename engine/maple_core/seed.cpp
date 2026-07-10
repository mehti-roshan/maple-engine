#include "seed.h"

#include <cstdint>

namespace maple {
Seed::Seed(uint64_t value) { mValue = value; };

uint64_t Seed::Value() const { return mValue; }

Seed Seed::Derive(const std::string& name) const {
  uint64_t hash = mValue;

  for (auto c : name) {
    hash ^= static_cast<uint8_t>(c);
    hash *= 1099511628211ull;  // FNV-1a prime
  }

  return Seed(hash);
};

}  // namespace maple