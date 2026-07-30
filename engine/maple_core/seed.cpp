#include "seed.h"

#include <cstdint>

namespace maple {
Seed::Seed(uint64_t value) { mValue = value; };

uint64_t Seed::Value() const { return mValue; }

uint64_t Seed::Mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xbf58476d1ce4e5b9ULL;
  x ^= x >> 27;
  x *= 0x94d049bb133111ebULL;
  x ^= x >> 31;
  return x;
}

Seed Seed::Derive(const std::string& name) const {
  uint64_t hash = mValue;

  for (auto c : name) {
    hash ^= static_cast<uint8_t>(c);
    hash *= 1099511628211ull;  // FNV-1a prime
  }

  return Seed(Mix64(hash));
};

}  // namespace maple