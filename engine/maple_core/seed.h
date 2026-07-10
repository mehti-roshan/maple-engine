#pragma once

#include <cstdint>
#include <string>
namespace maple {

class Seed {
 public:
  explicit Seed(uint64_t value);

  uint64_t Value() const;

  Seed Derive(const std::string& name) const;

 private:
  uint64_t mValue = 0;
};
}  // namespace maple