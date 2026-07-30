#include "utils.h"

#include <bit>
#include <cassert>
namespace maple {

uint64_t ToLittleEndian(uint64_t v) {
  if constexpr (std::endian::native == std::endian::little)
    return v;
  else
    return std::byteswap(v);
}

uint32_t ToLittleEndian(uint32_t v) {
  if constexpr (std::endian::native == std::endian::little)
    return v;
  else
    return std::byteswap(v);
}

uint16_t ToLittleEndian(uint16_t v) {
  if constexpr (std::endian::native == std::endian::little)
    return v;
  else
    return std::byteswap(v);
}

uint64_t ToHostEndian(uint64_t v) { return ToLittleEndian(v); }
uint32_t ToHostEndian(uint32_t v) { return ToLittleEndian(v); }
uint16_t ToHostEndian(uint16_t v) { return ToLittleEndian(v); }

uint32_t BitsRequired(uint64_t numStates) {
  assert(numStates > 0);
  return 64 - std::countl_zero(numStates);
}

}  // namespace maple