#pragma once

#include <cstdint>
namespace maple {

uint64_t ToLittleEndian(uint64_t v);
uint32_t ToLittleEndian(uint32_t v);
uint16_t ToLittleEndian(uint16_t v);

uint64_t ToHostEndian(uint64_t v);
uint32_t ToHostEndian(uint32_t v);
uint16_t ToHostEndian(uint16_t v);

uint32_t BitsRequired(uint64_t numStates);

}  // namespace maple