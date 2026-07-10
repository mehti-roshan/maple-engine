#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace maple {
std::vector<uint8_t> compileSlangToSpirv(const std::string& code,
                                         const std::string& fileName,
                                         const std::string& vertEntryFuncName,
                                         const std::string& fragEntryFuncName);
}  // namespace maple