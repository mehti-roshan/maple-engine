#include "enums.h"

namespace maple {
bool FormatIsDepth(Format format) {
  if (format == Format::D16_UNORM || format == Format::D16_UNORM_S8 || format == Format::D24_UNORM_S8 || format == Format::D32_SFLOAT ||
      format == Format::D32_SFLOAT_S8)
    return true;
  else {
    return false;
  }
}

bool FormatHasStencil(Format format) {
  if (format == Format::D16_UNORM_S8 || format == Format::D24_UNORM_S8 || format == Format::D32_SFLOAT_S8)
    return true;
  else {
    return false;
  }
}

bool FormatIsColor(Format format) { return !FormatIsDepth(format); }
}  // namespace maple