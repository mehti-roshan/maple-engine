#pragma once
namespace maple {
#define chkvk(call, err, msg) \
  do {                        \
    if (call != VK_SUCCESS) { \
      err = msg;              \
      return false;           \
    }                         \
  } while (0)
}  // namespace maple