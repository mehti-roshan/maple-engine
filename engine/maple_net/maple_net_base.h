#pragma once

#include <cstdint>

namespace maple {
class NetBase {
 public:
  enum SendFlags : uint32_t {
    /// regular old udp, if the underlying connection is having issues, will be buffered to send later
    Unreliable = 0,
    /// will not be buffered on nagle, however may be buffered if the connection is having issues
    NoNagle = 1,
    /// if the message cannot be sent very soon (because the connection is still doing some initial
    /// handshaking, route negotiations, etc), then just drop it.  This is only applicable for unreliable
    /// messages.  Using this flag on reliable messages is invalid.
    /// useful for voice packets for example
    NoDelay = 4,
    Reliable = 8,
  };

};
}  // namespace maple