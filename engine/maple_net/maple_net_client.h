#pragma once

#include <functional>
#include <memory>
#include <string>

#include "maple_net_base.h"

namespace maple {

class NetClient : NetBase {
 public:
  enum class ConnectionState {
    None = 0,  /// used to indicate an error condition, connection doesn't exist or has already been closed.
    Disconnected,
    Connecting,
    Connected,
    Failed,
  };

  using PacketProcessFunc = std::function<void(const void* data, uint32_t size, int64_t receivedAtMicros)>;

 public:
  /// attempts to connect to ip and port
  /// @param address string address of ip and port separated by a colon
  /// @param error if initialization fails, will contain an error message
  /// @returns if the actual connection request went out or not (or the entire network stack failed), connection status still needs to be checked
  bool TryConnect(const std::string& address, std::string& error);

  /// updates internal state, must be called each loop
  void Update() const;

  /// disconnects and closes the connection
  /// any unread data of the connection is discarded
  /// @param reason application defined reason for the disconnect (may be received on the other side)
  /// @param linger if true will attempt to send remaining data to client before disconnect, otherwise it's lost
  void Disconnect(int32_t reason = 0, bool linger = false);

  /// get the queued packets
  /// @param func the callback function called for each packet
  void ReceiveMessages(PacketProcessFunc func);

  bool Send(const void* data, uint32_t size, uint32_t flags = SendFlags::Unreliable);

  ConnectionState GetConnectionState() const;
  bool IsConnected() const { return GetConnectionState() == ConnectionState::Connected; };

  NetClient();
  ~NetClient();
  NetClient(NetClient&&) noexcept;
  NetClient& operator=(NetClient&&) noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}  // namespace maple