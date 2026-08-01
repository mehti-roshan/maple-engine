#pragma once

#include <functional>
#include <memory>

#include "maple_net_base.h"

namespace maple {
class NetServer : public NetBase {
 public:
  struct ClientID {
    uint32_t id = 0;
  };

  enum class DisconnectReason {
    ClosedByPeer,
    ProblemDetectedLocally,
  };

  /// called when a new connection request comes along
  /// function must return a bool stating whether this connected should be accepted or not
  /// WARN: even if client is accepted, it may disconnect just as we try to accept it, thus not actually get accepted
  /// in that case disconnect callback will be called
  using ClientConnectFunc = std::function<bool(ClientID clientID)>;
  /// called whenever one of our clients disconnects (for whatever reason, even if we disconnected them ourselves)
  using ClientDisconnectFunc = std::function<void(ClientID clientID, DisconnectReason reason)>;
  using PacketProcessingFunc = std::function<void(const void* data, uint32_t size, int64_t receivedAtMicros, ClientID clientID)>;

 public:
  /// start listening on a specific port, on all interfaces
  /// @param connectFunc will be called whenever a connection request arrives, and this callback must return a bool as whether to accept them or not
  /// @param disconnectFunc will be called whenever on client disconnects
  /// reasons include timeout, local errors, we kicked them ourselves or they disconnected during accept
  /// @param error if initialization fails, will contain an error message
  /// @returns if listening succeeded
  bool Listen(uint16_t port, ClientConnectFunc connectFunc, ClientDisconnectFunc disconnectFunc, std::string& error);

  /// updates internal state, must be called each loop
  void Update() const;

  void PollMessages(PacketProcessingFunc);

  bool Send(ClientID clientID, const void* data, uint32_t size, uint32_t flags = SendFlags::Unreliable);

  /// @param linger if true will attempt to send remaining data to client before disconnect, otherwise it's lost
  void DisconnectClient(ClientID clientID, bool linger = false);

 public:
  NetServer();
  ~NetServer();
  NetServer(NetServer&&) noexcept;
  NetServer& operator=(NetServer&&) noexcept;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
}  // namespace maple