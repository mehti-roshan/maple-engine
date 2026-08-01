#include "maple_net_client.h"

#include <array>

#include "steam/isteamnetworkingutils.h"
#include "steam/steamnetworkingsockets.h"
#include "steam/steamnetworkingtypes.h"

namespace maple {

static constexpr uint32_t MaxReceiveMessagesCount = 64;

struct NetClient::Impl {
  bool mNetworkingInit = false;
  HSteamNetConnection mConn = k_HSteamNetConnection_Invalid;
};

NetClient::NetClient(NetClient&&) noexcept = default;
NetClient& NetClient::operator=(NetClient&&) noexcept = default;

NetClient::NetClient() : impl(std::make_unique<Impl>()) {};
NetClient::~NetClient() {
  if (impl->mNetworkingInit) {
    GameNetworkingSockets_Kill();
  }
  Disconnect();
};

static ISteamNetworkingSockets* interface() { return SteamNetworkingSockets(); }

bool NetClient::TryConnect(const std::string& address, std::string& error) {
  if (!impl->mNetworkingInit) {
    SteamNetworkingErrMsg errMsg;
    if (!GameNetworkingSockets_Init(nullptr, errMsg)) {
      error = errMsg;
      return false;
    }

    impl->mNetworkingInit = true;
  }

  if (IsConnected()) return false;

  SteamNetworkingIPAddr serverAddress;

  if (!serverAddress.ParseString(address.c_str())) return false;
  if (serverAddress.m_port == 0) return false;

  interface()->SetConnectionUserData(impl->mConn, reinterpret_cast<int64_t>(this));
  impl->mConn = interface()->ConnectByIPAddress(serverAddress, 0, nullptr);
  if (impl->mConn == k_HSteamNetConnection_Invalid) return false;

  return true;
}

void NetClient::Update() const { interface()->RunCallbacks(); }

void NetClient::Disconnect(int32_t reason, bool linger) {
  if (!impl->mNetworkingInit) return;
  if (impl->mConn != k_HSteamNetConnection_Invalid) return;

  interface()->CloseConnection(impl->mConn, reason, "disconnecting", linger);
  impl->mConn = k_HSteamNetConnection_Invalid;
}

NetClient::ConnectionState NetClient::GetConnectionState() const {
  if (impl->mConn == k_HSteamNetConnection_Invalid) return ConnectionState::None;
  SteamNetConnectionInfo_t info;
  if (!interface()->GetConnectionInfo(impl->mConn, &info)) return ConnectionState::None;

  switch (info.m_eState) {
    case k_ESteamNetworkingConnectionState_Connecting:
      return ConnectionState::Connecting;
    case k_ESteamNetworkingConnectionState_Connected:
      return ConnectionState::Connected;
    case k_ESteamNetworkingConnectionState_ClosedByPeer:
      return ConnectionState::Disconnected;
    case k_ESteamNetworkingConnectionState_ProblemDetectedLocally:
      return ConnectionState::Failed;
    default:
      return ConnectionState::None;
  }
}

bool NetClient::Send(const void* data, uint32_t size, uint32_t flags) {
  if (!IsConnected()) return false;
  SteamNetworkingMicroseconds usec = 0;
  EResult result = interface()->SendMessageToConnection(impl->mConn, data, size, flags, &usec);
  return result == k_EResultOK;
}

void NetClient::ReceiveMessages(PacketProcessFunc func) {
  if (!IsConnected()) return;
  std::array<SteamNetworkingMessage_t*, MaxReceiveMessagesCount> messages;
  int numMessages = interface()->ReceiveMessagesOnConnection(impl->mConn, messages.data(), messages.size());

  for (int i = 0; i < numMessages; ++i) {
    SteamNetworkingMessage_t* message = messages[i];
    func(message->GetData(), message->GetSize(), message->GetTimeReceived());
    message->Release();
  }
}

}  // namespace maple