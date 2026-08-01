#include "maple_net_server.h"

#include <array>

#include "log_macros.h"
#include "steam/steamnetworkingsockets.h"
#include "steam/steamnetworkingtypes.h"

namespace maple {

static constexpr size_t MaxMessagesPerUpdate = 512;

static NetServer* sServerPtr = nullptr;
static ISteamNetworkingSockets* interface() { return SteamNetworkingSockets(); }

struct NetServer::Impl {
  bool mNetworkingInit = false;
  HSteamListenSocket mListen = k_HSteamListenSocket_Invalid;
  HSteamNetPollGroup mPollGroup = k_HSteamNetPollGroup_Invalid;

  ClientConnectFunc mClientConnectFunc;
  ClientDisconnectFunc mClientDisconnectFunc;

  static void OnConnectionStatusChanged(SteamNetConnectionStatusChangedCallback_t* info) {
    if (!sServerPtr) return;
    MAPLE_DEBUG("connection status changed");

    switch (info->m_info.m_eState) {
      case k_ESteamNetworkingConnectionState_Connecting: {
        bool accept = sServerPtr->impl->mClientConnectFunc({info->m_hConn});

        if (!accept) {
          interface()->CloseConnection(info->m_hConn, 0, nullptr, false);
          break;
        }

        // Try to accept the connection.
        if (interface()->AcceptConnection(info->m_hConn) != k_EResultOK) {
          // This could fail. If the remote host tried to connect, but then
          // disconnected, the connection may already be half closed.
          interface()->CloseConnection(info->m_hConn, 0, nullptr, false);
          break;
        }

        // Assign the poll group
        if (!interface()->SetConnectionPollGroup(info->m_hConn, sServerPtr->impl->mPollGroup)) {
          interface()->CloseConnection(info->m_hConn, 0, nullptr, false);
          break;
        }

        break;
      }
      case k_ESteamNetworkingConnectionState_ClosedByPeer:
      case k_ESteamNetworkingConnectionState_ProblemDetectedLocally: {
        DisconnectReason reason = info->m_info.m_eState == k_ESteamNetworkingConnectionState_ProblemDetectedLocally
          ? DisconnectReason::ProblemDetectedLocally
          : DisconnectReason::ClosedByPeer;
        sServerPtr->impl->mClientDisconnectFunc({info->m_hConn}, reason);

        // Clean up the connection.
        interface()->CloseConnection(info->m_hConn, 0, nullptr, false);
        break;
      }
      case k_ESteamNetworkingConnectionState_None:
        //  will get callbacks here when we destroy connections, can be ignored
        break;
      case k_ESteamNetworkingConnectionState_Connected:
        // We will get a callback immediately after accepting the connection.
        // Since we are the server, we can ignore this, it's not news to us.
        break;
      default:
        break;
    }
  }
};

NetServer::NetServer(NetServer&&) noexcept = default;
NetServer& NetServer::operator=(NetServer&&) noexcept = default;

NetServer::NetServer() : impl(std::make_unique<Impl>()) {};
NetServer::~NetServer() {
  if (impl->mPollGroup != k_HSteamNetPollGroup_Invalid) interface()->DestroyPollGroup(impl->mPollGroup);
  if (impl->mListen != k_HSteamListenSocket_Invalid) interface()->CloseListenSocket(impl->mListen);
  if (impl->mNetworkingInit) GameNetworkingSockets_Kill();

  sServerPtr = nullptr;
};

bool NetServer::Listen(uint16_t port, ClientConnectFunc clientConnectFunc, ClientDisconnectFunc clientDisconnectFunc, std::string& error) {
  if (!impl->mNetworkingInit) {
    SteamNetworkingErrMsg errMsg;
    if (!GameNetworkingSockets_Init(nullptr, errMsg)) {
      error = errMsg;
      return false;
    }

    impl->mNetworkingInit = true;
  }

  impl->mClientConnectFunc = clientConnectFunc;
  impl->mClientDisconnectFunc = clientDisconnectFunc;

  sServerPtr = this;

  SteamNetworkingIPAddr localAddr;
  localAddr.Clear();
  localAddr.m_port = port;

  std::array<SteamNetworkingConfigValue_t, 1> options;
  options[0].SetPtr(k_ESteamNetworkingConfig_Callback_ConnectionStatusChanged, reinterpret_cast<void*>(Impl::OnConnectionStatusChanged));

  impl->mListen = interface()->CreateListenSocketIP(localAddr, options.size(), options.data());
  if (impl->mListen == k_HSteamNetConnection_Invalid) return false;

  impl->mPollGroup = interface()->CreatePollGroup();

  return true;
}

void NetServer::Update() const { interface()->RunCallbacks(); }

void NetServer::PollMessages(PacketProcessingFunc func) {
  std::array<SteamNetworkingMessage_t*, MaxMessagesPerUpdate> messages;
  int numMessages = interface()->ReceiveMessagesOnPollGroup(impl->mPollGroup, messages.data(), messages.size());

  for (uint32_t i = 0; i < numMessages; i++) {
    func(messages[i]->GetData(), messages[i]->GetSize(), messages[i]->GetTimeReceived(), {messages[i]->m_conn});
    messages[i]->Release();
  }
}

void NetServer::DisconnectClient(ClientID clientID, bool linger) { interface()->CloseConnection(clientID.id, 0, nullptr, linger); }

}  // namespace maple