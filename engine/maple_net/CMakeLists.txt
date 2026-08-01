include(FetchContent)

FetchContent_Declare(
  GameNetworkingSockets
  GIT_REPOSITORY https://github.com/ValveSoftware/GameNetworkingSockets.git
  GIT_TAG v1.6.0
)

FetchContent_MakeAvailable(GameNetworkingSockets)

add_library(maple_net STATIC maple_net_client.cpp maple_net_server.cpp)
target_include_directories(maple_net PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(maple_net maple_logging GameNetworkingSockets::GameNetworkingSockets)