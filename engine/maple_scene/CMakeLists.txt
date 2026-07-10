include(FetchContent)

FetchContent_Declare(
  entt
  GIT_REPOSITORY https://github.com/skypjack/entt.git
  GIT_TAG 1333fa53129e7cfded5a9640c4336a254049917b
)

FetchContent_MakeAvailable(entt)

add_library(maple_scene INTERFACE)

target_include_directories(maple_scene INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(maple_scene INTERFACE EnTT::EnTT)