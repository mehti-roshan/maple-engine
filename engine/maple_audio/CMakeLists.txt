include(FetchContent)

# Configure Opus options before it is added
set(OPUS_BUILD_SHARED_LIBRARY OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(OPUS_BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  SDL3
  GIT_REPOSITORY https://github.com/libsdl-org/SDL.git
  GIT_TAG release-3.2.0
)

FetchContent_Declare(
  openal
  GIT_REPOSITORY https://github.com/kcat/openal-soft.git
  GIT_TAG 7f9c6de1162b1d4604b9a2f8bcd6c53021566221
)

FetchContent_Declare(
  opus
  GIT_REPOSITORY https://github.com/xiph/opus.git
  GIT_TAG v1.6.1
)
  
FetchContent_Declare(
  speexdsp
  GIT_REPOSITORY https://gitlab.xiph.org/xiph/speexdsp.git
  GIT_TAG 79d666118beae3af612a7dad74ee45acdffa7705
)

FetchContent_MakeAvailable(SDL3)
FetchContent_MakeAvailable(openal)
FetchContent_MakeAvailable(opus)
FetchContent_MakeAvailable(speexdsp)

find_package(Threads REQUIRED)

add_library(maple_audio STATIC maple_audio.cpp)

target_include_directories(maple_audio PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(maple_audio PRIVATE maple_logging SDL3::SDL3 OpenAL opus speexdsp Threads::Threads)

# Linux/BSD math + dl
if(UNIX AND NOT APPLE)
  target_link_libraries(maple_audio PRIVATE
    m
    dl
  )
endif()