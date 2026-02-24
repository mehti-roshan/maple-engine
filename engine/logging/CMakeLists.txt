find_package(spdlog REQUIRED)

add_library(maple_logging STATIC log.cpp)

target_include_directories(maple_logging PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(maple_logging PUBLIC spdlog::spdlog)