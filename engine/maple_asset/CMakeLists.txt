add_library(maple_asset STATIC maple_asset.cpp third_party/stb_image.cpp)
target_include_directories(maple_asset PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(maple_asset maple_logging)