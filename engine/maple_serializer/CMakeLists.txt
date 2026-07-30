add_library(maple_serializer STATIC bit_writer.cpp bit_reader.cpp write_stream.cpp read_stream.cpp utils.cpp)
target_include_directories(maple_core PUBLIC ${CMAKE_SOURCE_DIR})