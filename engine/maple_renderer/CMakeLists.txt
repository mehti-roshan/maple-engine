add_library(maple_renderer STATIC maple_renderer.cpp instance.cpp device.cpp vk_renderer_ctx.cpp enums.cpp shader_compilation.cpp third_party/VulkanMemoryAllocator/vk_mem_alloc.cpp)

target_compile_definitions(maple_renderer PRIVATE VULKAN_HPP_NO_STRUCT_CONSTRUCTORS)

target_link_directories(maple_renderer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/slang/lib)
target_include_directories(maple_renderer PUBLIC ${CMAKE_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/third_party/slang/include ${CMAKE_CURRENT_SOURCE_DIR}/third_party/VulkanMemoryAllocator)
target_link_libraries(maple_renderer maple_logging vulkan slang)