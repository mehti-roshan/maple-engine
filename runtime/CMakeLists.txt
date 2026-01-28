add_executable(game main.cpp)
target_link_libraries(game maple_engine)

add_custom_target(copy_data
	COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different 
        "${CMAKE_CURRENT_SOURCE_DIR}/assets/"
        "$<TARGET_FILE_DIR:game>/assets"
    COMMENT "Copying data"
)

add_dependencies(game copy_data)
