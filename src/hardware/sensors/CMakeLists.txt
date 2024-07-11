add_library(sensors INTERFACE)

target_sources(sensors INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/adxl345.c
)

target_include_directories(sensors INTERFACE ${CMAKE_CURRENT_LIST_DIR})

target_link_libraries(sensors INTERFACE pico_stdlib)
