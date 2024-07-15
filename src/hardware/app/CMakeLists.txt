add_executable(gesture_app)

file(GLOB_RECURSE SRC 
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)
target_sources(gesture_app PRIVATE ${SRC})
target_include_directories(gesture_app PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(gesture_app
    pico_stdlib
    hardware_setup
    sensor_lib
    pico-tflmicro
)

pico_enable_stdio_usb(gesture_app 1)
pico_enable_stdio_uart(gesture_app 0)
pico_add_extra_outputs(gesture_app)