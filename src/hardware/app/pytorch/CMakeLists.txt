add_executable(pytorch_app)

file(GLOB_RECURSE SRC 
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)
target_sources(pytorch_app PRIVATE ${SRC})
target_include_directories(pytorch_app PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(pytorch_app
    hardware_setup
    sensor_lib
    led_lib
    ml_utils
    pico_stdlib
    pico-tflmicro
)

pico_enable_stdio_usb(pytorch_app 1)
pico_enable_stdio_uart(pytorch_app 0)
pico_add_extra_outputs(pytorch_app)