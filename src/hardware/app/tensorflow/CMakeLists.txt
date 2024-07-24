add_executable(tensorflow_app)

file(GLOB_RECURSE SRC 
    ${CMAKE_CURRENT_LIST_DIR}/*.c
    ${CMAKE_CURRENT_LIST_DIR}/*.cpp
    ${CMAKE_CURRENT_LIST_DIR}/*.h
    ${CMAKE_CURRENT_LIST_DIR}/*.hpp
)
target_sources(tensorflow_app PRIVATE ${SRC})
target_include_directories(tensorflow_app PRIVATE ${CMAKE_CURRENT_LIST_DIR})
target_link_libraries(tensorflow_app
    hardware_setup
    sensor_lib
    led_lib
    ml_utils
    pico_stdlib
    hardware_pwm
    pico-tflmicro
)

pico_enable_stdio_usb(tensorflow_app 1)
pico_enable_stdio_uart(tensorflow_app 0)
pico_add_extra_outputs(tensorflow_app)