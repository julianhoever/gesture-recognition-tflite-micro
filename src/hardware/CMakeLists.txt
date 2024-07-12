cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

set(PICO_SDK_FETCH_FROM_GIT Off)
include(pico_sdk_import.cmake)

project(gesture_recognition)

pico_sdk_init()

add_subdirectory(sensors)
add_subdirectory(setup)
add_subdirectory(pico-tflmicro)
add_subdirectory(app)
