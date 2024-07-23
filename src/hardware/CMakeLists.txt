cmake_minimum_required(VERSION 3.13)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED On)

# set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --specs=nosys.specs")

set(PICO_SDK_FETCH_FROM_GIT Off)
set(PICO_SDK_PATH ${CMAKE_SOURCE_DIR}/third_party/pico-sdk)

include(pico_sdk_import.cmake)

project(gesture_recognition)

pico_sdk_init()

# ExecuTorch parameters
# set(EXECUTORCH_BUILD_ARM_BAREMETAL On)
# set(EXECUTORCH_BUILD_EXTENSION_RUNNER_UTIL ON)
# set(OPTIMIZE_SIZE Off)

add_subdirectory(sensors)
add_subdirectory(setup)
add_subdirectory(app)
add_subdirectory(third_party)
