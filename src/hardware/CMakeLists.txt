cmake_minimum_required(VERSION 3.13)

# initialize pico-sdk from GIT
# (note this can come from environment, CMake cache etc)
set(PICO_SDK_FETCH_FROM_GIT on)

# pico_sdk_import.cmake is a single file copied from this SDK
# note: this must happen before project()
include(pico_sdk_import.cmake)

project(gesture_recognition)

# initialize the Raspberry Pi Pico SDK
pico_sdk_init()

if (TARGET tinyusb_device)

    add_subdirectory(sensors)
    add_subdirectory(setup)
    # add_subdirectory(pico-tflmicro)

    add_executable(test_adxl345 app/test_adxl345.cpp)
    target_include_directories(test_adxl345 PUBLIC .)
    target_link_libraries(test_adxl345 pico_stdlib hardware_setup sensors)

    # enable usb output, disable uart output
    pico_enable_stdio_usb(test_adxl345 1)
    pico_enable_stdio_uart(test_adxl345 0)

    # create map/bin/hex/uf2 file etc.
    pico_add_extra_outputs(test_adxl345)

elseif(PICO_ON_DEVICE)
    message(WARNING "not building hello_usb because TinyUSB submodule is not initialized in the SDK")
endif()
