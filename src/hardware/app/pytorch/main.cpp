#include <cstdio>
#include <cstdint>
#include <cmath>

#include "pico/stdio.h"
#include "pico/stdlib.h"
#include "pico/time.h"

#include "signal_queue.h"
#include "led.h"
#include "hardware_setup.h"
#include "adxl345.h"

#define DEBUG_PRINT_FPS false

const uint32_t CHANNEL_COUNT = 3;
const uint32_t INPUT_FEATURE_COUNT = CHANNEL_COUNT * 125;
const uint32_t OUTPUT_FEATURE_COUNT = 4;
const uint32_t INFERENCE_EVERY_NTH_POINTS = 10;
enum TargetClasses { clsIdle, clsSnake, clsUpDown, clsWave, clsUndefined };



void displayPredictedClass(float* predictions) {
    if (predictions[clsIdle] >= 0.7) {
        setRgbLed(0, 0, 0);
    }
    else {
        uint8_t level_red = ceil((255 * predictions[clsSnake]));
        uint8_t level_green = ceil(255 * predictions[clsUpDown]);
        uint8_t level_blue = ceil(255 * predictions[clsWave]);
        setRgbLed(level_red, level_green, level_blue);
    }
}


void runInference(SignalQueue* queue) {
    float predictions[] = {0.1, 0.2, 0.6, 0.1};
    displayPredictedClass(predictions);
}


int main() {
    stdio_init_all();
    initializePeripherals();
    setup_adxl345();

    SignalQueue queue(INPUT_FEATURE_COUNT, CHANNEL_COUNT);
    queue.notifyOnOverflowingElement(INFERENCE_EVERY_NTH_POINTS, runInference);

    int16_t accel[CHANNEL_COUNT];

#if DEBUG_PRINT_FPS
    uint64_t current_time, previous_time;
    previous_time = 0;
#endif

    while (true) {
#if DEBUG_PRINT_FPS
        current_time = to_us_since_boot(get_absolute_time());
        printf("FPS: %f\n", 1.0f / (current_time - previous_time) / 1e-6);
        previous_time = current_time;
#endif

        adxl345_readData(&accel[0], &accel[1], &accel[2]);
        queue.add(accel);
        sleep_ms(15);
    }

    return 0;
}