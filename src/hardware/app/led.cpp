#include <cstdint>
#include "hardware/gpio.h"
#include "led.h"

const uint LED_RED = 18;
const uint LED_GREEN = 19;
const uint LED_BLUE = 20;

bool ledInitialized = false;

void setRgbLed(bool red, bool green, bool blue) {
    if (!ledInitialized) {
        gpio_init(LED_RED);
        gpio_init(LED_GREEN);
        gpio_init(LED_BLUE);

        gpio_set_dir(LED_RED, GPIO_OUT);
        gpio_set_dir(LED_GREEN, GPIO_OUT);
        gpio_set_dir(LED_BLUE, GPIO_OUT);

        ledInitialized = true;
    }

    gpio_put(LED_RED, !red);
    gpio_put(LED_GREEN, !green);
    gpio_put(LED_BLUE, !blue);
}
