#include "hardware_setup.h"
#include "pico/stdlib.h"
#include "adxl345.h"
#include <stdio.h>

int main(void)
{
    int16_t xAccl, yAccl, zAccl;

    initializePeripherals();

    while (1)
    {
        char c = getchar_timeout_us(10000);
        if (c == 't')
        {
            if (NO_ERROR == setup_adxl345())
            {
                adxl345_readData(&xAccl, &yAccl, &zAccl);
                printf("acc_x: %d, acc_y:%d, acc_z:%d\r\n", xAccl, yAccl, zAccl);
            }
        }
    }
}
