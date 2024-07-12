#include <iostream>
#include "pico/stdlib.h"
#include "hardware_setup.h"
#include "adxl345.h"

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
                
                std::cout << "acc_x: " << xAccl;
                std::cout << "acc_y: " << yAccl;
                std::cout << "acc_z: " << zAccl;
                std::cout << std::endl;
            }
        }
    }
}