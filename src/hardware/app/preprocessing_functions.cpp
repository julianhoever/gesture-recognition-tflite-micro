#include <cstdint>
#include <cmath>

#include "preprocessing_functions.h"


float calculate_mean(const float values[], const uint32_t length) {
    float summed_values = 0;
    for (uint32_t i = 0; i < length; i++) {
        summed_values += values[i];
    }
    return summed_values / length;
}


float calculate_variance(const float values[], const uint32_t length) {
    float mean = calculate_mean(values, length);
    float squared_errors = 0;
    for (uint32_t i = 0; i < length; i++) {
        squared_errors += pow(values[i] - mean, 2);
    }
    return squared_errors / length;
}

void normalize(float values[], const uint32_t length) {
    float mean = calculate_mean(values, length);
    float variance = calculate_variance(values, length);
    float standard_deviation = sqrt(variance);
    for (uint32_t i = 0; i < length; i++) {
        values[i] = (values[i] - mean) / standard_deviation;
    }
}
