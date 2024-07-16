#ifndef PREPROCESSING_FUNCTIONS_H
#define PREPROCESSING_FUNCTIONS_H

#include <cstdint>


void normalize_per_channel(float values[], const uint32_t length, const uint32_t channels);
uint32_t argmax(const float values[], const uint32_t length);

#endif