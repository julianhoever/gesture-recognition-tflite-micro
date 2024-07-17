#include <cstdint>
#include <cmath>
#include "processing_functions.h"


float* channelwiseMean(
        const float values[],
        const uint32_t length,
        const uint32_t channels) {

    float* means = new float[channels] { 0.0f };

    for (uint32_t chIdx = 0; chIdx < channels; chIdx++) {
        for (uint32_t valIdx = 0; valIdx < length; valIdx += chIdx + 1) {
            means[chIdx] += values[valIdx];
        }
        means[chIdx] /= length / channels;
    }
    
    return means;
}


void centerChannels(
        float values[],
        const uint32_t length,
        const uint32_t channels) {

    const float* const means = channelwiseMean(values, length, channels);
    
    for (uint32_t ch_idx = 0; ch_idx < channels; ch_idx++) {
        for (uint32_t val_idx = 0; val_idx < length; val_idx += ch_idx + 1) {
            values[val_idx] = values[val_idx] - means[ch_idx];
        }
    }
}


float maxAbs(const float values[], const uint32_t length) {
    float maxAbsValue = abs(values[0]);
    float currentAbsValue;
    for (uint32_t idx = 1; idx < length; idx++) {
        currentAbsValue = abs(values[idx]);
        if (currentAbsValue > maxAbsValue) {
            maxAbsValue = currentAbsValue;
        }
    }
    return maxAbsValue;
}


void rescale(float values[], const uint32_t length) {
    const float maxAbsValue = maxAbs(values, length);
    for (uint32_t idx = 0; idx < length; idx++) {
        values[idx] /= maxAbsValue;
    }
}


void preprocess(float values[], const uint32_t length, const uint32_t channels) {
    centerChannels(values, length, channels);
    rescale(values, length);
}


uint32_t argmax(const float values[], const uint32_t length) {
    uint32_t maxIdx = 0;
    for (uint32_t idx = 1; idx < length; idx++) {
        if (values[idx] > values[maxIdx]) {
            maxIdx = idx;
        }
    }
    return maxIdx;
}
