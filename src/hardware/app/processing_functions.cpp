#include <cstdint>
#include "processing_functions.h"


float* channelwiseMean(
        const float values[],
        const uint32_t length,
        const uint32_t channels) {

    float* means = new float[channels] { 0.0f };

    for (uint32_t chIdx = 0; chIdx < channels; chIdx++) {
        for (uint32_t valIdx = chIdx; valIdx < length; valIdx += channels) {
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
    
    for (uint32_t chIdx = 0; chIdx < channels; chIdx++) {
        for (uint32_t valIdx = chIdx; valIdx < length; valIdx += channels) {
            values[valIdx] = values[valIdx] - means[chIdx];
        }
    }
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
