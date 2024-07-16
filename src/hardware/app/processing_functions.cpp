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


float* channelwiseVariance(
        const float values[],
        const uint32_t length,
        const uint32_t channels) {

    const float* means = channelwiseMean(values, length, channels);
    float* variances = new float[channels] { 0.0f };
    
    for (uint32_t chIdx = 0; chIdx < channels; chIdx++) {
        for (uint32_t valIdx = 0; valIdx < length; valIdx += chIdx + 1) {
            variances[chIdx] += pow(values[valIdx] - means[chIdx], 2);
        }
        variances[chIdx] /= length / channels;
    } 
    
    return variances;
}


void normalizeChannelwise(
        float values[],
        const uint32_t length,
        const uint32_t channels) {

    const float* means = channelwiseMean(values, length, channels);
    const float* variances = channelwiseVariance(values, length, channels);

    for (uint32_t chIdx = 0; chIdx < channels; chIdx++) {
        float standardDeviation = sqrt(variances[chIdx]);
        for (uint32_t valIdx = 0; valIdx < length; valIdx += chIdx + 1) {
            values[valIdx] = (values[valIdx] - means[chIdx]) / standardDeviation;
        }
    }
}


uint32_t argmax(const float values[], const uint32_t length) {
    uint32_t maxIdx = 0;
    for (uint32_t idx = 0; idx < length; idx++) {
        if (values[idx] > values[maxIdx]) {
            maxIdx = idx;
        }
    }
    return maxIdx;
}
