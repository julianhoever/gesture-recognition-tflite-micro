#ifndef TFLITE_INTERPRETER_H
#define TFLITE_INTERPRETER_H

#include <cstdint>

#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"

class TFLiteInterpreter {
public:
    TFLiteInterpreter(
        const uint8_t *modelBuffer,
        const tflite::MicroOpResolver *resolver,
        const uint32_t tensorArenaSize);
    int runInference(float *const inputBuffer, float *const outputBuffer);
    int initialize();

private:
    bool initialized = false;
    const uint8_t *modelBuffer;
    const uint32_t tensorArenaSize;
    uint32_t inputFeatureCount, outputFeatureCount;
    uint8_t *tensorArena;
    const tflite::Model *model;
    const tflite::MicroOpResolver *resolver;
    tflite::MicroInterpreter *interpreter;
    TfLiteTensor *input;
    TfLiteTensor *output;
    uint8_t quantize(const float x);
    float dequantize(const uint8_t x);
};

#endif