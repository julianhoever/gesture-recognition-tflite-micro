#include <cstdio>
#include <cstdint>

#include "pico/stdio.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "preprocessing_functions.h"
#include "tflite_interpreter.h"
#include "model.h"


const uint32_t INPUT_FEATURE_COUNT = 125 * 3;
const uint32_t OUTPUT_FEATURE_COUNT = 4;
const uint32_t TENSOR_ARENA_SIZE = 1024 * 100;


TfLiteInterpreter getInterpreter() {
    tflite::MicroMutableOpResolver<11>* resolver = new tflite::MicroMutableOpResolver<11>();
    resolver->AddQuantize();
    resolver->AddExpandDims();
    resolver->AddDepthwiseConv2D();
    resolver->AddReshape();
    resolver->AddAdd();
    resolver->AddRelu();
    resolver->AddConv2D();
    resolver->AddMaxPool2D();
    resolver->AddMul();
    resolver->AddFullyConnected();
    resolver->AddSoftmax();

    TfLiteInterpreter interpreter(model_tflite, *resolver, TENSOR_ARENA_SIZE);
    interpreter.initialize();

    return interpreter;
}


int main() {
    stdio_init_all();

    while(getchar() != 'r') {}
    
    TfLiteInterpreter interpreter = getInterpreter();

    float inputBuffer[INPUT_FEATURE_COUNT] = {0.0f};
    float outputBuffer[OUTPUT_FEATURE_COUNT] = {0.0f};

    normalize(inputBuffer, INPUT_FEATURE_COUNT);

    interpreter.runInference(inputBuffer, outputBuffer);

    printf("Done\n");

    return 0;
}