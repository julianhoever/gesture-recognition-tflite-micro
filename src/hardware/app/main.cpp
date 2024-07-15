#include <stdio.h>
#include <stdint.h>

#include "pico/stdio.h"

#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tflite_interpreter.h"
#include "model.h"

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

    printf("### INITIALIZING BUFFERS ###\n");
    float inputBuffer[375] = {0.0f};
    float outputBuffer[4] = {0.0f};

    printf("### CREATE INTERPRETER ###\n");
    TfLiteInterpreter interpreter = getInterpreter();

    printf("### RUN INFERENCE ###\n");
    interpreter.runInference(inputBuffer, outputBuffer);

    return 0;
}