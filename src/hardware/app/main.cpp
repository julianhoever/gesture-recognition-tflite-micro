#include <cstdint>

#include "pico/stdlib.h"
#include "pico/stdio.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tflite_interpreter.h"
#include "model.h"

const uint32_t NUM_INPUTS = 125 * 3;
const uint32_t NUM_OUTPUTS = 4;
const uint32_t TENSOR_ARENA_SIZE = 10240;


TFLiteInterpreter setupInterpreter() {
    tflite::MicroMutableOpResolver<10> resolver;
    resolver.AddQuantize();
    resolver.AddExpandDims();
    resolver.AddConv2D();
    resolver.AddReshape();
    resolver.AddAdd();
    resolver.AddRelu();
    resolver.AddMaxPool2D();
    resolver.AddMul();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    
    TFLiteInterpreter interpreter(
        outputs_model_tflite, &resolver, TENSOR_ARENA_SIZE, NUM_INPUTS, NUM_OUTPUTS
    );
    interpreter.initialize();

    return interpreter;
}


int main() {
    stdio_init_all();

    while(getchar() != 'r') {}

    printf("### INITIALIZING BUFFERS ###\n");

    float inputBuffer[NUM_INPUTS] = {0};
    float outputBuffer[NUM_OUTPUTS] = {0};

    printf("### INITIALIZING INTERPRETER ###\n");

    TFLiteInterpreter interpreter = setupInterpreter();
    
    printf("### RUN INFERENCE ###\n");

    interpreter.runInference(inputBuffer, outputBuffer);
    
    printf("### DONE ###\n");
    
    return 0;
}
