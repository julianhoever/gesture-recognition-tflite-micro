#include <cstdio>
#include <cstdint>

#include "pico/stdio.h"
#include "pico/stdlib.h"
#include "pico/time.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

#include "model.h"
#include "tflite_interpreter.h"
#include "signal_queue.h"
#include "processing_functions.h"
#include "led.h"

const uint32_t TENSOR_ARENA_SIZE = 1024 * 100;
const uint32_t CHANNEL_COUNT = 3;
const uint32_t INPUT_FEATURE_COUNT = CHANNEL_COUNT * 125;
const uint32_t OUTPUT_FEATURE_COUNT = 4;
const uint32_t INFERENCE_EVERY_NTH_POINTS = 20;


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


TfLiteInterpreter interpreter = getInterpreter();


void runInference(SignalQueue* queue) {
    static float inputBuffer[INPUT_FEATURE_COUNT];
    float outputBuffer[OUTPUT_FEATURE_COUNT];
    
    queue->copyToBuffer(inputBuffer);

    normalizeChannelwise(inputBuffer, INPUT_FEATURE_COUNT, CHANNEL_COUNT);
    interpreter.runInference(inputBuffer, outputBuffer);
    const uint32_t predictedClass = argmax(outputBuffer, OUTPUT_FEATURE_COUNT);

    switch (predictedClass)
    {
    case 1:
        setRgbLed(255, 0, 0);
        break;
    case 2:
        setRgbLed(0, 255, 0);
        break;
    case 3:
        setRgbLed(0, 0, 255);
        break;
    default:
        setRgbLed(0, 0, 0);
        break;
    }
}


int main() {
    stdio_init_all();

    while(getchar() != 'r') {}
    
    SignalQueue queue(INPUT_FEATURE_COUNT, CHANNEL_COUNT);
    queue.notifyOnOverflowingElement(INFERENCE_EVERY_NTH_POINTS, runInference);

    uint64_t start, stop = 0; 

    while (true) {
        start = to_us_since_boot(get_absolute_time());

        // TODO: collect measurement

        uint16_t a[] = {1, 2, 3};
        queue.add(a);
        sleep_ms(16);

        stop = to_us_since_boot(get_absolute_time());
        printf("%f\n", 1.0f/(stop-start)/1e-6);
    }

    return 0;
}