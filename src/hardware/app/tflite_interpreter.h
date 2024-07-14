#include <cstdint>
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_interpreter.h"


class TFLiteInterpreter {
    public:
        TFLiteInterpreter(const uint8_t* modelBuffer, const uint32_t tensorArenaSize, const uint32_t inputSize, const uint32_t outputSize);
        int runInference(float* const inputBuffer, float* const outputBuffer);
        int initialize();
    private:
        const uint8_t* modelBuffer;
        const uint32_t tensorArenaSize, inputSize, outputSize;
        uint8_t* tensorArena;
        const tflite::Model* model;
        tflite::MicroInterpreter* interpreter;
        TfLiteTensor* input;
        TfLiteTensor* output;
        uint8_t quantize(const float x);
        float dequantize(const uint8_t x);
};