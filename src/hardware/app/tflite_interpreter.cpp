#include <cstdint>
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tflite_interpreter.h"


TFLiteInterpreter::TFLiteInterpreter(
    const uint8_t* modelBuffer, const uint32_t tensorArenaSize, const uint32_t inputSize, const uint32_t outputSize
) : modelBuffer(modelBuffer),
    tensorArenaSize(tensorArenaSize),
    inputSize(inputSize),
    outputSize(outputSize),
    tensorArena(new uint8_t[tensorArenaSize] { 0 }) { }


int TFLiteInterpreter::initialize() {
    tflite::InitializeTarget();

    this->model = tflite::GetModel(this->modelBuffer);
    if (this->model->version() != TFLITE_SCHEMA_VERSION) {
        MicroPrintf(
            "Model provided is schema version %d not equal "
            "to supported version %d.\n",
            this->model->version(), TFLITE_SCHEMA_VERSION
        );
        return -1;
    }

    static tflite::MicroMutableOpResolver<1> resolver;
    TfLiteStatus resolveStatus = resolver.AddFullyConnected();
    if (resolveStatus != kTfLiteOk) {
        MicroPrintf("Op resolution failed");
        return -2;
    }

    static tflite::MicroInterpreter staticInterpreter(
        this->model, resolver, this->tensorArena, this->tensorArenaSize
    );
    this->interpreter = &staticInterpreter;
    TfLiteStatus allocateStatus = this->interpreter->AllocateTensors();
    if (allocateStatus != kTfLiteOk) {
        MicroPrintf("AllocateTensors() failed");
        return -3;
    }

    this->input = interpreter->input(0);
    this->output = interpreter->output(0);

    return 0;
}


int TFLiteInterpreter::runInference(float* const inputBuffer, float* const outputBuffer) {
    for (uint32_t inputIdx = 0; inputIdx < this->inputSize; inputIdx++) {
        const float x = inputBuffer[inputIdx];
        this->input->data.uint8[inputIdx] = this->quantize(x);
    }

    TfLiteStatus invokeStatus = this->interpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
        MicroPrintf("Invoke failed");
        return -1;
    }

    for (uint32_t outputIdx = 0; outputIdx < this->outputSize; outputIdx++) {
        const uint8_t quant_y = this->output->data.uint8[outputIdx];
        outputBuffer[outputIdx] = dequantize(quant_y);
    }

    return 0;
}


uint8_t TFLiteInterpreter::quantize(const float x) {
    return x / this->input->params.scale + this->input->params.zero_point;
}


float TFLiteInterpreter::dequantize(const uint8_t x) {
    return (x - this->output->params.zero_point) * output->params.scale;
}
