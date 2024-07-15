#include <cstdint>

#include "pico/stdio.h"
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tflite_interpreter.h"


TFLiteInterpreter::TFLiteInterpreter(
    const uint8_t* modelBuffer,
    const tflite::MicroOpResolver* resolver,
    const uint32_t tensorArenaSize
) : modelBuffer(modelBuffer),
    resolver(resolver),
    tensorArenaSize(tensorArenaSize),
    tensorArena(new uint8_t[tensorArenaSize] { 0 }) { }


int TFLiteInterpreter::initialize() {
    tflite::InitializeTarget();

    this->model = tflite::GetModel(this->modelBuffer);
    if (this->model->version() != TFLITE_SCHEMA_VERSION) {
        printf(
            "Model provided is schema version %d not equal "
            "to supported version %d.\n",
            this->model->version(), TFLITE_SCHEMA_VERSION);
        return -1;
    }

    static tflite::MicroInterpreter staticInterpreter(
        this->model, *this->resolver, this->tensorArena, this->tensorArenaSize
    );
    this->interpreter = &staticInterpreter;
    TfLiteStatus allocateStatus = this->interpreter->AllocateTensors();
    if (allocateStatus != kTfLiteOk) {
        printf("AllocateTensors() failed (Code: %d)\n", allocateStatus);
        return -3;
    }

    this->input = interpreter->input(0);
    this->output = interpreter->output(0);

    const TfLiteType inputType = this->input->type;
    if (inputType != kTfLiteUInt8) {
        printf("Expect uint8 input type. Actual: TfLiteType == %d\n", inputType);
    }
    const TfLiteType outputType = this->output->type;
    if (outputType != kTfLiteUInt8) {
        printf("Expect uint8 output type. Actual: TfLiteType == %d\n", outputType);
    }

    this->inputFeatureCount = this->input->bytes;
    this->outputFeatureCount = this->output->bytes;

    this->initialized = true;

    return 0;
}


int TFLiteInterpreter::runInference(float* const inputBuffer, float* const outputBuffer) {
    if (!initialized) {
        printf("Uninitialized interpeter");
        return -1;
    }

    printf("Quantize inputs\n");
    for (uint32_t inputIdx = 0; inputIdx < this->inputFeatureCount; inputIdx++) {
        const float x = inputBuffer[inputIdx];
        this->input->data.uint8[inputIdx] = this->quantize(x);
    }

    printf("Invoke interpreter\n");

    TfLiteStatus invokeStatus = this->interpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
        printf("Invoke failed\n");
        return -1;
    }

    printf("Dequantize outputs\n");

    for (uint32_t outputIdx = 0; outputIdx < this->outputFeatureCount; outputIdx++) {
        const uint8_t quant_y = this->output->data.uint8[outputIdx];
        outputBuffer[outputIdx] = dequantize(quant_y);
    }

    printf("Done\n");

    return 0;
}


uint8_t TFLiteInterpreter::quantize(const float x) {
    return x / this->input->params.scale + this->input->params.zero_point;
}


float TFLiteInterpreter::dequantize(const uint8_t x) {
    return (x - this->output->params.zero_point) * output->params.scale;
}
