from pathlib import Path
from typing import cast

import torch
from torch.export import export
from executorch.exir import to_edge

from ml_dev.environment import (
    DATA_ROOT,
    PT_MODEL_WEIGHTS_FILE,
    PT_EXECUTORCH_MODEL_FILE,
    SAMPLE_SHAPE,
)
from ml_dev.pytorch.gesture_cnn_model import GestureCnnModel
from ml_dev.pytorch.gesture_dataset import GestureDataset
from ml_dev.pytorch.persistence import load_weights
from ml_dev.pytorch.preprocessing import preprocess


def main() -> None:
    calibration_samples = _load_calibration_samples()
    model_fp32 = _load_model()
    model_int8 = _quantize(model_fp32, calibration_samples)

    dummy_inputs = torch.ones(*SAMPLE_SHAPE[::-1])
    aten_dialect = export(model_int8, (dummy_inputs,))
    edge_program = to_edge(aten_dialect)
    executorch_program = edge_program.to_executorch()
    with PT_EXECUTORCH_MODEL_FILE.open("wb") as out_file:
        out_file.write(executorch_program.buffer)


def _load_model() -> GestureCnnModel:
    model = GestureCnnModel()
    load_weights(model, PT_MODEL_WEIGHTS_FILE)
    return model


def _load_calibration_samples() -> torch.Tensor:
    ds = GestureDataset(
        data_root=DATA_ROOT,
        training=True,
        window_size=SAMPLE_SHAPE[0],
        stride=1,
        transform_samples=preprocess,
    )
    return ds[:][0]


def _quantize(
    model: torch.nn.Module, calibration_samples: torch.Tensor
) -> torch.nn.Module:
    torch.backends.quantized.engine = "qnnpack"
    model_fp32 = model.eval()

    torch.backends.quantized.engine = "qnnpack"
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")

    model_fp32 = torch.ao.quantization.fuse_modules(
        model=model_fp32,
        modules_to_fuse=[
            [f"{block}.{layer}" for layer in ["pw_conv", "batchnorm"]]
            for block in ["conv0", "conv1"]
        ],
    )
    model_fp32 = torch.ao.quantization.prepare(model_fp32)
    model_fp32(calibration_samples)
    model_int8 = torch.ao.quantization.convert(model_fp32)

    return model_int8


if __name__ == "__main__":
    main()
