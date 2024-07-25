from pathlib import Path
import torch

# import torch.onnx
import torch.ao.quantization

from ml_dev.pytorch.gesture_cnn_model import GestureCnnModel
from ml_dev.pytorch.gesture_dataset import GestureDataset
from ml_dev.pytorch.preprocessing import preprocess
from ml_dev.pytorch.persistence import load_weights
from ml_dev.environment import (
    DATA_ROOT,
    PT_MODEL_WEIGHTS_FILE,
    SAMPLE_SHAPE,
    PT_ONNX_MODEL_FILE,
)


def main() -> None:
    ds_calibration = GestureDataset(
        data_root=DATA_ROOT,
        training=True,
        window_size=SAMPLE_SHAPE[0],
        stride=1,
        transform_samples=preprocess,
    )
    calibration_samples = ds_calibration[:][0]

    model_fp32 = GestureCnnModel()
    load_weights(model_fp32, PT_MODEL_WEIGHTS_FILE)

    model_int8 = _quantize(model_fp32, calibration_samples)
    _export_onnx(model_int8, PT_ONNX_MODEL_FILE)


def _quantize(
    model: torch.nn.Module, calibration_samples: torch.Tensor
) -> torch.nn.Module:
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


def _export_onnx(model: torch.nn.Module, destination: Path) -> None:
    model.eval()
    dummy_input = torch.randn(1, *SAMPLE_SHAPE[::-1])
    torch.onnx.export(
        model=model,
        args=dummy_input,
        f=str(destination),
        export_params=True,
        opset_version=19,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dict(
            input={0: "batch_size"},
            output={0: "batch_size"},
        ),
    )


if __name__ == "__main__":
    main()
