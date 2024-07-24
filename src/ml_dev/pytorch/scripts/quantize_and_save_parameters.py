import torch
import torch.ao.quantization

from ml_dev.pytorch.gesture_cnn_model import GestureCnnModel
from ml_dev.pytorch.gesture_dataset import GestureDataset
from ml_dev.pytorch.preprocessing import preprocess
from ml_dev.pytorch.persistence import load_weights
from ml_dev.environment import (
    DATA_ROOT,
    PT_MODEL_WEIGHTS_FILE,
    SAMPLE_SHAPE,
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

    for name, module in model_int8.named_modules():
        print(f"[{name}]")


def _quantize(
    model: torch.nn.Module, calibration_samples: torch.Tensor
) -> torch.nn.Module:
    model_fp32 = model.eval()

    torch.backends.quantized.engine = "qnnpack"
    model_fp32.qconfig = torch.ao.quantization.get_default_qconfig("qnnpack")

    model_fp32 = torch.ao.quantization.prepare(model_fp32)
    model_fp32(calibration_samples)
    model_int8 = torch.ao.quantization.convert(model_fp32)

    return model_int8


if __name__ == "__main__":
    main()
