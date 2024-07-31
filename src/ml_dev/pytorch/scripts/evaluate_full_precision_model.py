from functools import partial

import torch
from torch.utils.data import Dataset

from ml_dev.pytorch.gesture_cnn_model import GestureCnnModel
from ml_dev.pytorch.gesture_dataset import GestureDataset
from ml_dev.pytorch.preprocessing import preprocess
from ml_dev.pytorch.persistence import load_weights
from ml_dev.environment import (
    DATA_ROOT,
    PT_MODEL_WEIGHTS_FILE,
    SAMPLE_SHAPE,
    PT_DEVICE,
)


def main() -> None:
    model = _load_model()
    ds_train, ds_test = _get_datasets()

    print(f"Train Accuracy: {_compute_accuracy(model, ds_train):.04f}")
    print(f"Validation Accuracy: {_compute_accuracy(model, ds_test):.04f}")


def _load_model() -> GestureCnnModel:
    model = GestureCnnModel()
    load_weights(model, PT_MODEL_WEIGHTS_FILE)
    return model


def _get_datasets() -> tuple[GestureDataset, GestureDataset]:
    gesture_ds = partial(
        GestureDataset,
        data_root=DATA_ROOT,
        window_size=SAMPLE_SHAPE[0],
        stride=1,
        transform_samples=preprocess,
    )
    return gesture_ds(training=True), gesture_ds(training=False)


def _compute_accuracy(model: torch.nn.Module, ds: Dataset) -> float:
    model.to(PT_DEVICE)
    model.eval()

    samples, labels = ds[:]
    samples = samples.to(PT_DEVICE)
    labels = labels.to(PT_DEVICE)

    predictions = torch.argmax(model(samples), dim=-1)
    return float(torch.sum(predictions == labels) / len(labels))


if __name__ == "__main__":
    main()
