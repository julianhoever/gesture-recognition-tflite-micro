from collections.abc import Callable
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset

from ml_dev.gesture_dataset import (
    LABEL_NAMES,
    load_gesture_data as _load_raw_gesture_dataset,
)

Transformation = Callable[[torch.Tensor], torch.Tensor]


class GestureDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        training: bool,
        window_size: int = 125,
        stride: int = 1,
        transform_samples: Optional[Transformation] = None,
    ) -> None:
        super().__init__()
        raw_samples, raw_labels = _load_raw_gesture_dataset(
            data_root, training, window_size, stride
        )
        self._samples = torch.tensor(raw_samples)
        self._labels = torch.tensor(raw_labels)

        self._samples = torch.transpose(self._samples, -2, -1)

        if transform_samples is not None:
            self._samples = transform_samples(self._samples)

    def __len__(self) -> int:
        return len(self._labels)

    def __getitem__(self, idx: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return self._samples[idx], self._labels[idx]
